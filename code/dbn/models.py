import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)   # skip connection
        x = self.conv2(self.conv1(x))
        return x


class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(15, 64, stride=2)          # 1/2
        self.enc2 = ConvBlock(64, 128, stride=2)         # 1/4
        self.enc3 = ConvBlock(128, 256, stride=2)        # 1/8
        self.enc4 = ConvBlock(256, 512, stride=2)        # 1/16
        self.dec4 = DeconvBlock(512, 256)
        self.dec3 = DeconvBlock(256, 128)
        self.dec2 = DeconvBlock(128, 64)
        self.flow_pred = nn.Conv2d(64, 8, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        flow = self.flow_pred(d2)                        # (B, 8, H/2, W/2)
        flow = F.interpolate(flow, scale_factor=2, mode='bilinear',
                             align_corners=False) * 2.0  # 上采样到全分辨率
        return flow


def flow_warp(x, flow):
    # x: (B, C, H, W)，flow: (B, 2, H, W)，flow[:,0] 为 x 方向位移，flow[:,1] 为 y 方向
    B, _, H, W = x.size()
    xx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat([xx, yy], dim=1)  # (B, 2, H, W)

    flow_n = torch.zeros_like(flow)
    flow_n[:, 0] = flow[:, 0] / (W - 1) * 2
    flow_n[:, 1] = flow[:, 1] / (H - 1) * 2

    grid = (grid + flow_n).permute(0, 2, 3, 1)  # (B, H, W, 2)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


class DeBlurNet(nn.Module):
    # 输入 5 帧模糊图像 (B, 15, H, W)，输出中间帧去模糊结果 (B, 3, H, W)
    def __init__(self):
        super().__init__()
        self.flownet = FlowNet()
        # 输入: 5帧(15) + warp后的4帧(12) = 27 通道
        self.decoder = nn.Sequential(
            ConvBlock(27, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B = x.size(0)
        frames = x.view(B, 5, 3, *x.shape[2:])   # (B, 5, 3, H, W)

        flows = self.flownet(x)                  # (B, 8, H, W)
        flows = flows.view(B, 4, 2, *x.shape[2:])  # (B, 4, 2, H, W)

        neighbors = [frames[:, i] for i in [0, 1, 3, 4]]  # f1,f2,f4,f5
        warped = [flow_warp(f, flows[:, i]) for i, f in enumerate(neighbors)]

        # 5帧原始(15ch) + 4帧warp(12ch) = 27ch（与 decoder 首层 ConvBlock(27,64) 匹配）
        dbn_in = torch.cat([x] + warped, dim=1)
        out = self.decoder(dbn_in)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


class DBNLoss(nn.Module):
    def __init__(self, flow_w=0.01):
        super().__init__()
        self.flow_w = flow_w

    def charbonnier(self, pred, gt, eps=1e-6):
        return torch.sqrt((pred - gt) ** 2 + eps).mean()

    def forward(self, pred, gt, flows=None):
        loss = self.charbonnier(pred, gt)
        if flows is not None and self.flow_w > 0:
            # 一阶梯度平滑正则
            fx = flows[:, :, :, 1:] - flows[:, :, :, :-1]
            fy = flows[:, :, 1:, :] - flows[:, :, :-1, :]
            loss = loss + self.flow_w * (fx.abs().mean() + fy.abs().mean())
        return loss
