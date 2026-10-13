import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def flow_warp(x, flow):
    # x: (B, C, H, W)，flow: (B, 2, H, W)
    B, _, H, W = x.size()
    xx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat([xx, yy], dim=1)
    flow_n = torch.zeros_like(flow)
    flow_n[:, 0] = flow[:, 0] / (W - 1) * 2  # 像素位移转归一化坐标
    flow_n[:, 1] = flow[:, 1] / (H - 1) * 2
    grid = (grid + flow_n).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


class SpyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(6, 32), ConvBlock(32, 32, stride=2), ConvBlock(32, 32),
            ConvBlock(32, 32), nn.Conv2d(32, 2, 3, 1, 1))

    def forward(self, x1, x2):
        return self.body(torch.cat([x1, x2], dim=1))  # 输出 residual flow


class FlowNet(nn.Module):
    def __init__(self, scales=(0.25, 0.5, 1.0)):
        super().__init__()
        self.scales = scales
        self.blocks = nn.ModuleList([SpyBlock() for _ in scales])  # 由粗到精三级金字塔

    def forward(self, x1, x2):
        flow = None
        for block, s in zip(self.blocks, self.scales):
            x1s = F.interpolate(x1, scale_factor=s, mode='bilinear', align_corners=False)
            x2s = F.interpolate(x2, scale_factor=s, mode='bilinear', align_corners=False)
            if flow is not None:
                x2s = flow_warp(x2s, flow)  # 用粗尺度光流预对齐
            delta = block(x1s, x2s)
            # SpyBlock 首层 stride=2，delta 在 s/2 尺度：上采样回 s 尺度并放大位移量纲
            delta = F.interpolate(delta, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
            flow = delta if flow is None else flow + delta
            if s != 1.0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False) * 2
        return flow


class WindowAttn(nn.Module):
    def __init__(self, dim, heads=4, window=8, shifted=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.window = window
        self.shifted = shifted

    def forward(self, x):
        B, T, C, H, W = x.shape
        ws = self.window
        if self.shifted:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(3, 4))
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        # reflect pad 不支持 5D：把 T 折进 batch 再做空间 padding
        xp = x.reshape(B * T, C, H, W)
        xp = F.pad(xp, (0, pw, 0, ph), mode='reflect')
        xp = xp.reshape(B, T, C, *xp.shape[2:])  # 填充到窗口整数倍
        Hp, Wp = xp.shape[3], xp.shape[4]
        nwh, nww = Hp // ws, Wp // ws
        tok = xp.view(B, T, nwh, ws, nww, ws, C).permute(0, 2, 4, 1, 3, 5, 6).reshape(B, nwh * nww, T * ws * ws, C)
        tok = self.norm(tok)
        # MHA 只接受 (N, L, E)：把窗口维合并进 batch
        Bn, Nwin, Ltok, Cd = tok.shape
        out = self.attn(tok.reshape(Bn * Nwin, Ltok, Cd), *[tok.reshape(Bn * Nwin, Ltok, Cd)] * 2)[0]
        out = out.view(B, nwh, nww, T, ws, ws, C).permute(0, 3, 6, 1, 4, 2, 5).reshape(B, T, C, Hp, Wp)
        out = out[:, :, :, :H, :W]
        if self.shifted:
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(3, 4))
        return out


class TMSA(nn.Module):
    def __init__(self, dim, heads=4, window=8, shifted=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.window = window
        self.shifted = shifted

    def forward(self, x):
        B, T, C, H, W = x.shape
        wh = ww = self.window
        if self.shifted:
            x = torch.roll(x, shifts=(-wh // 2, -ww // 2), dims=(3, 4))
        ph, pw = (wh - H % wh) % wh, (ww - W % ww) % ww
        xp = x.reshape(B * T, C, H, W)
        xp = F.pad(xp, (0, pw, 0, ph), mode='reflect')
        xp = xp.reshape(B, T, C, *xp.shape[2:])
        Hp, Wp = xp.shape[3], xp.shape[4]
        nwh, nww = Hp // wh, Wp // ww
        tok = xp.view(B, T, nwh, wh, nww, ww, C).permute(0, 2, 4, 1, 3, 5, 6).reshape(B, nwh * nww, T * wh * ww, C)
        tok = self.norm(tok)
        # MHA 只接受 (N, L, E)：把窗口维合并进 batch
        Bn, Nwin, Ltok, Cd = tok.shape
        out = self.attn(tok.reshape(Bn * Nwin, Ltok, Cd), *[tok.reshape(Bn * Nwin, Ltok, Cd)] * 2)[0]
        out = out.view(B, nwh, nww, T, wh, ww, C).permute(0, 3, 6, 1, 4, 2, 5).reshape(B, T, C, Hp, Wp)
        out = out[:, :, :, :H, :W]
        if self.shifted:
            out = torch.roll(out, shifts=(wh // 2, ww // 2), dims=(3, 4))
        return out


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        # x: (B,T,C,H,W) -> (B,T,H,W,C) 做 C 维 MLP，再逆变换回 (B,T,C,H,W)
        return self.net(self.norm(x.permute(0, 1, 3, 4, 2))).permute(0, 1, 4, 2, 3)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        self.wa = WindowAttn(dim, heads, window, shifted=False)
        self.sa = WindowAttn(dim, heads, window, shifted=True)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.wa(x)
        x = x + self.sa(x)
        x = x + self.ffn(x)
        return x


class VRTStage(nn.Module):
    def __init__(self, dim, depth=2, heads=4, window=8):
        super().__init__()
        self.tmsa = nn.ModuleList([TMSA(dim, heads, window, shifted=bool(i % 2)) for i in range(depth)])
        self.swin = nn.ModuleList([SwinBlock(dim, heads, window) for i in range(depth)])
        self.fuse = nn.Conv3d(2 * dim, dim, kernel_size=1)  # 融合原始特征与对齐特征

    def forward(self, x, flows):
        B, T, C, h, w = x.shape
        mid = T // 2
        warps, fi = [], 0
        for t in range(T):
            f = x[:, t]
            if t != mid:
                f = flow_warp(f, flows[fi])  # 光流把邻帧特征对齐到中间帧
                fi += 1
            warps.append(f)
        x = self.fuse(torch.cat([x, torch.stack(warps, dim=1)], dim=2).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        for tmsa, swin in zip(self.tmsa, self.swin):
            x = x + tmsa(x)
            x = swin(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):  # (B, C, T, h, w) -> (B, C/4, T, 2h, 2w)
        x = self.conv(x)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w)
        x = self.ps(x)
        return x.reshape(B, T, C // 4, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)


class VRT(nn.Module):
    # 输入 T 帧 (B, T, 3, H, W)，T 为奇数，输出中间帧恢复结果 (B, 3, H, W)
    def __init__(self, in_ch=3, dim=48, depth=2, heads=4, window=8):
        super().__init__()
        self.flownet = FlowNet()
        self.embed = nn.Conv3d(in_ch, dim, (1, 3, 3), padding=(0, 1, 1))
        self.stage1 = VRTStage(dim, depth, heads, window)
        self.merge = nn.Conv3d(dim, 2 * dim, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))  # 空间 1/2
        self.stage2 = VRTStage(2 * dim, depth, heads, window)
        self.up = Upsample(2 * dim)
        self.fuse3 = nn.Conv3d(3 * dim, dim, kernel_size=1)  # 融合上采样特征(2dim)与 stage1 skip(dim)
        self.stage3 = VRTStage(dim, depth, heads, window)
        self.recon = nn.Conv3d(dim, in_ch, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        B, T, _, H, W = x.shape
        mid = T // 2
        flows, flows_half = [], []
        for t in range(T):  # 邻帧 -> 中间帧光流，供两个尺度使用
            if t == mid:
                continue
            f = self.flownet(x[:, mid], x[:, t])
            flows.append(f)
            flows_half.append(F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5)

        f = self.embed(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        s1 = self.stage1(f, flows)
        s2 = self.stage2(self.merge(s1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4), flows_half)
        s2 = self.up(s2.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)   # 上采样回全分辨率
        s3 = self.stage3(self.fuse3(torch.cat([s2.permute(0, 2, 1, 3, 4), s1.permute(0, 2, 1, 3, 4)], dim=1)).permute(0, 2, 1, 3, 4), flows)
        out = self.recon(s3.permute(0, 2, 1, 3, 4))                       # (B, 3, T, H, W)
        return out[:, :, mid]


class CharbonnierLoss(nn.Module):
    def forward(self, pred, gt, eps=1e-6):
        return torch.sqrt((pred - gt) ** 2 + eps).mean()
