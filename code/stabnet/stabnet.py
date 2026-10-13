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


class StabNet(nn.Module):
    def __init__(self, grid_sizes=(4, 8)):
        super().__init__()
        self.grid_sizes = grid_sizes  # 多尺度网格，如 4x4 和 8x8

        self.enc1 = ConvBlock(6, 32, stride=2)    # 1/2
        self.enc2 = ConvBlock(32, 64, stride=2)   # 1/4
        self.enc3 = ConvBlock(64, 128, stride=2)  # 1/8
        self.enc4 = ConvBlock(128, 256, stride=2) # 1/16

        self.dec3 = ConvBlock(256, 128)
        self.dec2 = ConvBlock(128, 64)

        self.heads = nn.ModuleDict({  # 每个尺度一个预测头，输出网格顶点残差位移
            str(g): nn.Conv2d(64, 2 * (g + 1) ** 2, 3, padding=1)
            for g in grid_sizes
        })

    def forward(self, x):
        e1 = self.enc1(x)  # 输入为当前帧与已稳定帧拼接 (B, 6, H, W)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))

        flows = {}
        for g in self.grid_sizes:
            f = self.heads[str(g)](d2)
            f = f.mean(dim=(2, 3))  # 全局池化得到一个网格
            # 钳制位移范围，防止训练初期发散（手持抖动的合理校正量在几十像素内）
            flows[g] = (f.view(-1, 2, g + 1, g + 1) * 8.0).clamp(-32.0, 32.0)
        return flows


def grid_warp(x, vertex_flow):
    # x: (B, 3, H, W)，vertex_flow: (B, 2, g+1, g+1) 网格顶点位移
    B, _, H, W = x.size()
    g = vertex_flow.size(2) - 1

    ys = torch.linspace(-1, 1, g + 1, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, g + 1, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid_vertices = torch.stack([gx, gy], dim=0).unsqueeze(0)  # 顶点坐标归一化到 [-1, 1]

    flow = vertex_flow.clone()
    flow[:, 0] = flow[:, 0] / (W - 1) * 2  # 像素位移转归一化坐标位移
    flow[:, 1] = flow[:, 1] / (H - 1) * 2
    warped_vertices = grid_vertices + flow  # 变形后的网格顶点

    dense = F.interpolate(warped_vertices, size=(H, W), mode='bilinear',
                          align_corners=True)  # 顶点双线性插值展开为 (H, W) 稠密网格
    grid = dense.permute(0, 2, 3, 1)         # (B, H, W, 2)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


class StabLoss(nn.Module):
    def __init__(self, w_rec=1.0, w_sta=0.5, w_dis=0.5, w_mesh=1.0):
        super().__init__()
        self.w_rec, self.w_sta, self.w_dis = w_rec, w_sta, w_dis  # 重建/时间稳定/形变约束权重
        self.w_mesh = w_mesh  # 网格位移监督权重（利用经典稳像得到的校正场）

    def forward(self, stab_frames, flows, gt_stable, prev_stab=None, gt_vertex_flow=None):
        g = max(stab_frames.keys())
        stab = stab_frames[g]  # 取最细网格的结果作为主输出

        loss = self.w_rec * F.l1_loss(stab, gt_stable)

        if prev_stab is not None:
            if isinstance(prev_stab, (list, tuple)):
                if len(prev_stab) >= 2:
                    # 时间稳定（加速度惩罚）：惩罚相邻帧运动的"变化"，抑制抖动而不压制运动本身
                    accel = (stab - prev_stab[-1]) - (prev_stab[-1] - prev_stab[-2])
                    loss = loss + self.w_sta * accel.abs().mean()
                elif len(prev_stab) == 1:
                    loss = loss + self.w_sta * F.l1_loss(stab, prev_stab[-1].detach())
            else:
                loss = loss + self.w_sta * F.l1_loss(stab, prev_stab.detach())

        if gt_vertex_flow is not None:
            # 网格位移监督：预测的顶点位移直接回归经典稳像的校正场（像素单位）
            for g_, f in flows.items():
                if g_ in gt_vertex_flow:
                    loss = loss + self.w_mesh * F.l1_loss(f, gt_vertex_flow[g_])

        for f in flows.values():
            dx = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
            dy = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
            loss = loss + self.w_dis * (dx + dy)  # 形变约束：相邻顶点位移差小，避免局部扭曲
        return loss
