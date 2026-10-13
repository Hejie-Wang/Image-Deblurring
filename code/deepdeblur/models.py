# DeepDeblur 网络实现（《第二次作业》4.1 节代码，forward 补充了三个尺度的输出用于多尺度损失）
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class DeepDeblurNet(nn.Module):
    def __init__(self):
        super(DeepDeblurNet, self).__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Scale 1
        x1 = nn.functional.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        out1 = self.scale1(x1)

        # Scale 2
        x2 = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out2 = self.scale2(torch.cat([x2, nn.functional.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)], dim=1))

        # Scale 3
        out3 = self.scale3(torch.cat([x, nn.functional.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False)], dim=1))

        return out1, out2, out3
