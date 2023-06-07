import torch
import torch.nn as nn


# https://arxiv.org/pdf/2103.02907.pdf


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        # print('x_h size: ', x_h.size()) # [B, mip, H, 1]
        # print('x_w size: ', x_w.size()) # [B, mip, H, 1]
        x_w = x_w.permute(0, 1, 3, 2)
        # print('x_w size: ', x_w.size()) # [B, mip, 1, W]
        a_h = self.conv_h(x_h).sigmoid() # [B, mip, H, 1] --> [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid() # [B, mip, 1, W] --> [B, C, 1, W]

        # 此处为矩阵点乘 等价于 torch.mul()
        # [B, C, H, W] * [B, C, H, 1] * [B, C, 1, W] = [B, C, H, W]
        out = residual * a_w * a_h

        return out