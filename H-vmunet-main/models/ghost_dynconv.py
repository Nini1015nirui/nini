import torch
from torch import nn
import torch.nn.functional as F

class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2):
        super().__init__()
        init_channels = int(out_channels / ratio)
        self.primary = nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.cheap = nn.Conv2d(init_channels, out_channels - init_channels, 3, padding=1, groups=init_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.primary(x)
        y = torch.cat([y, self.cheap(y)], dim=1)
        return self.bn(y)

class KernelWarehouseDynConv(nn.Module):
    def __init__(self, channels, kernels=4):
        super().__init__()
        self.kernels = kernels
        self.weight = nn.Parameter(torch.randn(kernels, channels, 3, 3))
        self.attn = nn.Conv2d(channels, kernels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        attn = torch.softmax(self.attn(x), dim=1)
        outs = []
        for i in range(self.kernels):
            conv = F.conv2d(x, self.weight[i:i+1], padding=1)
            outs.append(conv * attn[:, i:i+1])
        return sum(outs)

class GhostDynConv(nn.Module):
    """Combine GhostConv and KernelWarehouseDynConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local = GhostConv(in_channels, out_channels)
        self.global_branch = KernelWarehouseDynConv(out_channels)

    def forward(self, x):
        local_feat = self.local(x)
        global_feat = self.global_branch(local_feat)
        return local_feat + global_feat
