import torch
from torch import nn
import torch.nn.functional as F

class PatchifyStem(nn.Module):
    """Conv 3x3 -> BN -> GELU followed by 4x4 patchify via strided conv"""
    def __init__(self, in_channels, out_channels, patch_size=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.patch = nn.Conv2d(out_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.patch(x)
        return x
