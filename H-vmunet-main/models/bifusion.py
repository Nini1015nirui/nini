import torch
from torch import nn
import torch.nn.functional as F

class BiFusion(nn.Module):
    """BiFusion module with soft gating"""
    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, 1)
        self.conv_y = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x, y):
        fused = self.conv_x(x) + self.conv_y(y)
        weight = self.gate(fused)
        return fused * weight
