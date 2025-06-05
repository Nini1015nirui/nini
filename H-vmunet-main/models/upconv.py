import torch
from torch import nn
import torch.nn.functional as F

class UpConv(nn.Module):
    """3x3 transpose convolution for upsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return F.gelu(self.conv(x))
