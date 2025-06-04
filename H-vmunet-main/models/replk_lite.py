import torch
import torch.nn as nn
import torch.nn.functional as F

class RepLKLiteBlock(nn.Module):
    """RepLK-Lite Large-Kernel Block with re-parameterization"""
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.reparam = False

    def forward(self, x):
        if self.reparam:
            return self.rep(x)
        out = self.dw(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pw(out)
        return out

    def reparameterize(self):
        weight = self.dw.weight + self.pw.weight.unsqueeze(-1).unsqueeze(-1)
        self.rep = nn.Conv2d(self.dw.in_channels, self.dw.out_channels, self.dw.kernel_size,
                             padding=self.dw.padding, bias=True)
        self.rep.weight.data = weight
        self.rep.bias = nn.Parameter(torch.zeros(self.dw.out_channels))
        self.reparam = True
        # remove old layers
        del self.dw
        del self.pw
        del self.bn
