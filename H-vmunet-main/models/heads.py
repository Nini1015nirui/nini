import torch
from torch import nn
from .ghost_dynconv import GhostDynConv
from .replk_lite import RepLKLiteBlock

class HyperSpaceHead(nn.Module):
    """Segmentation, edge and thickness heads sharing common backbone."""
    def __init__(self, in_channels, num_classes=1, thickness_spacing=1.0):
        super().__init__()
        self.shared = nn.Sequential(
            GhostDynConv(in_channels, in_channels),
            RepLKLiteBlock(in_channels)
        )
        self.seg_head = nn.Conv2d(in_channels, num_classes, 1)
        self.edge_head = nn.Conv2d(in_channels, 1, 1)
        self.thick_head = nn.Conv2d(in_channels, 1, 1)
        self.spacing = thickness_spacing

    def forward(self, x):
        feat = self.shared(x)
        seg = torch.sigmoid(self.seg_head(feat))
        edge = torch.sigmoid(self.edge_head(feat))
        thickness = self.thick_head(feat) * self.spacing
        return seg, edge, thickness
