import torch
from torch import nn
import torch.nn.functional as F

from .patchify import PatchifyStem
from .vss_block import VSSBlock
from .ghost_dynconv import GhostDynConv
from .bifusion import BiFusion
from .replk_lite import RepLKLiteBlock
from .heads import HyperSpaceHead

class LightSegNet(nn.Module):
    """Lightweight medical image segmentation framework."""
    def __init__(self, in_channels: int = 3, base_channels: int = 16, num_classes: int = 1):
        super().__init__()
        # 1. Conv3x3->BN->GELU->Patchify
        self.stem = PatchifyStem(in_channels, base_channels)
        # 2. three Vision-Mamba blocks
        self.mamba = nn.Sequential(
            VSSBlock(base_channels),
            VSSBlock(base_channels),
            VSSBlock(base_channels),
        )
        # 3-5. GhostConv + KernelWarehouseDynConv inside
        self.enhance = GhostDynConv(base_channels, base_channels)
        self.fuse = BiFusion(base_channels)
        # 6. RepLK-Lite large kernel
        self.replk = RepLKLiteBlock(base_channels, kernel_size=17)
        # 7-9. segmentation, edge and thickness heads
        self.head = HyperSpaceHead(base_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        """Input: (B,C,H,W). Output: seg(B,1,H/4,W/4), edge(B,1,H/4,W/4), thickness(B,1,1,1)."""
        x = self.stem(x)
        x = self.mamba(x)
        x = self.enhance(x)
        x = self.fuse(x, x)
        x = self.replk(x)
        seg, edge, thick = self.head(x)
        return seg, edge, thick

def reparameterize_replk(model: nn.Module):
    """Convert all RepLK-Lite blocks to inference mode with 3x3 kernels."""
    for m in model.modules():
        if isinstance(m, RepLKLiteBlock):
            m.reparameterize()
