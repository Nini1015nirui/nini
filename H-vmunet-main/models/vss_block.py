import torch
from torch import nn
import torch.nn.functional as F
from .vmamba import SS2D
from .ghost_dynconv import GhostDynConv

try:
    from flash_attn.modules.mha import FlashSelfAttention
    _flash_available = True
except Exception:
    _flash_available = False

class VSSBlock(nn.Module):
    """Vision-Mamba based block with optional FlashAttention."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ssm = SS2D(d_model=dim)
        if _flash_available:
            self.attn = FlashSelfAttention()
        else:
            self.attn = None
        self.conv = GhostDynConv(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.ssm(x)
        x = x.permute(0, 3, 1, 2)
        if self.attn is not None:
            q = x.flatten(2).transpose(1, 2)
            attn_out = self.attn(q)
            x = attn_out.transpose(1, 2).view(b, c, h, w)
        x = self.conv(x)
        return x + residual
