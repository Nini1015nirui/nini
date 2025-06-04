import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def forward(self, input, target):
        smooth = 1.0
        input = input.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (input * target).sum()
        d = input.sum() + target.sum()
        return 1 - (2 * intersection + smooth) / (d + smooth)

class clDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()

    def forward(self, pred, target):
        pred_s = F.max_pool2d(pred, 3, 1, 1) - F.max_pool2d(-pred, 3, 1, 1)
        target_s = F.max_pool2d(target, 3, 1, 1) - F.max_pool2d(-target, 3, 1, 1)
        return self.dice(pred, target) + self.dice(pred_s, target_s)

class cbDiceLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        denom = tp + self.beta * fp + (1 - self.beta) * fn + 1e-6
        return 1 - tp / denom


class CombinedLoss(nn.Module):
    """Combination of BCE-Dice, clDice and cbDice."""
    def __init__(self, bce_dice, use_cl=True, use_cb=True):
        super().__init__()
        self.bce_dice = bce_dice
        self.use_cl = use_cl
        self.use_cb = use_cb
        if use_cl:
            self.cl = clDiceLoss()
        if use_cb:
            self.cb = cbDiceLoss()

    def forward(self, pred, target):
        loss = self.bce_dice(pred, target)
        if self.use_cl:
            loss = loss + self.cl(pred, target)
        if self.use_cb:
            loss = loss + self.cb(pred, target)
        return loss
