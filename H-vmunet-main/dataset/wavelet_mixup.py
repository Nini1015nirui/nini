import numpy as np

__all__ = ['wavelet_mixup']

def wavelet_mixup(img1, img2, alpha=0.5):
    return alpha * img1 + (1 - alpha) * img2
