import numpy as np
import pywt

__all__ = ['wavelet_mixup']

def wavelet_mixup(img1, img2, alpha=0.5):
    coeffs1 = pywt.dwt2(img1, 'haar')
    coeffs2 = pywt.dwt2(img2, 'haar')
    mixed = []
    for c1, c2 in zip(coeffs1, coeffs2):
        if isinstance(c1, tuple):
            mixed.append(tuple(alpha * a + (1 - alpha) * b for a, b in zip(c1, c2)))
        else:
            mixed.append(alpha * c1 + (1 - alpha) * c2)
    return pywt.idwt2(tuple(mixed), 'haar')
