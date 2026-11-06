"""Evaluation metrics, in the order the problem statement lists them:
PSNR, SSIM, LPIPS, NIQE, latency, noise suppression / edge preservation.

LPIPS and NIQE models load lazily (they download weights on first use).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

from .analysis import luminance

_lpips = None
_niqe = None


def psnr(gt, out):
    return float(peak_signal_noise_ratio(gt, out, data_range=1.0))


def ssim(gt, out):
    return float(structural_similarity(gt, out, channel_axis=2, data_range=1.0))


def _to_tensor(img):
    import torch
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()


def lpips(gt, out):
    """LPIPS (AlexNet), lower is better."""
    global _lpips
    import torch
    if _lpips is None:
        import lpips as _l
        _lpips = _l.LPIPS(net="alex", verbose=False).eval()
    with torch.no_grad():
        return float(_lpips(_to_tensor(gt) * 2 - 1, _to_tensor(out) * 2 - 1).item())


def niqe(out):
    """NIQE no-reference score via pyiqa, lower is better."""
    global _niqe
    import torch
    if _niqe is None:
        import pyiqa
        _niqe = pyiqa.create_metric("niqe", device="cpu")
    with torch.no_grad():
        return float(_niqe(_to_tensor(out)).item())


def noise_sigma(img):
    """Estimated residual Gaussian noise std (lower = better noise suppression)."""
    try:
        return float(np.mean(estimate_sigma(img, channel_axis=-1)))
    except (ValueError, FloatingPointError):
        return float("nan")


def grad_mag(img):
    y = luminance(img) if img.ndim == 3 else img
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def edge_correlation(gt, out):
    """Pearson correlation of gradient magnitudes (1 = edges perfectly preserved)."""
    a, b = grad_mag(gt).ravel(), grad_mag(out).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def edge_f1(gt, out, tol=1):
    """Canny edge F1 vs. ground truth with `tol`-pixel tolerance."""
    def canny(x):
        y8 = (np.clip(luminance(x), 0, 1) * 255).astype(np.uint8)
        v = np.median(y8)
        return cv2.Canny(y8, int(max(0, 0.66 * v)), int(min(255, 1.33 * v) + 1)) > 0
    eg, eo = canny(gt), canny(out)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dg, do = cv2.dilate(eg.astype(np.uint8), k) > 0, cv2.dilate(eo.astype(np.uint8), k) > 0
    p = (eo & dg).sum() / max(eo.sum(), 1)
    r = (eg & do).sum() / max(eg.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-8))


def full_reference(gt, out, with_lpips=True):
    d = {"psnr": psnr(gt, out), "ssim": ssim(gt, out)}
    if with_lpips:
        d["lpips"] = lpips(gt, out)
    d["edge_corr"] = edge_correlation(gt, out)
    d["edge_f1"] = edge_f1(gt, out)
    d["noise_sigma"] = noise_sigma(out)
    d["noise_sigma_gt"] = noise_sigma(gt)
    return d


@contextmanager
def timer(store: dict, key: str = "latency_ms"):
    t = time.perf_counter()
    yield
    store[key] = (time.perf_counter() - t) * 1000.0
