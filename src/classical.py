"""Classical low-light enhancement: illumination estimation + Retinex + CLAHE +
edge-preserving denoising + colour correction, with adaptive parameters.

Main entry: `enhance_classical(img, params=None, return_intermediates=False)`.
"""
from __future__ import annotations

import cv2
import numpy as np

from .analysis import FIXED_PARAMS, adaptive_params, analyze, luminance, post_noise_eps


# ---------- building blocks ----------

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter (gray guide). Works for 2D src or per-channel 3D src."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REFLECT)

    I = guide.astype(np.float32)
    if src.ndim == 3:
        return np.stack([guided_filter(I, src[..., c], radius, eps) for c in range(src.shape[2])], -1)
    p = src.astype(np.float32)
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def estimate_illumination(img: np.ndarray, radius: int | None = None, eps: float = 1e-3) -> np.ndarray:
    """LIME-style illumination: max over RGB, refined with an edge-aware guided filter.

    Max-RGB gives an upper bound on illumination per pixel; the guided filter smooths
    it within regions but keeps it sharp at object boundaries (prevents halos).
    """
    h, w = img.shape[:2]
    radius = radius or max(2, min(h, w) // 50)
    L0 = img.max(axis=2)
    L = guided_filter(luminance(img), L0, radius, eps)
    return np.clip(L, 1e-3, 1.0)


def retinex_enhance(img: np.ndarray, L: np.ndarray, gamma: float):
    """Decompose I = R * L, brighten L with gamma, recombine.

    Returns (enhanced, reflectance, adjusted_illumination).
    """
    R = img / L[..., None]
    L_adj = np.power(L, gamma)
    out = np.clip(R * L_adj[..., None], 0.0, 1.0)
    return out, np.clip(R, 0.0, 1.0), L_adj


def apply_clahe(img: np.ndarray, clip: float, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel of Lab (keeps colours stable)."""
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)  # L in [0,100]
    l8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(l8).astype(np.float32) / 2.55
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)


def denoise_guided(img: np.ndarray, eps: float, radius: int = 3, dark_weight: np.ndarray | None = None) -> np.ndarray:
    """Edge-preserving denoise: self-guided filter per channel, guided by luma.

    If `dark_weight` (HxW in [0,1]) is given, the filtered result is blended in
    mostly where the input was dark (where noise gets amplified most).
    """
    y = luminance(img)
    filt = np.clip(guided_filter(y, img, radius, eps), 0.0, 1.0)
    if dark_weight is None:
        return filt
    w = dark_weight[..., None]
    return filt * w + img * (1.0 - w)


def denoise_bilateral(img: np.ndarray, sigma_color: float = 0.08, d: int = 7) -> np.ndarray:
    return cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, d * 1.5)


def gray_world_wb(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    means = img.reshape(-1, 3).mean(0) + 1e-6
    gain = means.mean() / means
    gain = 1.0 + strength * (gain - 1.0)
    return np.clip(img * gain[None, None, :], 0.0, 1.0)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


# ---------- full pipeline ----------

def enhance_classical(img: np.ndarray, params: dict | None = None, *, adaptive: bool = True,
                      rules: dict | None = None, return_intermediates: bool = False,
                      disable: tuple = ()):
    """Run the classical pipeline.

    params: explicit parameter dict (skips analysis). If None, parameters are chosen
            adaptively from image statistics (or FIXED_PARAMS if adaptive=False).
    disable: names of steps to skip for ablation: {"retinex","clahe","denoise","color"}.
    """
    stats = analyze(img)
    if params is None:
        params = adaptive_params(stats, rules) if adaptive else dict(FIXED_PARAMS)
    inter = {"input": img}

    # 1) illumination estimation + decomposition
    L = estimate_illumination(img)
    inter["illumination"] = L

    # 2) Retinex enhancement (or plain gamma on the image if Retinex is ablated)
    if "retinex" in disable:
        out = np.clip(np.power(img, params["gamma"]), 0.0, 1.0)
    else:
        out, R, L_adj = retinex_enhance(img, L, params["gamma"])
        inter["reflectance"] = R
        inter["illumination_adjusted"] = L_adj
    inter["retinex"] = out

    # 3) edge-preserving denoising — before CLAHE so noise isn't amplified further.
    #    Weighted toward originally dark pixels.
    if params.get("denoise_mode") == "auto":
        on, eps, sig = post_noise_eps(out, params)
        params = {**params, "denoise_on": on, "denoise_eps": eps, "noise_after_brightening": sig}
    if params.get("denoise_on", True) and "denoise" not in disable:
        dark_w = np.clip(1.0 - L / 0.5, 0.0, 1.0) ** 0.5
        fl = params.get("denoise_floor", 0.3)
        out = denoise_guided(out, params["denoise_eps"], radius=params.get("denoise_radius", 3),
                             dark_weight=fl + (1 - fl) * dark_w)
        inter["denoised"] = out

    # 4) contrast (CLAHE)
    if params.get("clahe_on", True) and "clahe" not in disable:
        out = apply_clahe(out, params["clahe_clip"])
        inter["clahe"] = out

    # 5) colour correction
    if "color" not in disable:
        if params.get("wb_on", True):
            out = gray_world_wb(out, params.get("wb_strength", 0.7))
        out = adjust_saturation(out, params.get("saturation", 1.0))
        inter["color"] = out

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    if return_intermediates:
        return out, inter, {"stats": stats.as_dict(), "params": params}
    return out
