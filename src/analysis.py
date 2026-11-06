"""Image statistics → adaptive enhancement parameters.

`analyze()` measures the input; `adaptive_params()` maps the measurements to
pipeline settings. The mapping thresholds live in configs/*.yaml and are tuned
on the validation split only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.restoration import estimate_sigma


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 luma of float RGB [0,1]."""
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


@dataclass
class ImageStats:
    mean_lum: float          # average brightness [0,1]
    median_lum: float
    dark_ratio: float        # fraction of pixels with luma < 0.05
    highlight_ratio: float   # fraction of pixels with luma > 0.95 (light sources)
    noise_sigma: float       # estimated Gaussian noise std (in [0,1] units)
    non_uniformity: float    # std of heavily blurred luma / its mean (lighting unevenness)
    color_cast: float        # gray-world channel deviation (0 = neutral)
    contrast: float          # RMS contrast of luma

    def as_dict(self):
        return asdict(self)


def analyze(img: np.ndarray) -> ImageStats:
    y = luminance(img)
    h, w = y.shape
    # Noise estimate on a downsized copy for speed; dark images understate noise
    # in absolute terms, so this is measured on the raw input (pre-amplification).
    small = img if max(h, w) <= 512 else cv2.resize(img, (w * 512 // max(h, w), h * 512 // max(h, w)),
                                                    interpolation=cv2.INTER_AREA)
    try:
        sigma = float(np.mean(estimate_sigma(small, channel_axis=-1)))
    except (ValueError, FloatingPointError):
        sigma = 0.0
    k = max(3, (min(h, w) // 8) | 1)
    blur = cv2.GaussianBlur(y, (k, k), 0)
    means = img.reshape(-1, 3).mean(0)
    gray = means.mean() + 1e-6
    return ImageStats(
        mean_lum=float(y.mean()),
        median_lum=float(np.median(y)),
        dark_ratio=float((y < 0.05).mean()),
        highlight_ratio=float((y > 0.95).mean()),
        noise_sigma=sigma if np.isfinite(sigma) else 0.0,
        non_uniformity=float(blur.std() / (blur.mean() + 1e-6)),
        color_cast=float(np.abs(means / gray - 1.0).max()),
        contrast=float(y.std()),
    )


# Default mapping constants. Overridden by configs/final.yaml (tuned on val).
DEFAULT_RULES = {
    "target_lum": 0.45,        # desired mean brightness after enhancement
    "gamma_min": 0.25, "gamma_max": 0.9,
    "clahe_clip_min": 1.0, "clahe_clip_max": 3.0,
    "denoise_sigma_lo": 0.005, "denoise_sigma_hi": 0.06,
    "guided_eps_min": 1e-4, "guided_eps_max": 2e-2,
    "denoise_k": 2.5,          # guided-filter eps = (k * sigma_after_brightening)^2
    "denoise_min_sigma": 0.004,  # skip denoising below this post-brightening noise
    "denoise_radius": 3,
    "denoise_floor": 0.3,      # min denoise blend in bright areas (1.0 = uniform)
    "wb_cast_thresh": 0.08,    # apply white balance only above this cast
    "wb_strength": 0.7,
    "saturation_boost": 1.15,
}

# Fixed parameters for the "non-adaptive" comparison baseline.
FIXED_PARAMS = {
    "gamma": 0.45, "clahe_clip": 2.0, "clahe_on": True,
    "denoise_eps": 4e-3, "denoise_on": True,
    "wb_on": True, "wb_strength": 0.7, "saturation": 1.15,
}


def adaptive_params(stats: ImageStats, rules: dict | None = None) -> dict:
    r = {**DEFAULT_RULES, **(rules or {})}
    m = max(stats.mean_lum, 1e-3)
    # Gamma on illumination so that mean brightness moves toward target:
    # m ** g = target  →  g = log(target) / log(m)
    g = np.log(r["target_lum"]) / np.log(m) if m < r["target_lum"] else 1.0
    gamma = float(np.clip(g, r["gamma_min"], r["gamma_max"])) if m < r["target_lum"] else 1.0

    # CLAHE: low-contrast images get a higher clip limit; very noisy ones a lower one
    c = np.clip(1.0 - stats.contrast / 0.25, 0.0, 1.0)
    n = np.clip(stats.noise_sigma / r["denoise_sigma_hi"], 0.0, 1.0)
    clip = r["clahe_clip_min"] + (r["clahe_clip_max"] - r["clahe_clip_min"]) * c * (1.0 - 0.5 * n)

    # Denoise strength (guided-filter eps) grows with estimated noise and darkness
    # (darker inputs are amplified more, so their noise becomes more visible).
    amp = 1.0 / m
    t = np.clip((stats.noise_sigma * np.sqrt(amp) - r["denoise_sigma_lo"]) /
                (r["denoise_sigma_hi"] - r["denoise_sigma_lo"]), 0.0, 1.0)
    eps = float(r["guided_eps_min"] * (r["guided_eps_max"] / r["guided_eps_min"]) ** t)

    return {
        "gamma": gamma,
        "clahe_clip": float(clip),
        "clahe_on": True,
        "denoise_eps": eps,
        "denoise_on": t > 0.0 or stats.noise_sigma > 0.002,
        "wb_on": stats.color_cast > r["wb_cast_thresh"],
        "wb_strength": r["wb_strength"],
        "saturation": r["saturation_boost"],
        "target_lum": r["target_lum"],
        # "auto": measure noise after brightening and set eps from it (see post_noise_eps)
        "denoise_mode": "auto",
        "denoise_k": r["denoise_k"],
        "denoise_min_sigma": r["denoise_min_sigma"],
        "denoise_radius": int(r["denoise_radius"]),
        "denoise_floor": float(r["denoise_floor"]),
    }


def measure_noise(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    s = 512 / max(h, w)
    small = img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    try:
        v = float(np.mean(estimate_sigma(small, channel_axis=-1)))
    except (ValueError, FloatingPointError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def post_noise_eps(brightened: np.ndarray, params: dict):
    """Noise is only visible after brightening, so estimate it there.

    Returns (denoise_on, eps, sigma). The guided filter smooths regions whose local
    variance is below eps, so eps ~ (k * sigma)^2 targets noise-level fluctuations.
    """
    sigma = measure_noise(brightened)
    on = sigma > params.get("denoise_min_sigma", 0.004)
    eps = float(np.clip((params.get("denoise_k", 2.5) * sigma) ** 2, 1e-5, 5e-2))
    return on, eps, sigma
