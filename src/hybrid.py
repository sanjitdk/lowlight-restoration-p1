"""Hybrid enhancement: classical illumination estimation guides Zero-DCE++,
followed by noise-adaptive denoising, exposure-fusion of the classical and DL
outputs, and a validated quality-aware selector.

Main entry: `HybridEnhancer(cfg).enhance(img, mode="hybrid")`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .analysis import adaptive_params, analyze, luminance, post_noise_eps
from .classical import (adjust_saturation, denoise_guided, enhance_classical,
                        estimate_illumination, gray_world_wb)

try:  # PyTorch backend (training, evaluation, local runs)
    from .dl.zerodce import enhance_dl as _enhance_torch, load_model
except ImportError:  # light deploy image without torch: same network via ONNX + OpenCV DNN
    _enhance_torch = load_model = None
from .dl.zerodce_onnx import ZeroDCEppONNX, enhance_dl_onnx, load_model_onnx


def enhance_dl(model, img, strength=1.0):
    if isinstance(model, ZeroDCEppONNX):
        return enhance_dl_onnx(model, img, strength)
    return _enhance_torch(model, img, strength)


def _load_any(weights: str):
    if load_model is not None and not str(weights).endswith(".onnx"):
        return load_model(weights)
    return load_model_onnx(str(Path(weights).with_suffix(".onnx")))

DEFAULT_CFG = {
    "weights": "weights/zerodcepp_finetuned.pth",
    "rules": {},                    # overrides for analysis.DEFAULT_RULES
    "guide_min": 0.35,              # curve strength in the brightest regions
    "guide_power": 1.0,             # shape of (1 - L) guidance
    "fusion_sigma": 0.2,            # well-exposedness width (Mertens)
    "fusion_wc": 1.0, "fusion_ws": 1.0, "fusion_we": 1.0,
    "fusion_dl_bias": 1.0,          # >1 favours the DL output in fusion
    "pyr_levels": 5,
    "selector": {"enabled": False, "weights": None, "use_niqe": False},
}


def load_config(path: str | None) -> dict:
    cfg = {**DEFAULT_CFG}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        sel = {**DEFAULT_CFG["selector"], **(user.pop("selector", {}) or {})}
        cfg.update(user)
        cfg["selector"] = sel
    return cfg


# ---------------- exposure fusion (Mertens et al.) ----------------

def _fusion_weights(img, sigma, wc, ws, we):
    y = luminance(img)
    contrast = np.abs(cv2.Laplacian(y, cv2.CV_32F))
    sat = img.std(axis=2)
    well = np.prod(np.exp(-((img - 0.5) ** 2) / (2 * sigma ** 2)), axis=2)
    return (contrast + 1e-6) ** wc * (sat + 1e-6) ** ws * (well + 1e-6) ** we


def _gauss_pyr(x, n):
    p = [x]
    for _ in range(n - 1):
        p.append(cv2.pyrDown(p[-1]))
    return p


def _lap_pyr(x, n):
    g = _gauss_pyr(x, n)
    lp = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(n - 1)]
    return lp + [g[-1]]


def exposure_fusion(imgs, weights, levels=5):
    """Laplacian-pyramid blending of images with per-pixel weights (no seams/halos)."""
    h, w = imgs[0].shape[:2]
    levels = max(1, min(levels, int(np.log2(min(h, w))) - 2))
    W = np.stack(weights, 0)
    W = W / (W.sum(0, keepdims=True) + 1e-12)
    out = None
    for img, wt in zip(imgs, W):
        lp = _lap_pyr(img.astype(np.float32), levels)
        gw = _gauss_pyr(wt.astype(np.float32), levels)
        blended = [l * g[..., None] for l, g in zip(lp, gw)]
        out = blended if out is None else [o + b for o, b in zip(out, blended)]
    res = out[-1]
    for i in range(levels - 2, -1, -1):
        res = cv2.pyrUp(res, dstsize=(out[i].shape[1], out[i].shape[0])) + out[i]
    return np.clip(res, 0.0, 1.0)


# ---------------- quality features for the selector ----------------

def quality_features(out, inp, use_niqe=False):
    """Interpretable no-reference checks (lower is better for every feature)."""
    y = luminance(out)
    yin = luminance(inp)
    lab = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2LAB)
    gm_out = np.abs(cv2.Laplacian(y, cv2.CV_32F)).mean()
    gm_in = np.abs(cv2.Laplacian(yin, cv2.CV_32F)).mean() * (y.mean() / max(yin.mean(), 1e-3))
    from skimage.restoration import estimate_sigma
    try:
        sig = float(np.mean(estimate_sigma(out, channel_axis=-1)))
    except (ValueError, FloatingPointError):
        sig = 0.0
    f = {
        "exposure_err": abs(float(y.mean()) - 0.45),
        "clipped": float((y > 0.98).mean() + (y < 0.02).mean()),
        "color_cast": float(np.hypot(lab[..., 1].mean(), lab[..., 2].mean()) / 100.0),
        "detail_loss": float(max(0.0, 1.0 - gm_out / max(gm_in, 1e-6))),
        "noise": sig,
    }
    if use_niqe:
        from .metrics import niqe
        f["niqe"] = niqe(out) / 10.0
    return f


FEATURE_ORDER = ["exposure_err", "clipped", "color_cast", "detail_loss", "noise", "niqe"]


def selector_score(feats: dict, weights: dict) -> float:
    """Higher score = predicted better. Weights are fitted on val (scripts/fit_selector.py)."""
    return -sum(weights.get(k, 0.0) * v for k, v in feats.items())


# ---------------- main enhancer ----------------

class HybridEnhancer:
    MODES = ("hybrid", "auto", "classical", "dl")

    def __init__(self, cfg: dict | str | None = None, model=None):
        self.cfg = load_config(cfg) if (cfg is None or isinstance(cfg, str)) else {**DEFAULT_CFG, **cfg}
        self.model = model if model is not None else _load_any(self.cfg["weights"])

    def enhance(self, img, mode="hybrid", disable=(), adaptive=True, return_intermediates=False):
        """mode: 'hybrid' (fused), 'auto' (selector over candidates), 'classical', 'dl'.
        disable (ablation): {'guidance','denoise','fusion','clahe','color','retinex'}.
        adaptive=False uses fixed parameters everywhere.
        """
        cfg = self.cfg
        stats = analyze(img)
        params = adaptive_params(stats, cfg["rules"]) if adaptive else None
        inter, info = {"input": img}, {"stats": stats.as_dict()}

        cls_disable = tuple(d for d in disable if d in ("clahe", "color", "retinex", "denoise"))
        cls, cinter, cinfo = enhance_classical(img, params, adaptive=adaptive, rules=cfg["rules"],
                                               return_intermediates=True, disable=cls_disable)
        params = cinfo["params"]
        info["params"] = params
        if mode == "classical":
            return (cls, {**inter, **cinter}, info) if return_intermediates else cls

        # DL branch, guided by the classical illumination map
        L = cinter["illumination"]
        inter["illumination"] = L
        if "guidance" in disable or mode == "dl":
            strength = 1.0
        else:
            strength = cfg["guide_min"] + (1 - cfg["guide_min"]) * np.power(1.0 - L, cfg["guide_power"])
            inter["guidance_map"] = strength
        dl = enhance_dl(self.model, img, strength)
        inter["dl_raw"] = dl
        if mode == "dl":
            return (dl, inter, info) if return_intermediates else dl

        # Noise-adaptive, edge-preserving denoise of the DL output (strongest in dark areas)
        dparams = params
        if params.get("denoise_mode") == "auto":
            on, eps, sig = post_noise_eps(dl, params)
            dparams = {**params, "denoise_on": on, "denoise_eps": eps}
            info["dl_noise_after_brightening"] = sig
        if dparams.get("denoise_on", True) and "denoise" not in disable:
            dark_w = np.clip(1.0 - L / 0.5, 0.0, 1.0) ** 0.5
            fl = params.get("denoise_floor", 0.3)
            dl = denoise_guided(dl, dparams["denoise_eps"], radius=params.get("denoise_radius", 3),
                                dark_weight=fl + (1 - fl) * dark_w)
            inter["dl_denoised"] = dl
        if "color" not in disable:
            if params.get("wb_on", False):
                dl = gray_world_wb(dl, params.get("wb_strength", 0.7))
            dl = adjust_saturation(dl, 1.0 + (params.get("saturation", 1.0) - 1.0) * 0.5)

        # Fusion of classical + DL outputs
        if "fusion" in disable:
            fused = dl
        else:
            wargs = (cfg["fusion_sigma"], cfg["fusion_wc"], cfg["fusion_ws"], cfg["fusion_we"])
            wc_ = _fusion_weights(cls, *wargs)
            wd_ = _fusion_weights(dl, *wargs) * cfg["fusion_dl_bias"]
            fused = exposure_fusion([cls, dl], [wc_, wd_], cfg["pyr_levels"])
            inter["fusion_weight_dl"] = wd_ / (wc_ + wd_ + 1e-12)
        # Highlight guard: never darken or blow out regions that were already bright
        bright = np.clip((L - 0.8) / 0.2, 0.0, 1.0)[..., None]
        fused = np.clip(fused * (1 - bright) + np.maximum(img, np.minimum(fused, 1.0)) * bright, 0, 1)
        fused = fused.astype(np.float32)
        inter["hybrid"] = fused

        if mode == "auto" and cfg["selector"].get("enabled") and cfg["selector"].get("weights"):
            cands = {"hybrid": fused, "dl": inter["dl_raw"], "classical": cls}
            use_niqe = cfg["selector"].get("use_niqe", False)
            scores = {k: selector_score(quality_features(v, img, use_niqe), cfg["selector"]["weights"])
                      for k, v in cands.items()}
            choice = max(scores, key=scores.get)
            info["selector"] = {"scores": scores, "choice": choice}
            fused = cands[choice]
        elif mode == "auto":
            info["selector"] = {"choice": "hybrid", "note": "selector disabled (did not help on val)"}

        if return_intermediates:
            inter["classical"] = cls
            return fused, {**cinter, **inter}, info
        return fused
