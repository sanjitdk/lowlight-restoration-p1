"""Fit the quality-aware selector on the VALIDATION split and decide whether to enable it.

Usage: python scripts/fit_selector.py --config configs/final.yaml [--niqe]

For every val image, three candidates are produced (hybrid, DL-only, classical).
Each candidate gets no-reference features (lower = better): exposure error,
clipped/crushed pixels, colour cast, detail loss, residual noise [, NIQE].
Target quality (uses GT, val only): J = PSNR/20 + SSIM - LPIPS (same as tune_val.py).

Weights are fitted with non-negative least squares on per-image-centred values
(so only *relative* quality between candidates matters). Selection quality is
measured with leave-one-image-out cross-validation and compared with
"always hybrid" and the oracle. The selector is enabled in the config ONLY if
it beats always-hybrid on val; the result is written to results/selector_val.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hybrid import FEATURE_ORDER, HybridEnhancer, load_config, quality_features  # noqa: E402
from src.io_utils import load_image  # noqa: E402
from src.metrics import lpips, psnr, ssim  # noqa: E402

CANDS = ["hybrid", "dl", "classical"]


def fit(X, y):
    """X: (n_img, n_cand, n_feat), y: (n_img, n_cand). Returns non-negative weights."""
    Xc = (X - X.mean(1, keepdims=True)).reshape(-1, X.shape[2])
    yc = (y - y.mean(1, keepdims=True)).reshape(-1)
    w, _ = nnls(-Xc, yc)  # score = -w.x should track quality
    return w


def pick(X, w):
    return np.argmax(-(X @ w), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--niqe", action="store_true", help="include NIQE as a feature")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / args.config))
    H = HybridEnhancer(cfg)
    names = (ROOT / "data/splits/val.txt").read_text().split()
    base = ROOT / "data/LOL/our485"
    feats = [k for k in FEATURE_ORDER if k != "niqe" or args.niqe]
    X, Y, P, S = [], [], [], []
    for i, n in enumerate(names):
        lo, hi = load_image(base / "low" / n, None), load_image(base / "high" / n, None)
        _, inter, _ = H.enhance(lo, mode="hybrid", return_intermediates=True)
        cands = {"hybrid": inter["hybrid"], "dl": inter["dl_raw"], "classical": inter["classical"]}
        xr, yr, pr, sr = [], [], [], []
        for c in CANDS:
            f = quality_features(cands[c], lo, use_niqe=args.niqe)
            xr.append([f[k] for k in feats])
            p, s = psnr(hi, cands[c]), ssim(hi, cands[c])
            pr.append(p); sr.append(s); yr.append(p / 20 + s - lpips(hi, cands[c]))
        X.append(xr); Y.append(yr); P.append(pr); S.append(sr)
        print(f"[{i + 1}/{len(names)}] {n}", flush=True)
    X, Y, P, S = map(np.array, (X, Y, P, S))

    # leave-one-image-out CV
    loo = []
    for i in range(len(names)):
        m = np.arange(len(names)) != i
        w = fit(X[m], Y[m])
        loo.append(pick(X[i:i + 1], w)[0])
    loo = np.array(loo)
    idx = np.arange(len(names))
    hyb = CANDS.index("hybrid")
    res = {
        "features": feats,
        "always_hybrid": {"J": float(Y[:, hyb].mean()), "psnr": float(P[:, hyb].mean()), "ssim": float(S[:, hyb].mean())},
        "selector_loo": {"J": float(Y[idx, loo].mean()), "psnr": float(P[idx, loo].mean()),
                         "ssim": float(S[idx, loo].mean()),
                         "choice_counts": {c: int((loo == k).sum()) for k, c in enumerate(CANDS)}},
        "oracle": {"J": float(Y.max(1).mean()), "psnr": float(P[idx, Y.argmax(1)].mean()),
                   "ssim": float(S[idx, Y.argmax(1)].mean()),
                   "best_counts": {c: int((Y.argmax(1) == k).sum()) for k, c in enumerate(CANDS)}},
        "selection_accuracy_loo": float((loo == Y.argmax(1)).mean()),
    }
    for k, c in enumerate(CANDS):
        res[f"always_{c}"] = {"J": float(Y[:, k].mean()), "psnr": float(P[:, k].mean()), "ssim": float(S[:, k].mean())}
    w_all = fit(X, Y)
    res["weights"] = {k: float(v) for k, v in zip(feats, w_all)}
    best_fixed = max(CANDS, key=lambda c: res[f"always_{c}"]["J"])
    enable = res["selector_loo"]["J"] > res["always_hybrid"]["J"] + 1e-3
    res["enabled"] = bool(enable)
    res["best_fixed_candidate_on_val"] = best_fixed
    res["decision"] = ("selector enabled: beats always-hybrid on val (LOO)" if enable else
                       "selector disabled: does not beat always-hybrid on val (LOO)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/selector_val.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    cfg_file = ROOT / args.config
    raw = yaml.safe_load(cfg_file.read_text())
    raw["selector"] = {"enabled": bool(enable), "weights": res["weights"], "use_niqe": bool(args.niqe)}
    cfg_file.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"updated {args.config}: selector.enabled={enable}")


if __name__ == "__main__":
    main()
