"""Tune adaptive-rule constants and hybrid settings on the VALIDATION split only.

Usage: python scripts/tune_val.py --weights weights/zerodcepp_finetuned.pth --out configs/final.yaml

Coordinate descent: each parameter is swept over a small grid while the others
stay fixed, keeping the value with the best objective
    J = mean(PSNR)/20 + mean(SSIM)
(both terms are on a similar ~0–1 scale for LOL). Stage 1 tunes the classical
rules using the classical output; stage 2 tunes the hybrid-only settings.
Test images are never loaded here. A log of every trial goes to results/tuning_log.csv.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hybrid import DEFAULT_CFG, HybridEnhancer  # noqa: E402
from src.io_utils import load_image  # noqa: E402
from src.metrics import lpips, psnr, ssim  # noqa: E402

CLASSICAL_GRID = {
    "denoise_k": [1.5, 2.5, 4.0, 6.0, 8.0],
    "denoise_floor": [0.3, 0.6, 1.0],
    "denoise_radius": [3, 5, 7],
    "target_lum": [0.35, 0.45, 0.55],
    "clahe_clip_max": [1.5, 3.0, 4.5],
    "saturation_boost": [1.0, 1.15, 1.3],
}
HYBRID_GRID = {
    "guide_min": [0.2, 0.5, 0.8, 1.0],
    "fusion_dl_bias": [0.5, 1.0, 2.0, 4.0],
}


def load_val(root, max_n=None):
    names = (root / "data/splits/val.txt").read_text().split()[:max_n]
    base = root / "data/LOL/our485"
    return [(load_image(base / "low" / n, None), load_image(base / "high" / n, None)) for n in names]


W_LPIPS = 1.0


def objective(H, data, mode):
    ps, ss, ls = [], [], []
    for lo, hi in data:
        o = H.enhance(lo, mode=mode)
        ps.append(psnr(hi, o)); ss.append(ssim(hi, o))
        ls.append(lpips(hi, o) if W_LPIPS else 0.0)
    p, s_, l_ = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))
    return p / 20 + s_ - W_LPIPS * l_, p, s_, l_


def sweep(H, data, grid, target, mode, log):
    for key, values in grid.items():
        best = None
        for v in values:
            target[key] = v
            t = time.time()
            J, p, s, lp = objective(H, data, mode)
            log.writerow([mode, key, v, f"{J:.4f}", f"{p:.3f}", f"{s:.4f}", f"{lp:.4f}", f"{time.time() - t:.0f}"])
            print(f"[{mode}] {key}={v}: J={J:.4f} PSNR={p:.2f} SSIM={s:.4f} LPIPS={lp:.4f}", flush=True)
            if best is None or J > best[0]:
                best = (J, v, p, s, lp)
        target[key] = best[1]
        print(f"  -> {key}={best[1]} (PSNR {best[2]:.2f}, SSIM {best[3]:.4f}, LPIPS {best[4]:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/zerodcepp_finetuned.pth")
    ap.add_argument("--out", default="configs/final.yaml")
    ap.add_argument("--max-n", type=int, default=None, help="use first N val images (speed)")
    ap.add_argument("--w-lpips", type=float, default=1.0)
    args = ap.parse_args()
    global W_LPIPS
    W_LPIPS = args.w_lpips

    data = load_val(ROOT, args.max_n)
    cfg = {**DEFAULT_CFG, "weights": args.weights, "rules": {}}
    H = HybridEnhancer(cfg)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/tuning_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["stage", "param", "value", "J", "psnr", "ssim", "lpips", "secs"])
        sweep(H, data, CLASSICAL_GRID, H.cfg["rules"], "classical", log)
        sweep(H, data, HYBRID_GRID, H.cfg, "hybrid", log)
        J, p, s, lp = objective(H, data, "hybrid")
        print(f"final val hybrid: PSNR {p:.2f} SSIM {s:.4f} LPIPS {lp:.4f}")

    out = {k: H.cfg[k] for k in DEFAULT_CFG if k != "selector"}
    out["rules"] = {k: (float(v) if isinstance(v, float) else v) for k, v in H.cfg["rules"].items()}
    out["selector"] = {"enabled": False, "weights": None, "use_niqe": False}
    out["_tuned_on"] = f"val ({len(data)} images from data/splits/val.txt), J = PSNR/20 + SSIM - {W_LPIPS}*LPIPS"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
