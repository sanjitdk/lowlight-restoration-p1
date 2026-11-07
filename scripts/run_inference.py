"""Enhance every image in a folder with the frozen configuration.

    python scripts/run_inference.py --input_dir examples --output_dir results/inference
    python scripts/run_inference.py --input_dir examples --output_dir docs/figures --mode all --save_intermediates

Uses configs/final.yaml (the same settings the evaluation and the app use).
Works with PyTorch installed or, without it, through the ONNX backend.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hybrid import HybridEnhancer  # noqa: E402
from src.io_utils import ImageInputError, load_image, save_image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAPS = ("illumination", "guidance_map", "fusion_weight_dl")  # single-channel maps → viridis
STAGES = ("reflectance", "retinex", "denoised", "clahe", "color", "dl_raw", "dl_denoised", "classical")


def _viridis(x: np.ndarray) -> np.ndarray:
    u8 = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "classical", "dl", "all"])
    ap.add_argument("--save_intermediates", action="store_true",
                    help="also save illumination/guidance/fusion maps and pipeline stages")
    ap.add_argument("--max_side", type=int, default=1024)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No images found in {args.input_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enh = HybridEnhancer(args.config)
    modes = ["classical", "dl", "hybrid"] if args.mode == "all" else [args.mode]

    for f in files:
        try:
            img = load_image(f, max_side=args.max_side)
        except ImageInputError as e:
            print(f"skip {f.name}: {e}")
            continue
        for mode in modes:
            t0 = time.perf_counter()
            out, inter, _ = enh.enhance(img, mode=mode, return_intermediates=True)
            ms = (time.perf_counter() - t0) * 1000
            save_image(out, out_dir / f"{f.stem}_{mode}.png")
            print(f"{f.name}  {mode:<9} {ms:7.0f} ms")
        if args.save_intermediates:
            _, inter, _ = enh.enhance(img, mode="hybrid", return_intermediates=True)
            save_image(img, out_dir / f"{f.stem}_input.png")
            for k in MAPS:
                if k in inter:
                    m = inter[k] if inter[k].ndim == 2 else inter[k].mean(-1)
                    save_image(_viridis(m), out_dir / f"{f.stem}_{k}.png")
            for k in STAGES:
                if k in inter:
                    save_image(np.clip(inter[k], 0, 1), out_dir / f"{f.stem}_{k}.png")


if __name__ == "__main__":
    main()
