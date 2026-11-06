"""Find and visualise the worst test cases for the hybrid and diagnose them.

Usage: python scripts/failure_cases.py --split test_lolv2 [--k 5]

Ranks images by hybrid PSNR (from results/<split>/per_image.csv), re-runs them,
and saves a grid (input | classical | DL | hybrid | GT) per case to
results/failures/, plus failures.md with an automatic diagnosis based on the
measured statistics (darkness, noise, colour cast, over/under-exposure vs. GT).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import split_items  # noqa: E402
from src.analysis import luminance  # noqa: E402
from src.hybrid import HybridEnhancer  # noqa: E402
from src.io_utils import load_image, to_uint8  # noqa: E402
from src.metrics import noise_sigma  # noqa: E402

HYB = "hybrid (adaptive) [ours]"


def diagnose(stats, out, gt):
    notes = []
    dy = luminance(out).mean() - luminance(gt).mean()
    if dy < -0.08:
        notes.append(f"under-exposed vs. GT (mean luma {dy:+.2f})")
    elif dy > 0.08:
        notes.append(f"over-exposed vs. GT (mean luma {dy:+.2f})")
    if stats["mean_lum"] < 0.03:
        notes.append(f"extremely dark input (mean luma {stats['mean_lum']:.3f})")
    if stats["non_uniformity"] > 0.6:
        notes.append(f"strongly non-uniform lighting ({stats['non_uniformity']:.2f})")
    if stats["highlight_ratio"] > 0.01:
        notes.append(f"light sources / clipped highlights ({stats['highlight_ratio'] * 100:.1f}% px)")
    cast_out = np.abs(out.reshape(-1, 3).mean(0) - gt.reshape(-1, 3).mean(0)).max()
    if cast_out > 0.06:
        notes.append(f"colour shift vs. GT (max channel-mean diff {cast_out:.2f})")
    n_out, n_gt = noise_sigma(out), noise_sigma(gt)
    if n_out > n_gt + 0.01:
        notes.append(f"residual noise (σ {n_out:.3f} vs. GT {n_gt:.3f})")
    elif n_out < n_gt - 0.01:
        notes.append(f"over-smoothed (σ {n_out:.3f} vs. GT {n_gt:.3f})")
    return notes or ["no single dominant cause; GT style/tone mismatch"]


def grid(images, labels):
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (w * len(images), h + 24), "white")
    d = ImageDraw.Draw(canvas)
    for i, (im, lb) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(to_uint8(im)), (i * w, 24))
        d.text((i * w + 6, 5), lb, fill="black")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_lolv2")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "results" / args.split / "per_image.csv")
    h = df[df["method"] == HYB].sort_values("psnr").head(args.k)
    items = {p.name: (p, g) for p, g, _ in split_items(args.split)}
    H = HybridEnhancer(str(ROOT / args.config))
    out_dir = ROOT / "results/failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# Failure cases — worst {args.k} by hybrid PSNR on {args.split}", ""]
    for _, r in h.iterrows():
        lp, gp = items[r["image"]]
        lo, gt = load_image(lp, None), load_image(gp, None)
        out, inter, info = H.enhance(lo, mode="hybrid", return_intermediates=True)
        g = grid([lo, inter["classical"], inter["dl_raw"], out, gt], ["input", "classical", "DL", "hybrid", "GT"])
        fn = f"{args.split}_{Path(r['image']).stem}.jpg"
        g.save(out_dir / fn, quality=88)
        notes = diagnose(info["stats"], out, gt)
        md += [f"## {r['image']} — PSNR {r['psnr']:.2f} dB, SSIM {r['ssim']:.3f}, LPIPS {r['lpips']:.3f}", "",
               f"![{r['image']}]({fn})", "", "Diagnosis (automatic, from measured statistics):", ""]
        md += [f"- {n}" for n in notes] + [""]
        print(r["image"], notes)
    (out_dir / f"failures_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote results/failures/failures_{args.split}.md")


if __name__ == "__main__":
    main()
