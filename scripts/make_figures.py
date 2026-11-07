"""Report figures from the example images (frozen config, no tuning).

    python scripts/make_figures.py      → docs/figures/fig_*.jpg

Picks the darkest example (extreme low light) and the most unevenly lit one
(non-uniform), and shows the hybrid pipeline's intermediate maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import analyze  # noqa: E402
from src.hybrid import HybridEnhancer  # noqa: E402
from src.io_utils import load_image  # noqa: E402

OUT = Path("docs/figures")
H = 300  # tile height


def _viridis(x):
    u8 = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def _tile(img, label):
    img = img if img.ndim == 3 else _viridis(img)
    w = int(img.shape[1] * H / img.shape[0])
    t = cv2.resize(np.clip(img, 0, 1), (w, H), interpolation=cv2.INTER_AREA)
    t = (t * 255).astype(np.uint8)
    bar = np.full((28, w, 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def _row(tiles):
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    out = [tiles[0]]
    for t in tiles[1:]:
        out += [gap, t]
    return np.hstack(out)


def _save(rows, name):
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(OUT / name), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    enh = HybridEnhancer("configs/final.yaml")
    imgs = {p.stem: load_image(p, max_side=768) for p in sorted(Path("examples").glob("*.jpg"))}
    stats = {k: analyze(v) for k, v in imgs.items()}
    for k, s in stats.items():
        print(f"{k}: mean_lum {s.mean_lum:.3f}  non_uniformity {s.non_uniformity:.3f}  noise {s.noise_sigma:.4f}")
    darkest = min(stats, key=lambda k: stats[k].mean_lum)
    uneven = max((k for k in stats if k != darkest), key=lambda k: stats[k].non_uniformity)

    def methods(k):
        img = imgs[k]
        hyb, inter, _ = enh.enhance(img, mode="hybrid", return_intermediates=True)
        return img, inter, [_tile(img, f"Input ({k})"), _tile(enh.enhance(img, mode="classical"), "Classical"),
                            _tile(enh.enhance(img, mode="dl"), "DL (Zero-DCE++ ft)"), _tile(hyb, "Hybrid (ours)")]

    _, _, row = methods(darkest)
    _save([_row(row)], "fig_extreme_lowlight.jpg")
    _, _, row = methods(uneven)
    _save([_row(row)], "fig_nonuniform.jpg")

    img, inter, _ = methods(darkest)
    r1 = [_tile(img, "Input"), _tile(inter["illumination"], "Illumination L"),
          _tile(np.clip(inter["reflectance"], 0, 1), "Reflectance R"), _tile(inter["retinex"], "Retinex (adaptive gamma)")]
    r2 = [_tile(inter["guidance_map"], "DL guidance strength"), _tile(inter["dl_raw"], "Guided Zero-DCE++"),
          _tile(inter["fusion_weight_dl"], "Fusion weight (DL)"), _tile(inter["hybrid"], "Hybrid output")]
    _save([_row(r1), _row(r2)], "fig_intermediates.jpg")


if __name__ == "__main__":
    main()
