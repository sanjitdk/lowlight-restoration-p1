---
title: Adaptive Low-Light Image Restoration
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Adaptive Low-Light Image Restoration (CV Hackathon 2026, P1)

**Live demo:** https://lowlight-restoration-p1.onrender.com  
(Render free tier: if it has been idle, the first load takes ~30–60 s while it wakes up.)

This project enhances dark, noisy, colour-distorted photos with three approaches, which are compared on the same data:

| Approach | What it does |
|---|---|
| **Classical** | Illumination estimation (max-RGB + guided filter) → Retinex decomposition with adaptive gamma → noise-adaptive guided-filter denoising → CLAHE → gray-world white balance |
| **Deep learning** | **Zero-DCE++** (official architecture and weights, 10,561 params), fine-tuned on the LOL-v1 train split |
| **Hybrid (ours)** | The classical illumination map scales the Zero-DCE++ curve per pixel (protecting bright areas), followed by noise-adaptive denoising and Laplacian-pyramid exposure fusion of both branches, plus a highlight guard |

Every parameter is set from measured image statistics (brightness, noise *after* brightening, uneven lighting, colour cast). The rule constants were tuned on a held-out validation split only.

- Results: [`eval_P1_T31.md`](eval_P1_T31.md)
- Design document: [`docs/design.md`](docs/design.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)

## Quick start

### Docker (one command)
```bash
docker build -t lowlight . && docker run -p 7860:7860 lowlight
# open http://localhost:7860
```

### Python
```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt                         # app only
python app.py                                           # http://localhost:7860
```

## Reproducing the results
```bash
pip install -r requirements-dev.txt

# 1. Data (public mirrors, no login required) → data/
#    LOL-v1:      https://huggingface.co/datasets/geekyrakshit/LoL-Dataset  (lol_dataset.zip → data/LOL/{our485,eval15})
#    LOL-v2 Real: https://huggingface.co/datasets/okhater/lolv2-real       (Test/{Input,GT} → data/LOLv2/Real_captured/Test)
python scripts/make_splits.py                 # our485 → 435 train / 50 val (seed 42)

# 2. Fine-tune Zero-DCE++ (CPU ≈ 5 min/epoch; best epoch chosen on val)
#    Official weights: https://github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth)
#    → weights/zerodcepp_pretrained.pth
python -m src.dl.train --epochs 12 --bs 8 --crop 256 --sup 1.0

# 3. Tune adaptive rules + hybrid settings on VAL only → configs/final.yaml
python scripts/tune_val.py
python scripts/fit_selector.py                # enables the selector only if it beats always-hybrid on val

# 4. Final evaluation on the test sets (run once; results/test_lock.json guards the config)
python scripts/evaluate.py --split test_lol
python scripts/evaluate.py --split test_lolv2
python scripts/failure_cases.py --split test_lolv2
python scripts/make_eval_md.py --team T31 --usns "1RUA24CSE0319,1RUA24CSE0320,1RUA24CSE0352,1RUA24CSE0413"

# Tests
python tests/test_smoke.py
```

## Repository layout
```
app.py                 Gradio app (Docker entrypoint, deployed on Render)
src/io_utils.py        safe image loading / validation
src/analysis.py        image statistics → adaptive parameters
src/classical.py       classical pipeline
src/dl/zerodce.py      Zero-DCE++ model + guided inference
src/dl/train.py        fine-tuning
src/hybrid.py          hybrid pipeline, fusion, quality selector
src/metrics.py         PSNR, SSIM, LPIPS, NIQE, noise σ, edge metrics
scripts/               splits, tuning, selector fitting, evaluation, failure analysis, eval report
configs/final.yaml     frozen configuration (tuned on val)
weights/               Zero-DCE++ pretrained + fine-tuned (≈52 KB each)
results/               evaluation tables, logs, failure cases
docs/                  design document, architecture diagram, viva notes
```

## Limitations and ethics (summary)
See `docs/design.md` for the full discussion. In short:
- Trained and tuned on LOL (indoor, static, roughly uniform darkness). Outdoor night scenes with strong light sources are less well covered.
- Very heavy noise can leave grain, or get over-smoothed.
- Enhanced images are *plausible reconstructions*, not ground truth. They should not be used as forensic or evidential material.
- No uploaded images are stored by the app.
