# To-do list (tick `[x]` as each step finishes; log details in CLAUDE.md)

## Phase 0 — Planning docs
- [x] plan.md
- [x] goal.md
- [x] todolist.md
- [x] CLAUDE.md

## Phase 1 — Setup (~10 min)
- [x] Create folder structure (`src/`, `src/dl/`, `scripts/`, `configs/`, `notebooks/`, `weights/`, `results/`, `docs/`, `data/`)
- [x] `requirements.txt` (app runtime) + `requirements-dev.txt` (eval/train tools)
- [x] `.gitignore` (data/, venv, kaggle.json, *.env, caches)
- [x] Python venv + install; `git init`

## Phase 2 — Data (~15 min)
- [x] Download LOL-v1 (our485 + eval15) → `data/LOL/` (HF mirror geekyrakshit/LoL-Dataset; 485 + 15 pairs verified)
- [x] Download LOL-v2 Real (test 100) → `data/LOLv2/` (HF mirror okhater/lolv2-real; Input/GT 100 pairs)
- [ ] Collect 10–20 unseen real low-light photos → `data/unseen/`
- [x] `scripts/make_splits.py`: our485 → 435 train / 50 val (seed 42) → `data/splits/*.txt`

## Phase 3 — Classical + analysis (~30 min)
- [x] `src/io_utils.py`: safe load (alpha, grayscale, corrupt, tiny, huge → max side 1024)
- [x] `src/analysis.py`: luminance, dark ratio, noise σ, non-uniformity, colour cast, highlight ratio → params
- [x] `src/classical.py`: illumination map, Retinex, CLAHE, guided/bilateral denoise, colour correction; returns intermediates
- [x] Fixed-param variant for comparison (`FIXED_PARAMS`, `adaptive=False`)
- [ ] Quick visual check on 3 val images

## Phase 4 — Deep learning (~40 min)
- [x] `src/dl/zerodce.py`: Zero-DCE++ port from the official repo
- [x] Strict load of the pretrained weights: OK, 10,561 params
- [x] Sanity check: pretrained val PSNR 14.46 dB (50 imgs), consistent with published ~14–15 dB on LOL
- [x] `src/dl/train.py` (train split only, best ckpt by val). Kaggle not needed: CPU fine-tune is fast enough (~6 min/epoch)
- [x] Fine-tune on local CPU → `weights/zerodcepp_finetuned.pth` (best = epoch 1, val 16.23 dB; stopped after epoch 3 as val fell)

## Phase 5 — Hybrid (~25 min)
- [x] Illumination-guided curve scaling (`src/hybrid.py`)
- [x] Noise-adaptive guided denoise (noise measured AFTER brightening; dark-region weighted)
- [x] Mertens-style Laplacian-pyramid fusion of classical + DL; highlight guard
- [x] Combined quality selector code + `scripts/fit_selector.py` (NNLS, leave-one-out on val, enable only if it helps)
- [x] `scripts/tune_val.py` + `scripts/fit_selector.py` on **val only** → frozen `configs/final.yaml` (selector disabled: did not beat always-hybrid)

## Phase 6 — Evaluation (~25 min)
- [x] `src/metrics.py`: PSNR, SSIM, LPIPS, NIQE, latency, noise σ, edge corr/F1
- [x] `scripts/evaluate.py` (methods + ablations, test lock); val numbers come from tuning logs
- [x] Ablations: −illum guidance, −CLAHE, −denoise, −fusion, −colour, −Retinex, adaptive→fixed (on both test sets)
- [x] Selector vs. always-hybrid vs. oracle (val LOO: 1.234 vs 1.256 vs 1.327 → disabled)
- [x] `scripts/evaluate.py --split test_lol` + `test_lolv2` **run once** with the frozen config
- [x] Generated `eval_P1_TXX.md` (placeholders: Team ID, USN) (USNs first line; PS metric order)
- [x] `scripts/failure_cases.py`: worst 5 + diagnosis → `results/failures/`

## Phase 7 — App + deploy (~25 min)
- [x] `app.py` Gradio (run() tested: outputs, bad file, no file): upload/examples, mode selector, before/after, intermediates gallery, metrics/latency readout, error messages
- [x] Bad-input tests via the running server (corrupt, 8×8, alpha, 4000 px) + tests/test_smoke.py
- [x] `Dockerfile` + `.dockerignore`; `python app.py` checked locally (Docker Desktop not running, so no local image build)
- [x] UI redesign: slider, stat cards, compare tab, thumbnails, mobile-checked
- [ ] Push to Hugging Face Spaces (waiting for HF login)
- [ ] Measure HF CPU latency (600 / 1024 / 4000 px, cold vs. warm) → eval file
- [ ] Test the URL on phone + laptop

## Phase 8 — Docs + submission (~15 min)
- [x] `docs/architecture.md` (Mermaid), matching `src/`
- [x] `docs/design.md` (all sections; deploy latency pending)
- [x] `README.md` (setup, one command, reproduce results; live URL pending)
- [x] `docs/viva_notes.md` + 3-min demo script
- [ ] Fill in USNs / Team ID
- [ ] Push to GitHub; final check that no secrets are committed
