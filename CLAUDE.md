# CLAUDE.md — CV Hackathon 2026, Problem P1 (Low-Light Image Restoration)

## Project overview
Solo hackathon entry. It builds a low-light image enhancement system with three approaches:
- **Classical:** Retinex + illumination estimation + CLAHE + guided/bilateral denoise + colour correction, with adaptive params
- **Deep learning:** Zero-DCE++, pretrained and then fine-tuned on LOL-v1
- **Hybrid:** the classical illumination map guides the DL output, followed by adaptive denoise, fusion and a validated quality selector

It is deployed as a Gradio app on Hugging Face Spaces.

Key docs: `goal.md` (what done means), `plan.md` (approach + phases), `todolist.md` (progress checklist).

## Environment
- Windows 11, Python 3.9 local, **no local GPU**. Training runs on Kaggle/Colab GPU.
- Hosting: Hugging Face Spaces (CPU, Gradio).

## Conventions
- Source code lives in `src/` (modular: `analysis`, `classical`, `dl/`, `hybrid`, `metrics`, `io_utils`). Runnable scripts live in `scripts/`.
- `data/` is gitignored. **Never commit datasets, kaggle.json or tokens.**
- Split: LOL-v1 our485 → 435 train / 50 val (seed 42). Test = LOL-v1 eval15 + LOL-v2 Real test (100).
- **Tune only on val.** Test runs once with `configs/final.yaml`; no changes after seeing test numbers.
- Results come only from `scripts/evaluate.py`. Never hand-edit numbers. Report losses honestly.
- After each completed step, tick it in `todolist.md` and add an entry to the Progress log below.

## Commands (fill in as they're created)
- Install: `pip install -r requirements.txt`
- Run app: `python app.py`
- Evaluate (dev): `python scripts/evaluate.py --split val`
- Evaluate (final, once): `python scripts/evaluate.py --split test --config configs/final.yaml`
- Batch inference: `python scripts/run_inference.py --input_dir examples --output_dir results/inference --mode all [--save_intermediates]`
- Report figures: `python scripts/make_figures.py` → `docs/figures/`

## Progress log
### 2026-11-06
- Read the problem statement (P1) and the rubric/instruction sheet.
- Decisions: solo; Zero-DCE++ (tiny, fast on CPU) as the DL model; Kaggle GPU for fine-tuning; Hugging Face Spaces for hosting; MIT-FiveK skipped.
- Integrity rules added after review: measured-only reporting, val-only tuning, combined quality selector that must be shown to help, strict pretrained-weight check, latency measured on the deployed CPU.
- Created `plan.md`, `goal.md`, `todolist.md`, `CLAUDE.md`. **Next: Phase 1 (setup).**
- **Phase 1 (setup):** folders, `requirements.txt` (light, for the app) + `requirements-dev.txt` (lpips, pyiqa, pandas, kaggle…), `.gitignore`, `git init`, `scripts/make_splits.py` (our485 → 435/50, seed 42). The venv install was still running at this point.
- **Phase 2 (data):** blocked; waiting for `~/.kaggle/kaggle.json` from the user.
- **Phase 3 started early (no data needed):** `src/io_utils.py` (safe loading: alpha, gray, 16-bit, corrupt, tiny, huge → max side 1024; `ImageInputError` for user-facing errors), `src/analysis.py` (`analyze()` stats + `adaptive_params()`; `FIXED_PARAMS` is the non-adaptive baseline), `src/classical.py` (guided-filter illumination, Retinex, guided denoise weighted to dark regions, CLAHE on Lab-L, gray-world WB + saturation; `disable=` for ablations; returns intermediates). `tests/test_smoke.py` uses a synthetic dark image + bad inputs.
- **Phase 4 started early:** `src/dl/zerodce.py` is ported from the official Zero-DCE++ (layer names match the checkpoint, strict load, expected 10,561 params; `strength` scalar or map for hybrid guidance). The official weights were downloaded to `weights/zerodcepp_pretrained.pth` (52 KB). `src/dl/train.py` covers zero-reference losses + optional supervised L1+SSIM (`--sup`), best epoch chosen by val PSNR, train/val overlap asserted.
- **Phase 2 (data) done without Kaggle:** used public Hugging Face mirrors. LOL-v1: `geekyrakshit/LoL-Dataset` (347 MB zip, 485 + 15 pairs, `.DS_Store` removed). LOL-v2 Real test: `okhater/lolv2-real` (Test/Input + Test/GT, 100 pairs). Splits: train 435 / val 50 / test_lol 15 / test_lolv2 100, no overlap. (The first LOL-v2 attempt failed on Windows CRLF line endings in the file list; fixed with `tr -d '\r'`.)
- **Checks passed:** smoke test (synthetic dark astronaut: classical 6.8 → 14.1 dB; bad inputs handled). Zero-DCE++ strict load OK, 10,561 params. **Pretrained val PSNR = 14.46 dB**, in line with published LOL numbers, so the port is correct.
- **Bug found and fixed:** `estimate_sigma` needs PyWavelets (missing). A broad `except Exception` silently returned noise=0, so denoising never ran. Added `PyWavelets==1.6.0`, narrowed excepts to `(ValueError, FloatingPointError)`. **Design change:** noise is now measured *after brightening* (`analysis.post_noise_eps`), because dark inputs hide noise in absolute terms; eps = (k·σ)². On 3 val images SSIM rose from ~0.21 to ~0.49 as k increased. Tuning on the full val set is pending.
- **Fine-tune on local CPU** instead of Kaggle (Zero-DCE++ is tiny): `python -m src.dl.train --epochs 12 --bs 8 --crop 256 --sup 1.0`, launched as a detached process; log in `results/finetune_log.txt`. Epoch 1 val = 16.23 dB (from 14.46). About 6 min/epoch.
- **Written (Phase 5–7 code):** `src/hybrid.py` (`HybridEnhancer`: modes hybrid/auto/classical/dl; ablation `disable=`), `src/metrics.py`, `scripts/tune_val.py` (coordinate descent on val, J = PSNR/20 + SSIM → `configs/final.yaml`), `scripts/fit_selector.py` (NNLS + LOO; enables the selector only if it beats always-hybrid), `scripts/evaluate.py` (all methods + 6 ablations; `results/test_lock.json` refuses re-running test with a changed config), `app.py` (Gradio 4.44; intermediates gallery, errors via `gr.Error`), `Dockerfile` (python:3.10-slim, uid 1000, port 7860).
- **Fine-tune stopped after epoch 3** (val 16.23 → 16.20 → 15.75 dB). Saved weights = epoch 1. History is in `weights/zerodcepp_finetuned.history.json`.
- **Tuning run 1** (J = PSNR/20 + SSIM): hybrid val 17.62 dB / 0.735. A 2-image dry-run of `evaluate.py` showed adaptive LPIPS much worse than fixed (0.46 vs. 0.35), i.e. this objective favoured over-smoothing. Kept as `configs/tuned_psnr_ssim_only.yaml` and `results/*_psnr_ssim_only.*`.
- **Tuning run 2 (final)**: J = PSNR/20 + SSIM − LPIPS, still val only. Chosen: denoise_k 8, floor 0.6, radius 7, target_lum 0.35, clahe_clip_max 1.5, saturation 1.0, guide_min 0.2, fusion_dl_bias 2.0. **Val:** classical 16.18 / 0.727 / 0.345; hybrid **17.16 / 0.741 / 0.343** (PSNR / SSIM / LPIPS). Findings: illumination guidance adds only ~0.1 dB on LOL (uniform darkness); strong CLAHE hurts LPIPS; several choices sit at the grid edge (denoise_k, radius, clahe).
- **Selector**: disabled. Val LOO J 1.234 < always-hybrid 1.256 (oracle 1.327); NIQE got weight 0. See `results/selector_val*.json`.
- **Config frozen** (`configs/final.yaml` sha 99acdc7c…, weights sha e48336cf…; `results/test_lock.json`). Test evaluation launched on test_lol (15) and test_lolv2 (100) in parallel. Do not change `configs/final.yaml` or `src/` behaviour after this point.
- Written: `scripts/make_eval_md.py` (eval file from CSVs), `scripts/failure_cases.py` (worst-k grids + automatic diagnosis), `README.md` (HF Spaces header: sdk docker, port 7860), `docs/architecture.md`, `examples/` (4 test images for the demo).
- **Test results (run once).** LOL-v1 eval15: hybrid 17.21 / 0.734 / 0.331 vs. classical 16.05 / 0.721 / 0.335 vs. DL-ft 17.33 / 0.542 / 0.375. **LOL-v2 Real (unseen): classical 20.27 / 0.779 / 0.315 beats hybrid 18.92 / 0.761 / 0.336.** DL fine-tuned 15.26 vs. pretrained 18.21: **fine-tuning overfit LOL-v1 exposure**, and the hybrid inherits it (fusion_dl_bias 2). Fixed-param variants have better LPIPS (~0.26), so adaptive denoising over-smooths. Ablation on LOL-v2: every component helps PSNR. On LOL-v1, removing fusion/colour/Retinex raises PSNR. Everything is reported as measured in `eval_P1_TXX.md` and `docs/design.md` §6–8. Not re-tuned (that would use test).
- **Failure cases** (`results/failures/`): LOL-v2 shaded daytime scenes over-exposed by the DL branch; LOL-v1 very dark non-uniform scenes under-exposed vs. GT.
- **App fixes after local server testing:** (1) HTTP 500 caused by gradio 4.44 + pydantic ≥2.11 schema bug, so pinned `pydantic==2.10.6`. (2) A corrupt upload leaked a raw error with a server path through `gr.Image`, so the input is now `gr.File` and our loader validates it. (3) The default mode is now Hybrid, and Auto is labelled experimental (selector disabled). UI-only changes: no change to enhancement code or config after the test run.
- Local latency via HTTP: 600×400 about 0.9–1.5 s processing / 2.6–3.0 s round-trip; 4000 px photo → 1024 px, 2.1–2.5 s / 5.9–6.7 s.
- `.gitignore`: excludes `results/*/images/` (127 MB, regenerable).
- **Pending:** HF login (`huggingface-cli login`), GitHub repo + push (gh is logged in as sanjitdk), Team ID + USN, deployed-latency measurement.
- **UI redesign (`app.py`, user request "improvise on the UI"):** gradient header with method chips; numbered steps (① photo, ② method); **custom before/after drag slider** (`slider_html`: HTML + CSS clip-path, driven by a `<script>` in `gr.Blocks(head=...)` with event delegation). `gradio_imageslider` was tried but fails on local Python 3.9, so it was removed to avoid an extra dependency. Stat cards replace the raw JSON (time, brightness in→out, noise after brightening, unevenness, colour cast, γ/CLAHE). Tabs: Result / **Compare all 3 methods** (`compare_all`, per-method time + no-reference indicators) / Pipeline steps / About & results (tables read from `results/*/summary.csv`, not typed). Thumbnail example gallery (tap → runs); auto-run on upload; download (PNG) hidden until there is a result. Mobile: verified at a true 390 px via an iframe wrapper (headless Chrome's minimum viewport is 512 px, so plain screenshots were misleading).
- `.dockerignore` now ships `results/*.csv` (for the About tab) and excludes only result images and logs.
- Local check: hybrid 600×400 ≈ 0.8–1.6 s processing; compare-all on a 4000 px photo 6.2 s round-trip. Bad inputs show clean messages.
- How to run locally: `.venv\Scripts\python app.py`, then open **http://localhost:7860** (not the printed 0.0.0.0 address). Phone on the same Wi-Fi: http://<laptop-IP>:7860.
- **UI redesign v2** (feedback: "looks like AI slop"): removed the gradient hero, glow, emoji, pill chips, ①② markers and uppercase card labels. Now: plain top bar (title, subtitle, links to Source / Design document / Evaluation); neutral `gr.themes.Base` (neutral hue, black/white primary button, no blue accents, small radius, no shadows); sidebar (Input, Method, Enhance, Examples thumbnails) + main tabs (Result, Comparison, Pipeline, About). Result = slim-handle comparison slider + three key/value tables (Run, Measured on input, Adaptive parameters) with tabular numbers. Pipeline maps use viridis. The Auto selector is removed from the UI (disabled in config; documented in design.md §4.6). Verified by driving headless Chrome over CDP (click example → screenshot): desktop light/dark and a true 390 px phone view.

### 2026-11-07
- **Hosting changed to Render.** Hugging Face now requires PRO for Gradio/Docker Spaces (HTTP 402 on create), so we deploy on Render free (512 MB RAM). `render.yaml` is a Blueprint: Docker, free plan, PORT 7860, APP_MAX_SIDE 768, single-thread env vars.
- Deploy crash 1: `huggingface_hub>=1.0` removed `HfFolder` (imported by gradio 4.44). Fixed by pinning gradio's dependencies to the versions tested locally (`requirements.txt`).
- Deploy crash 2: out of memory on the first hybrid request (PyTorch ≈150 MB + processing). **Fix: the deployed app does not use PyTorch.** `src/dl/zerodce_onnx.py` runs the same Zero-DCE++ curve network from `weights/zerodcepp_finetuned.onnx` with OpenCV DNN; resizing and the curve are done in NumPy to match the PyTorch code. `src/hybrid.py` uses it only when torch is not installed. Checked: full-pipeline max |diff| vs PyTorch = 5e-4 (< 1/255) over all modes; `tests/test_onnx_backend.py`. torch moved to `requirements-dev.txt` (training/evaluation unchanged). Local peak memory without torch at 768 px: ~406 MB (Windows). Heavy UI events share one queue (`concurrency_id="cpu"`).
- `app.py`: optional `APP_MAX_SIDE`, `TORCH_THREADS`, `CV_THREADS` env vars.
- Still OOM after the ONNX switch: OpenCV DNN's conv buffers for a full 600×400 frame took ~330 MB, and a 4000 px photo decoded at full size took ~150 MB more. Fixes: `ZeroDCEppONNX` runs the network in 96-row strips with an 8-row halo (exact: receptive radius is 7), and `app._open_small` decodes big JPEGs at reduced scale (PIL draft) before the final resize. Local peak without torch, all modes + compare: ~310 MB (600×400 and 4000 px).
- **Submission report** `REPORT_P1.md` (hackathon template), filled from `eval_P1_TXX.md` / `docs/design.md`. Added `scripts/run_inference.py` (folder CLI, frozen config) and `scripts/make_figures.py` (3 figures in `docs/figures/`: extreme low-light, non-uniform, intermediates). No change to `src/` or config.
- **Deployed latency measured** (Render, warm, 600×400, 4 req/mode): processing classical 0.55–0.87 s, DL 2.8–3.9 s, hybrid 3.4–4.3 s; round-trip 7–16 s. Live URL: https://lowlight-restoration-p1.onrender.com. Still pending: Team name/ID + USN.
- Speed on Render: Gradio was PNG-encoding ~13 arrays per click (slow on the shared CPU). Previews are now JPEG files (pipeline ≤480 px), the download PNG uses light compression, and temp files are cleaned.
- **Live: https://lowlight-restoration-p1.onrender.com** (Render free). Measured warm, median of 3, upload → result: hybrid 5.6 s (600×400) / 7.4 s (4000×3000 photo); classical 2.5 / 5.9 s; DL 4.3 / 6.4 s → `results/deploy_latency.json`, eval file, design.md §10. Cold start after ~15 min idle ≈ 30–60 s (Render docs, not measured). Checked with headless Chrome at phone and laptop widths.
- Google Cloud Run was tried but needs a ₹1000 prepayment, so it was dropped (an empty project `lowlight-p1-sdk` exists without billing).
- **Remaining:** Team ID + USN → `python scripts/make_eval_md.py --team <ID> --usns "<USN>"`, rename the eval file, update the app's Evaluation link.
