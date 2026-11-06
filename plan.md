# Plan — P1: Robust Low-Light Image Restoration (CV Hackathon 2026)

## Context
Solo entry for Problem Statement P1. The goal is to finish in about 3 hours (a bit more is OK), with a 24-hour hard deadline.
The project folder `C:\Users\Sanjit DK\OneDrive\Desktop\CVHackathon` is empty. The laptop has Python 3.9 and no GPU, so training runs on a Kaggle/Colab GPU and hosting is on Hugging Face Spaces (Gradio, CPU).

**Rubric (100 pts) and how this plan targets it:**
| Criterion | Pts | How we get top band |
|---|---|---|
| Evaluation metrics | 15* | Strong PSNR/SSIM/LPIPS/NIQE on LOL via the hybrid; all metrics in `eval_P1_<TeamID>.md` in PS order |
| Implementation | 15* | Modular `src/`, proper train/val/test split, works on unseen images, bad input handled |
| UI/UX + deploy | 15 | HF Spaces URL that works on phone, responds in under 10 s on CPU (Zero-DCE++ is about 10K params), loading and error states, Dockerfile, no secrets |
| Documentation | 15 | Architecture diagram matching the code, design doc, README that reproduces results, limitations and ethics |
| Demo + viva | 25 | 3-min demo script plus a viva cheat-sheet (every choice, every metric, one failure case) |
(*the rubric PDF's table is inconsistent; the stated weights are 25 for demo and 15 for each other criterion)

**Mandatory deliverables:** live URL; GitHub repo with source, architecture diagram, design doc, `eval_P1_<TeamID>.md` (first line = comma-separated USNs, metrics in PS order: PSNR, SSIM, LPIPS, NIQE, latency, noise suppression/edge preservation), ablation study and failure-case analysis.

## Approach (recommended)
- **Classical:** a LIME/Retinex-style pipeline. Illumination is the max-RGB map, refined with a guided filter. Reflectance is R = I / L, and L gets an adaptive gamma. CLAHE runs on the Lab L channel. Denoising uses a guided or bilateral filter, and colour is corrected with gray-world white balance and saturation control. All parameters are chosen from image statistics.
- **Deep learning:** **Zero-DCE++** (official pretrained weights from Li-Chongyi/Zero-DCE_extension), fine-tuned for a few epochs on the LOL-v1 train split with its no-reference losses on a Kaggle GPU. It is tiny and fast on CPU, which suits HF Spaces. Its quality-aware part is the exposure-level target, set from the image's measured brightness.
- **Hybrid (core contribution):**
  1. `analyze()` computes mean luminance, dark-pixel ratio, noise σ (`skimage.estimate_sigma`), illumination non-uniformity (std of blurred luma), colour cast, and a highlight ratio.
  2. The classical illumination map L guides the DL step. The Zero-DCE++ input is pre-conditioned, and its curve strength is scaled per pixel by (1 − L), so dark regions get boosted and bright regions are protected. This handles non-uniform lighting and prevents overexposure.
  3. Noise-adaptive guided-filter denoising, with strength set by σ and applied mostly in dark regions (where noise gets amplified). Edges are preserved because the filter is guided.
  4. **Adaptive fusion:** pixel-wise Mertens-style weights (well-exposedness × contrast × saturation) blend the classical and DL outputs. A highlight-clip guard and colour-shift guard (chroma deviation vs. input) step in where needed.
  5. **Quality-aware selection (combined, not a single metric):** NIQE/BRISQUE alone can favour over-smoothed images, so each candidate (hybrid-fused, DL-only, classical) gets a weighted combination of interpretable checks:
     - exposure: mean luminance near a target, and the fraction of clipped highlights and crushed shadows
     - colour shift: chroma deviation from the input, measured after gray-world correction
     - detail retention: gradient energy relative to the input, which penalises over-smoothing
     - residual noise: estimated σ in flat regions
     - a no-reference IQA score (NIQE or BRISQUE)
     The weights are fitted on the **validation set only**, by checking how well the selector picks the candidate with the best PSNR, SSIM and LPIPS. We then **report whether selection helps**: selector vs. always-fused vs. an oracle (best per image) on test. If it doesn't help, we say so and the app defaults to always-fused.
- **Data split (no leakage):** LOL-v1 `our485` → 435 train / 50 **val** (fixed seed, list saved in `data/splits/`).
  - **Val is used for everything that gets tuned:** fine-tune checkpoint/early stopping, the hybrid's adaptive-parameter rules and thresholds, fusion and selector weights, and choosing ablation variants.
  - **Test sets are run once, at the end, with frozen config**: LOL-v1 `eval15` and **LOL-v2 Real test (100)** (unseen paired set, different capture conditions). The frozen config is saved to `configs/final.yaml` with a git commit hash in the eval file, and no parameter is changed after seeing test numbers.
  - DICM/LIME-style real photos plus own phone shots form the unseen no-reference set (NIQE and visual check only).
  - MIT-FiveK is skipped (around 50 GB, not low-light paired). This is stated in the docs.
- **Metrics:** PSNR, SSIM (`skimage`), LPIPS (`lpips`, alex), NIQE (`pyiqa`), latency (ms/image, CPU), noise suppression (estimated σ out vs. GT and in flat regions), edge preservation (gradient-magnitude correlation and Canny-edge F1 vs. GT).
- **Comparisons:** classical-only vs. DL-only (pretrained and fine-tuned) vs. hybrid, plus fixed-param vs. adaptive. **Ablations:** hybrid minus {Retinex illumination guidance, CLAHE, denoise, fusion (DL only), colour correction, adaptive params → fixed}.

## Repo layout (to create in the project folder)
```
CVHackathon/
├── goal.md            # what "done" means, rubric targets, scope
├── plan.md            # copy of this plan (phases + time budget)
├── todolist.md        # step-by-step checklist, ticked as we go
├── CLAUDE.md          # running log of what has been done + conventions
├── README.md          # setup, one-command run, reproduce results, live URL
├── app.py             # Gradio app (HF Spaces entrypoint)
├── Dockerfile  requirements.txt  .gitignore (data/, weights cache, kaggle.json)
├── src/
│   ├── analysis.py    # image statistics → adaptive params
│   ├── classical.py   # illumination est., Retinex, CLAHE, guided/bilateral, colour
│   ├── dl/zerodce.py  # Zero-DCE++ model + inference
│   ├── dl/train.py    # fine-tune script (Kaggle notebook calls it)
│   ├── hybrid.py      # guided DL + denoise + fusion + quality-aware selection
│   ├── metrics.py     # PSNR/SSIM/LPIPS/NIQE/latency/noise/edge
│   └── io_utils.py    # safe load, resize cap (max side 1024), validation
├── scripts/evaluate.py   # runs all methods + ablations → results/*.csv + eval md
├── scripts/failure_cases.py
├── notebooks/kaggle_finetune.ipynb
├── weights/zerodcepp_finetuned.pth   (small, ~40 KB → commit)
├── results/ (tables, figures, failure grids)
├── docs/design.md  docs/architecture.md (Mermaid + PNG)  docs/viva_notes.md
└── eval_P1_<TeamID>.md
```

## Phases and time budget (~3–3.5 h; the extra time goes on the checks, not new features)
1. **Setup (10 min):** write goal.md, plan.md, todolist.md, CLAUDE.md; create venv, requirements, folders, git init.
2. **Data (15 min):** download LOL-v1 and LOL-v2 (Kaggle CLI or manual) into `data/` (gitignored). Write split files.
3. **Classical pipeline + analysis (30 min):** `analysis.py`, `classical.py`, with intermediates returned as a dict for the UI.
4. **DL (40 min, overlapping):** port Zero-DCE++ from the official repo (Li-Chongyi/Zero-DCE_extension), then check it before going further:
   - **Architecture check:** `load_state_dict(strict=True)` with the official pretrained `.pth` must succeed with no missing or unexpected keys, and the parameter count must be about 10K. Run it on 3 LOL val images and confirm the pretrained PSNR is in the range the paper reports. If the numbers are far off, debug the port before fine-tuning.
   - **Fine-tune:** push the Kaggle notebook (about 10–20 epochs, train split only), keep the checkpoint with the best **val** score, and download the weights. Fine-tuned vs. pretrained is reported on test as a comparison, even if fine-tuning doesn't help.
5. **Hybrid (25 min):** `hybrid.py` with illumination-guided curve scaling, adaptive denoise, fusion, guards and selection.
6. **Evaluation (25 min):** `scripts/evaluate.py` runs every method and ablation on LOL eval15 and LOL-v2 real test, plus NIQE on the unseen set. Writes the CSV, the `eval_P1_<TeamID>.md` and comparison grids. The failure-case script saves the worst 5 with a diagnosis.
7. **App + deploy (25 min):** Gradio UI with upload or example images, mode selector (Auto-Hybrid / Classical / DL), before/after slider, intermediates gallery (illumination map, reflectance, DL output, fusion weights, denoise), NIQE and latency readout, and friendly errors for non-image or huge files. Push to HF Spaces and test on phone.
8. **Docs (15 min):** README, design.md (problem, data, method, architecture, adaptive logic, results, ablation, failures, limitations, ethics), architecture diagram, viva_notes.md and a 3-min demo script. Push to GitHub.

## What goes in the four requested files
- **goal.md:** the challenge question, success criteria mapped to every mandatory requirement (A1–A6, B1–B4, C1–C4, advanced 1–5, metrics), deliverables checklist and rubric targets.
- **plan.md:** this plan (approach, architecture, data split, phases and time budget, risks).
- **todolist.md:** granular `- [ ]` checklist per phase, each item tied to a file. It gets ticked off after each step.
- **CLAUDE.md:** project overview, conventions (run commands, folder meanings, never commit data or secrets), plus a dated "Progress log" section appended after every completed step (what was built, key decisions, current metrics).

## Risks and fallbacks
- Kaggle fine-tune slow or failing → ship pretrained Zero-DCE++ (still counts as "adapt") and note it. CPU fine-tune of 2–3 epochs on 256 px crops is also feasible.
- `pyiqa` heavy on HF CPU → use NIQE only offline in eval. The app shows a fast proxy (or BRISQUE via `piq`/OpenCV) to stay under 10 s.
- Large uploads → resize to max 1024 px side before processing.
- USNs and Team ID unknown → placeholders in `eval_P1_TXX.md`, which the user fills in.

## Honest reporting rule
Targets such as "hybrid ≥ classical and DL, about 20 dB PSNR / 0.75 SSIM on LOL" are **hopes, not expected results**. `eval_P1_<TeamID>.md` and the design doc report the **measured** numbers for every method and ablation, including cases where the hybrid loses on a metric or an ablated component turns out not to help. Each loss gets a short explanation in the failure and limitations analysis. Numbers are never hand-edited; they are generated by `scripts/evaluate.py` from the frozen config.

## Verification
- `python scripts/evaluate.py --split val` runs end to end (used during development). `--split test` is run **once** at the end with `configs/final.yaml`. The results tables are generated automatically and committed as-is.
- Zero-DCE++ strict weight load passes. The pretrained model's PSNR on val matches the expected range.
- Selector check: on test, compare selector vs. always-fused vs. oracle and report whichever wins.
- **Deployment performance:** measure the latency of each mode on the HF Spaces CPU (not just locally) for 600 px, 1024 px and 4000 px (downscaled) phone photos. Log cold start vs. warm. If hybrid mode is over about 10 s, cut the work in the app: skip the NIQE score, reduce the max side, or cache the model. The measured numbers go in the eval file's latency row.
- Unit smoke test: run every mode on a black image, a white image, a grayscale image, a PNG with alpha, a tiny 8×8 image and a corrupt file. Each one either produces no crash or shows a clear error.
- `docker build . && docker run -p 7860:7860` works locally (or `python app.py`).
- The HF Space URL opens on laptop and phone, processes a new phone photo in under 10 s, and shows the intermediates.
- Check the docs: the diagram's modules match the `src/` files, and the README commands reproduce `results/`.
