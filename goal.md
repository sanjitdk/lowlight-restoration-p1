# Goal — P1: Robust Low-Light Image Restoration and Enhancement

## Challenge question
Design an intelligent, illumination-adaptive image restoration system. It should enhance severely underexposed, noisy and colour-distorted images while preserving fine textures, edges and structure across different lighting conditions. It must do so without overexposure, halo artifacts or unnatural colours.

## Definition of done
A **deployed** low-light restoration app (public URL, hosted on Render free tier) with adaptive enhancement, comparative evaluation, intermediate visualizations and documented failure analysis. All numbers are **measured, not assumed**.

## Requirement coverage (each must be met and point to code)
### A. Classical CV → `src/classical.py`, `src/analysis.py`
- [ ] A1 Illumination estimation and decomposition (max-RGB + guided-filter refinement, I = R·L)
- [ ] A2 Retinex-based enhancement (reflectance + adaptive gamma on L)
- [ ] A3 CLAHE (Lab L channel, clip limit chosen adaptively)
- [ ] A4 Edge-preserving denoising (guided or bilateral filter)
- [ ] A5 Contrast and colour correction (gray-world WB, saturation control)
- [ ] A6 Adaptive parameter selection from image statistics (`analyze()`)

### B. Deep Learning → `src/dl/`
- [ ] B1 Zero-DCE++ implemented, with the official pretrained weights loading strictly
- [ ] B2 Fine-tuned on the LOL-v1 train split (Kaggle GPU); best checkpoint chosen on val
- [ ] B3 Generalization shown on LOL-v2 Real test + unseen real photos
- [ ] B4 Image-quality-aware strategy (exposure target from measured brightness + combined quality selector)

### C. Hybrid → `src/hybrid.py`
- [ ] C1 Classical illumination map guides the DL curve strength
- [ ] C2 Adaptive fusion and selection of outputs (Mertens weights + validated selector)
- [ ] C3 Edge/texture preserved, noise amplification minimized (noise-adaptive guided denoise)
- [ ] C4 Extremely dark and non-uniform inputs handled (per-pixel guidance, highlight guard)

### Advanced
- [ ] 1 Adaptive (not fixed) parameters, with fixed-vs-adaptive comparison reported
- [ ] 2 Classical vs. DL vs. Hybrid compared
- [ ] 3 Ablation of each hybrid component
- [ ] 4 Overexposure, colour shift and detail loss measured and minimized
- [ ] 5 Tested on unseen images (LOL-v2 Real test, real photos)

### Metrics (this order in the eval file)
PSNR · SSIM · LPIPS · NIQE · Latency · Noise suppression / Edge preservation

## Integrity rules
- Tune **only on val** (50 images held out from LOL-v1 our485). Test sets (LOL-v1 eval15, LOL-v2 Real test) are run **once** with the frozen `configs/final.yaml`.
- Report measured results even when the hybrid loses. "≈20 dB / 0.75 SSIM" is a hope, not a promise.
- The quality selector must be shown to help (selector vs. always-fused vs. oracle); otherwise it is disabled and that is reported.
- No secrets (kaggle.json, HF token) committed.

## Mandatory deliverables
- [ ] Live hosted app URL (works on phone and laptop, < ~10 s, loading and error states)
- [ ] GitHub repo: source code, Dockerfile / one-command run
- [ ] System architecture diagram (matches the code)
- [ ] Design document (every section, incl. limitations and ethics)
- [ ] `eval_P1_<TeamID>.md` (first line = comma-separated USNs; metrics in PS order)
- [ ] Ablation study + failure-case analysis
- [ ] Viva notes + 3-minute demo script

## Rubric targets
| Criterion | Max | Target band |
|---|---|---|
| Evaluation metrics | 15 | Top: strong measured metrics |
| Implementation | 15 | Top: works on unseen inputs, proper split, modular, bad input handled |
| UI/UX + deployment | 15 | Top: public URL, fast, clear, Dockerfile |
| Documentation | 15 | Top: diagram = code, full design doc, reproducible README |
| Demo + viva | 25 | Top: clear 3-min demo on a judge's image; can explain every choice and a failure |

## Out of scope
MIT-Adobe FiveK (around 50 GB, not paired low-light), and training large models (EnlightenGAN/RetinexNet) given the time budget.
