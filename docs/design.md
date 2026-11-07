# Design Document: Adaptive Low-Light Image Restoration (P1)

## 1. Problem
Low-light photos are dark, noisy (sensor noise becomes visible once brightened), colour-shifted, and often lit unevenly (lamps next to deep shadow). The goal is to restore them while keeping edges and texture, without overexposure, halos or unnatural colours, and to show that the system works on images it has never seen.

## 2. Requirements → implementation
| Requirement | Where |
|---|---|
| A1 Illumination estimation and decomposition | `classical.estimate_illumination` (max-RGB, refined by a guided filter); `retinex_enhance` (I = R·L) |
| A2 Retinex-based enhancement | `classical.retinex_enhance`: adaptive gamma on L, recombined with R |
| A3 CLAHE | `classical.apply_clahe`: Lab L channel; clip limit from contrast and noise |
| A4 Edge-preserving denoising | `classical.denoise_guided` (guided filter); `denoise_bilateral` also available |
| A5 Contrast and colour correction | CLAHE + `gray_world_wb` (only when a cast is detected) + `adjust_saturation` |
| A6 Adaptive parameter selection | `analysis.analyze` → `analysis.adaptive_params`; `analysis.post_noise_eps` |
| B1 DL architecture | `dl/zerodce.py`: Zero-DCE++ with official layer names, strict-loaded official weights |
| B2 Fine-tuning | `dl/train.py`: LOL-v1 train split; best epoch by val PSNR |
| B3 Generalisation | Test on LOL-v1 eval15 **and** LOL-v2 Real (a different capture set, never used for anything else) |
| B4 Image-quality-aware strategy | Exposure target from measured brightness; noise-aware denoising; quality selector (validated, see §6.4) |
| C1 Classical + DL integration | `hybrid.HybridEnhancer`: classical L scales the Zero-DCE++ curve per pixel |
| C2 Adaptive select/combine | `hybrid.exposure_fusion` (per-pixel Mertens weights) + `quality_features`/`selector_score` |
| C3 Edges/texture vs. noise | Noise measured *after* brightening; guided filter (edge-aware); edge correlation/F1 reported |
| C4 Extremely dark / non-uniform | Guidance map protects bright regions; highlight guard; adaptive gamma bounded by `gamma_min` |
| Adv. 1 Adaptive vs. fixed | `FIXED_PARAMS` baseline: "classical (fixed)" and "hybrid (fixed)" rows |
| Adv. 2 Three approaches compared | classical / DL (pretrained and fine-tuned) / hybrid rows in `eval_P1_*.md` |
| Adv. 3 Ablation | 6 ablations (`disable=`), §7 |
| Adv. 4 Overexposure, colour shift, detail loss | Highlight guard, conditional WB, detail and noise metrics; failure analysis §8 |
| Adv. 5 Unseen images | LOL-v2 Real test + live demo on judges' images |

## 3. Data
| Split | Source | Size | Used for |
|---|---|---|---|
| train | LOL-v1 `our485` (seed-42 shuffle) | 435 pairs | Gradients for Zero-DCE++ fine-tuning only |
| val | LOL-v1 `our485` (held out) | 50 pairs | **Everything tuned**: epoch choice, rule constants, hybrid settings, selector |
| test_lol | LOL-v1 `eval15` | 15 pairs | Final evaluation, run once |
| test_lolv2 | LOL-v2 Real `Test` | 100 pairs | Final evaluation, run once (unseen capture conditions) |

Lists are in `data/splits/*.txt`. `train.py` asserts train ∩ val = ∅.
MIT-Adobe FiveK was not used: it is about 50 GB and consists of expert retouches of normally exposed RAW photos, not paired low-light data.

**Leakage control.** The first test run writes `results/test_lock.json` with the SHA of the config and weights. `evaluate.py` refuses a later test run with different ones. The config was frozen before any test image was processed.

## 4. Method

### 4.1 Image analysis (`analysis.analyze`)
The analysis measures mean and median luma, dark-pixel ratio (luma < 0.05), highlight ratio (luma > 0.95), noise σ (wavelet estimator), non-uniformity (std/mean of heavily blurred luma), colour cast (max gray-world channel deviation), and RMS contrast.

### 4.2 Adaptive rules (`analysis.adaptive_params`)
- **Gamma:** γ = log(target)/log(mean), clamped to [0.25, 0.9]. Darker images get a stronger lift.
- **CLAHE clip:** grows as contrast falls and shrinks as noise rises (CLAHE amplifies noise).
- **White balance:** applied only if the colour cast exceeds a threshold, so naturally coloured light isn't neutralised.
- **Denoising, post-brightening:** a dark input hides its noise in absolute terms, so σ is estimated *after* Retinex brightening, and the guided-filter eps is set to (k·σ)². The first version estimated σ on the raw input and almost never switched denoising on; see §9.

### 4.3 Classical pipeline
Illumination (max-RGB, guided filter) → Retinex with adaptive γ → guided-filter denoise (stronger in dark regions) → CLAHE → colour correction.

### 4.4 Deep learning: Zero-DCE++
Zero-DCE++ predicts a 3-channel curve map A ∈ [−1, 1] and applies LE(x) = x + A·(x² − x) eight times. We chose it because it is tiny (10,561 parameters, about 52 KB), runs on CPU in well under a second at 600×400, and works on arbitrary resolutions. That suits a free CPU host.
- **Checks:** strict state-dict load of the official weights, the parameter count matches the paper, and pretrained val PSNR is 14.46 dB, in the expected range.
- **Fine-tuning:** zero-reference losses (spatial consistency, exposure, colour constancy, illumination smoothness) plus a supervised L1 + SSIM term, since LOL provides pairs. Adam 1e-4, batch 8, 256 px crops, CPU.

| Epoch | Val PSNR |
|---|---|
| 0 (pretrained) | 14.46 |
| **1 (kept)** | **16.23** |
| 2 | 16.20 |
| 3 | 15.75 |

Val PSNR fell for two epochs, so training was stopped early. With 435 images and a ~10K-parameter model, one epoch already adapts the exposure level; more epochs overfit the curve to the training crops.

### 4.5 Hybrid
1. **Illumination guidance:** curve strength s = s_min + (1 − s_min)(1 − L)^p. Dark regions get the full curve; already-bright regions (lamps, windows) get at most s_min of it. This addresses non-uniform lighting and overexposure.
2. **Noise-adaptive denoising** of the DL output (σ measured after brightening).
3. **Exposure fusion:** per-pixel Mertens weights (contrast × saturation × well-exposedness) blend the classical and DL outputs in a Laplacian pyramid. This avoids seams and halos. `fusion_dl_bias` scales the DL weight.
4. **Highlight guard:** where L > 0.8, the result is never darker than the input.

### 4.6 Quality-aware selector (B4/C2)
Each candidate (hybrid, DL, classical) gets interpretable no-reference features, all lower-is-better: exposure error, clipped and crushed pixels, colour cast, detail loss (Laplacian energy vs. a brightness-normalised input), and residual noise σ. NIQE is optional. Weights are fitted with NNLS on val, against J = PSNR/20 + SSIM − LPIPS, centred per image. The selector is judged by **leave-one-image-out** cross-validation against always-hybrid and an oracle, and is enabled only if it wins.

## 5. Tuning procedure (val only)
`scripts/tune_val.py` uses coordinate descent: each parameter is swept over a small grid with the others held fixed. Stage 1 covers the classical rules, stage 2 the hybrid settings.

**First objective, PSNR/20 + SSIM:** val hybrid 17.62 dB / 0.735. A dry run then showed the adaptive outputs had *worse* LPIPS than the fixed ones (0.46 vs. 0.35). The objective was rewarding over-smoothing.

**Final objective, PSNR/20 + SSIM − LPIPS.** Chosen values:

| Parameter | Value | Note |
|---|---|---|
| denoise_k | 8.0 | Grid edge; eps is capped at 0.05 anyway, so larger k mostly saturates |
| denoise_floor | 0.6 | Denoise blend in bright areas |
| denoise_radius | 7 | Grid edge; trades a little LPIPS for PSNR/SSIM |
| target_lum | 0.35 | Darker than the PSNR-only optimum (0.45); better SSIM/LPIPS |
| clahe_clip_max | 1.5 | Grid edge; strong CLAHE amplifies noise (LPIPS 0.385 → 0.350) |
| saturation_boost | 1.0 | No boost |
| guide_min | 0.2 | Only about 0.1 dB better than no guidance on LOL |
| fusion_dl_bias | 2.0 | Favours the DL branch |

Val result: classical 16.18 / 0.727 / 0.345 → **hybrid 17.16 / 0.741 / 0.343** (PSNR / SSIM / LPIPS).

## 6. Results
Test sets were evaluated **once** with the frozen config. The full tables, generated from `results/*/summary.csv`, are in `eval_P1_T31.md`; they are repeated here.

### LOL-v1 eval15 (15 images)

#### Comparison: classical vs. deep learning vs. hybrid

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/img, CPU) ↓ | Residual noise σ ↓ | Edge corr ↑ | Edge F1 ↑ |
|---|---|---|---|---|---|---|---|---|
| input (no enhancement) | 7.77 | 0.1914 | 0.5596 | 6.749 | 0 | 0.0058 | 0.8383 | 0.6553 |
| classical (fixed params) | 15.41 | 0.7016 | 0.2898 | 5.734 | 247 | 0.0116 | 0.8779 | 0.7954 |
| classical (adaptive) | 16.05 | 0.7208 | 0.3347 | 6.707 | 275 | 0.0085 | 0.8689 | 0.8298 |
| DL: Zero-DCE++ pretrained | 15.14 | 0.5661 | 0.3421 | 7.937 | 739 | 0.0271 | 0.8957 | 0.7283 |
| DL: Zero-DCE++ fine-tuned | 17.33 | 0.5419 | 0.3748 | 7.975 | 743 | 0.0368 | 0.8662 | 0.7455 |
| hybrid (fixed params) | 17.11 | 0.7471 | 0.2612 | 6.021 | 948 | 0.0107 | 0.8912 | 0.8548 |
| hybrid (adaptive) [ours] | 17.21 | 0.7335 | 0.3309 | 6.743 | 993 | 0.0073 | 0.8603 | 0.7704 |
| auto (quality selector) | 17.21 | 0.7335 | 0.3309 | 6.743 | 1021 | 0.0073 | 0.8603 | 0.7704 |

#### Ablation (each row removes one component from the adaptive hybrid)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/img, CPU) ↓ | Residual noise σ ↓ | Edge corr ↑ | Edge F1 ↑ |
|---|---|---|---|---|---|---|---|---|
| hybrid (adaptive) [ours] | 17.21 | 0.7335 | 0.3309 | 6.743 | 993 | 0.0073 | 0.8603 | 0.7704 |
| hybrid - illumination guidance | 17.03 | 0.7315 | 0.3329 | – | 989 | 0.0077 | 0.8659 | 0.7769 |
| hybrid - denoising | 16.24 | 0.4811 | 0.4566 | – | 886 | 0.0483 | 0.8107 | 0.6373 |
| hybrid - fusion (DL branch only) | 18.27 | 0.7227 | 0.3469 | – | 844 | 0.0039 | 0.8568 | 0.7001 |
| hybrid - CLAHE | 16.87 | 0.7195 | 0.3442 | – | 1089 | 0.0043 | 0.8577 | 0.7164 |
| hybrid - colour correction | 17.33 | 0.7351 | 0.3291 | – | 1065 | 0.0073 | 0.8604 | 0.7699 |
| hybrid - Retinex (plain gamma) | 17.64 | 0.7205 | 0.3657 | – | 1131 | 0.0059 | 0.8338 | 0.7048 |

### LOL-v2 Real test (100 images)

#### Comparison: classical vs. deep learning vs. hybrid

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/img, CPU) ↓ | Residual noise σ ↓ | Edge corr ↑ | Edge F1 ↑ |
|---|---|---|---|---|---|---|---|---|
| input (no enhancement) | 9.72 | 0.1960 | 0.5195 | 6.791 | 0 | 0.0067 | 0.8236 | 0.6454 |
| classical (fixed params) | 18.33 | 0.7333 | 0.2605 | 6.269 | 233 | 0.0178 | 0.8758 | 0.7919 |
| classical (adaptive) | 20.27 | 0.7792 | 0.3150 | 6.866 | 265 | 0.0108 | 0.8623 | 0.8152 |
| DL: Zero-DCE++ pretrained | 18.21 | 0.5813 | 0.3157 | 8.142 | 612 | 0.0336 | 0.8902 | 0.6976 |
| DL: Zero-DCE++ fine-tuned | 15.26 | 0.4973 | 0.4041 | 8.325 | 601 | 0.0488 | 0.8356 | 0.7104 |
| hybrid (fixed params) | 18.26 | 0.7508 | 0.2610 | 6.610 | 793 | 0.0171 | 0.8741 | 0.8440 |
| hybrid (adaptive) [ours] | 18.92 | 0.7614 | 0.3357 | 6.889 | 846 | 0.0089 | 0.8417 | 0.6796 |
| auto (quality selector) | 18.92 | 0.7614 | 0.3357 | 6.889 | 849 | 0.0089 | 0.8417 | 0.6796 |

#### Ablation (each row removes one component from the adaptive hybrid)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/img, CPU) ↓ | Residual noise σ ↓ | Edge corr ↑ | Edge F1 ↑ |
|---|---|---|---|---|---|---|---|---|
| hybrid (adaptive) [ours] | 18.92 | 0.7614 | 0.3357 | 6.889 | 846 | 0.0089 | 0.8417 | 0.6796 |
| hybrid - illumination guidance | 18.48 | 0.7601 | 0.3352 | – | 848 | 0.0093 | 0.8471 | 0.6895 |
| hybrid - denoising | 16.80 | 0.4764 | 0.4635 | – | 748 | 0.0612 | 0.7872 | 0.6489 |
| hybrid - fusion (DL branch only) | 17.56 | 0.7256 | 0.3819 | – | 733 | 0.0052 | 0.8175 | 0.5794 |
| hybrid - CLAHE | 18.06 | 0.7357 | 0.3666 | – | 831 | 0.0057 | 0.8237 | 0.6018 |
| hybrid - colour correction | 18.49 | 0.7589 | 0.3465 | – | 817 | 0.0087 | 0.8412 | 0.6794 |
| hybrid - Retinex (plain gamma) | 18.67 | 0.7394 | 0.3794 | – | 840 | 0.0068 | 0.8194 | 0.5893 |


### 6.1 What the numbers say
- **Hybrid vs. single approaches.** On LOL-v1 eval15, the adaptive hybrid (17.21 dB / 0.734) beats adaptive classical (16.05 / 0.721) on PSNR and SSIM. It also beats fine-tuned DL on SSIM (0.542), and is 0.1 dB behind it on PSNR.
- **On the unseen LOL-v2 Real test, adaptive classical is best** (20.27 dB / 0.779), ahead of the hybrid (18.92 / 0.761). The hybrid did *not* generalise better than its classical branch.
- **Fine-tuning overfit to LOL-v1's exposure.** Fine-tuned Zero-DCE++ gains on LOL-v1 (15.14 → 17.33 dB) but **loses 3 dB on LOL-v2** (18.21 → 15.26). With `fusion_dl_bias = 2` chosen on LOL-v1 val, the hybrid inherits this bias. This is the main reason the hybrid trails classical on LOL-v2.
- **Adaptive vs. fixed parameters.** Adaptive is better on PSNR on both test sets (hybrid +0.1 / +0.7 dB; classical +0.6 / +1.9 dB). But **fixed parameters give clearly better LPIPS** (about 0.26 vs. 0.31–0.34) and edge F1. The val-tuned adaptive rules denoise heavily (k = 8, radius 7) and over-smooth texture. Val tuning never compared against the fixed baseline, so this gap was not caught before the test run, and we did not re-tune on test.
- **Noise suppression.** Every adaptive variant reaches lower residual σ than its fixed counterpart, and lower than the DL outputs (0.037–0.049). The GT images themselves have σ ≈ 0.01–0.03, so outputs below that level are over-smoothed.
- **Latency** (laptop CPU, 3 threads): classical ≈ 0.25 s, DL ≈ 0.6–0.75 s, hybrid ≈ 0.85–1.0 s per 600×400 image.
- **Quality selector:** disabled after validation (§4.6). The "auto" rows are therefore identical to the hybrid.


## 7. Ablation
Rows remove one component from the adaptive hybrid (tables above). Differences in PSNR (dB):

| Removed component | LOL-v1 eval15 (15) | LOL-v2 Real (100) | Verdict |
|---|---|---|---|
| Denoising | −0.97 | −2.12 | Largest single contribution; SSIM collapses (0.73 → 0.48) without it |
| CLAHE | −0.34 | −0.86 | Helps on both |
| Fusion (DL branch only) | **+1.06** | −1.36 | Hurts on LOL-v1, helps on LOL-v2: fusion pulls the result toward classical, which generalises better |
| Illumination guidance | −0.19 | −0.44 | Small but consistent gain; LOL is mostly uniformly dark, so a small effect is expected |
| Colour correction | **+0.12** | −0.43 | Mixed; neutral on LOL-v1 |
| Retinex (plain gamma instead) | **+0.43** | −0.25 | Mixed on PSNR; Retinex gives better LPIPS on both (0.331 vs. 0.366, 0.336 vs. 0.379) |

On the larger and unseen LOL-v2 set, **every component contributes positively to PSNR**. On the 15-image LOL-v1 set, three components do not, and we report that as measured.


## 8. Failure analysis
Worst cases by hybrid PSNR (grids: input | classical | DL | hybrid | GT) are in `results/failures/`. The diagnosis is generated automatically from measured statistics (`scripts/failure_cases.py`).

| Image | Pattern | Cause |
|---|---|---|
| LOL-v2 00728–00730, 00718–00719 | **Over-exposed vs. GT (+0.09 to +0.18 mean luma)**, colour shift 0.11–0.20 | These are shaded *daytime* scenes that are only moderately dark. The fine-tuned DL branch has a fixed brightness target learned from LOL-v1 and washes them out, and the hybrid inherits this through fusion (DL weight × 2). The classical branch's gamma rule adapts to the input brightness and stays closer to GT. 00728 is also over-smoothed (σ 0.013 vs. GT 0.035). |
| LOL-v1 55, 23 | **Under-exposed (−0.20 to −0.22)**, strongly non-uniform lighting | The guidance map protects the bright region, but the reference photo is much brighter overall. With `target_lum = 0.35` chosen on val, very dark scenes stay darker than the GT style. |
| LOL-v1 179 | Over-exposed (+0.19), colour shift 0.24 | Mixed lighting; the gray-world assumption is violated by a dominant coloured surface. |

**What would fix these (not done, because it would mean tuning on test):**
- Make the DL exposure target input-adaptive (scale the curve by the measured brightness gap), instead of relying on the fine-tuned global level.
- Tune `fusion_dl_bias` with a cross-dataset validation set.
- Add LPIPS-aware tuning *against the fixed baseline*, to avoid over-smoothing.
- Replace gray-world with a more robust illuminant estimate (for example gray-edge).


## 9. Engineering issues found and fixed
- **Silent noise-estimation failure:** `skimage.estimate_sigma` needs PyWavelets, which was not installed. A broad `except Exception` returned σ = 0, so denoising never ran. Fixed by adding the dependency and narrowing the exception handling.
- **Noise measured in the wrong place:** moved to after brightening (§4.2).
- **Over-smoothing from a PSNR/SSIM-only objective:** fixed by adding LPIPS to the objective (§5).
- **Windows CRLF line endings** broke the LOL-v2 download file list; fixed.

## 10. Deployment
Gradio app (`app.py`) in Docker on **Render's free tier** (https://lowlight-restoration-p1.onrender.com; `render.yaml`). Hugging Face Spaces was the plan, but it now requires a paid plan for Docker/Gradio apps. Inputs are capped at 1024 px locally and 768 px on the deployed instance (`APP_MAX_SIDE`). Errors (corrupt, tiny or oversized files, missing input) appear as messages in the UI, and Gradio provides the loading state. All 11 intermediate stages are shown.

**Fitting a 512 MB host.** The first deploys were killed for running out of memory. Profiling showed three causes, each fixed without changing the results:
- *PyTorch runtime (~150 MB):* the deployed image does not install torch. The Zero-DCE++ curve network is exported to ONNX and run with OpenCV DNN (`src/dl/zerodce_onnx.py`); resizing and the curve LE(x) are done in NumPy, matching `zerodce.py`. Full-pipeline max difference vs PyTorch: 5×10⁻⁴ (< 1/255); `tests/test_onnx_backend.py` checks it. Training and evaluation still use PyTorch.
- *OpenCV DNN conv buffers (~330 MB for a 600×400 frame):* the network runs in 96-row strips with an 8-row halo. The receptive radius is 7 px, so the output is identical.
- *Full-size decode of phone photos (~150 MB):* large JPEGs are decoded at reduced scale before the final resize.

Peak memory went from >530 MB to ~310 MB. Previews are sent as JPEG files (PNG encoding of ~13 images per click dominated response time on the shared CPU), and heavy requests share one queue.

**Measured on the live URL** (warm, median of 3, upload → result): hybrid 5.6 s at 600×400 and 7.4 s for a 4000×3000 photo; classical 2.5 / 5.9 s; Zero-DCE++ 4.3 / 6.4 s. Full table in `eval_P1_*.md` and `results/deploy_latency.json`. A sleeping free instance needs ~30–60 s to wake on the first request.

## 11. Limitations
- Tuned and fine-tuned only on LOL (indoor, static scenes, mostly uniform darkness). The illumination guidance, meant for non-uniform scenes, barely registers in LOL metrics.
- Several tuned values sit at the edge of their grids, so the optimum may lie further out.
- The quality selector did not beat always-hybrid. Simple no-reference features predict the best candidate only about 60% of the time, and NIQE added nothing on val.
- PSNR/SSIM penalise any tone difference from one photographer's reference. Some "failures" are plausible alternative exposures.
- CPU latency is around a second per image at 600×400 on a laptop and 3–4 s on the free host's shared CPU, and grows with resolution (hence the size caps). The free host sleeps when idle (~30–60 s first load).

## 12. Ethics
- Enhancement *invents* plausible detail and tone. The outputs must not be used as forensic or evidential images (for example to identify faces or number plates in surveillance footage) or presented as what the scene "really" looked like.
- Low-light enhancement can make covert or night-time surveillance easier. The app is meant for personal photography and research.
- The app does not store uploaded images, and no personal data is collected. The datasets (LOL, LOL-v2) are public research datasets used under their research terms.
- Results are reported as measured, including where the proposed hybrid or a component does not help.
