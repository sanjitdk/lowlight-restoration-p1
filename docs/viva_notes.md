# Viva Notes and 3-Minute Demo Script

## 3-minute demo (live URL)
| Time | Do | Say |
|---|---|---|
| 0:00–0:20 | Open the URL on a phone or laptop | "Adaptive low-light restoration: classical Retinex, Zero-DCE++, and a hybrid of the two." |
| 0:20–0:50 | Click an example, mode **Hybrid**, Enhance | "It first *measures* the image: brightness, noise, uneven lighting, colour cast. Every parameter comes from those measurements." Open the analysis accordion. |
| 0:50–1:30 | Scroll the intermediate gallery | Illumination map (2) → reflectance (3) → Retinex (4) → denoise (5) → CLAHE (6) → classical (7) → **guidance map (8)**: "bright areas get a weaker curve, so lamps don't blow out" → Zero-DCE++ (9) → fusion weights (10) → result (11) |
| 1:30–2:10 | **Judge's image**: upload it, run Hybrid, then Classical, then DL | Compare the three side by side; point at noise and highlights |
| 2:10–2:40 | Show `eval_P1_*.md` | Hybrid vs. classical vs. DL; the ablation; "the quality selector did *not* beat always-hybrid on validation, so it's off. We report that." |
| 2:40–3:00 | Show a failure case | "Here's where it fails and why" (see `results/failures/`) |

## Likely questions and answers

**Why Zero-DCE++?** It has 10,561 parameters (52 KB) and runs on CPU in under a second at 600×400. It handles any resolution, and its zero-reference losses mean it can train without pairs; we add L1 + SSIM because LOL does provide pairs. EnlightenGAN and RetinexNet are 100–1000× larger and need a GPU to serve.

**How do you know your port is correct?** The official weights load with `strict=True` (no missing or unexpected keys), the parameter count equals the paper's, and pretrained val PSNR is 14.46 dB, in line with published LOL numbers.

**Why did you stop fine-tuning at epoch 1?** Val PSNR went 14.46 → 16.23 → 16.20 → 15.75. It peaked at epoch 1 and then fell twice, so we kept epoch 1, as chosen on val.

**What makes it adaptive?** `analysis.adaptive_params`:
- gamma = log(target)/log(mean brightness)
- CLAHE clip depends on contrast and noise
- white balance only when a colour cast is measured
- denoise strength (k·σ)², with σ measured *after* brightening
- DL curve strength per pixel from the illumination map

**Why measure noise after brightening?** In a dark image, noise is tiny in absolute terms, so the estimate is near zero. Brightening amplifies it, so that's where you have to measure. Our first version measured it on the raw input, and denoising almost never triggered.

**How does the hybrid combine the two?**
1. The illumination map L scales the Zero-DCE++ curve: s = 0.2 + 0.8·(1−L).
2. Both outputs are denoised.
3. Mertens exposure fusion blends them per pixel (weights from contrast, saturation and well-exposedness) in a Laplacian pyramid, so there are no seams.
4. A highlight guard keeps already-bright areas from getting darker.

**What is the quality selector, and why is it off?** Each candidate (hybrid, DL, classical) gets no-reference checks: exposure error, clipping, colour cast, detail loss, noise. Weights were fitted on val. In leave-one-out cross-validation it picked the best candidate only 60% of the time and scored *below* always-hybrid (J 1.234 vs. 1.256). NIQE got zero weight. So it stays disabled, and we report it. NIQE alone would favour over-smoothed images.

**How did you avoid tuning on test?**
- Splits are fixed with seed 42: 435 train / 50 val.
- Everything was tuned on val.
- The config was frozen and its SHA written to `results/test_lock.json` before the test run; `evaluate.py` refuses a re-run with a different config.
- Test was run once.

**Metrics?**
- **PSNR:** pixel error in dB.
- **SSIM:** structure, contrast and luminance similarity.
- **LPIPS:** deep-feature perceptual distance (lower is better).
- **NIQE:** no-reference naturalness (lower is better).
- **Noise suppression:** residual σ (wavelet estimate) vs. GT.
- **Edge preservation:** correlation of Sobel gradient magnitudes, and Canny-edge F1 vs. GT with 1 px tolerance.
- **Latency:** ms per image on CPU.

**Biggest weakness? (Say this before they find it.)**
1. **On unseen LOL-v2, adaptive classical (20.27 dB) beats our hybrid (18.92).** The cause is that **fine-tuning overfit LOL-v1's exposure.** The fine-tuned DL model gains about 2 dB on LOL-v1 but loses 3 dB on LOL-v2 compared with the pretrained model, and the hybrid weights the DL branch ×2.
2. **The fixed-parameter versions get better LPIPS** (about 0.26 vs. 0.31–0.34). The val-tuned denoising (k = 8, radius 7) over-smooths texture, and our val tuning never compared against the fixed baseline. We did not re-tune on test.
3. Illumination guidance is only worth +0.2 to +0.4 dB on LOL, because LOL darkness is mostly uniform.

**Which result are you proudest of?** The LOL-v2 ablation: removing *any* component lowers PSNR (denoise −2.1, fusion −1.4, CLAHE −0.9, guidance −0.4, colour −0.4, Retinex −0.25 dB). Also, our integrity checks caught a selector that didn't help, and turned it off.

**Failure case to show:** LOL-v2 00730, a shaded daytime ATM. DL and hybrid wash it out (+0.16 mean luma vs. GT) because the fine-tuned model has a fixed brightness target; classical adapts its gamma to the input brightness. Fix: make the DL curve strength depend on the measured brightness gap, and tune fusion on a cross-dataset validation set.

**Ethics?** The outputs are plausible reconstructions, not evidence. Not for forensic identification. No uploads are stored.

## Key numbers (val, 50 images)
- Pretrained Zero-DCE++: 14.46 dB → fine-tuned 16.23 dB (DL only)
- Classical (adaptive): 16.18 / 0.727 / 0.345 (PSNR / SSIM / LPIPS)
- Hybrid (adaptive): 17.16 / 0.741 / 0.343
- Selector LOO J 1.234 vs. always-hybrid 1.256 vs. oracle 1.327
