# System Architecture

Every box below is a function in `src/`. File and function names are shown, so the diagram can be checked against the code.

```mermaid
flowchart TD
    U[User upload / judge image<br/>app.py: run] --> IO[io_utils.load_image<br/>validate, alpha/gray/16-bit, resize ≤1024px]
    IO -->|ImageInputError| ERR[gr.Error message in UI]
    IO --> AN[analysis.analyze<br/>mean/median luma, dark ratio, highlight ratio,<br/>noise σ, non-uniformity, colour cast, contrast]
    AN --> AP[analysis.adaptive_params<br/>gamma, CLAHE clip, WB on/off, saturation<br/>rules tuned on val → configs/final.yaml]

    subgraph CLASSICAL [Classical branch — classical.enhance_classical]
        IL[estimate_illumination<br/>max-RGB + guided filter → L] --> RX[retinex_enhance<br/>R = I/L, L^γ, recombine]
        RX --> PN[analysis.post_noise_eps<br/>σ measured AFTER brightening → eps]
        PN --> DN[denoise_guided<br/>edge-preserving, dark-weighted]
        DN --> CL[apply_clahe<br/>Lab L channel]
        CL --> CC[gray_world_wb + adjust_saturation]
    end

    subgraph DL [Deep-learning branch — dl/zerodce.py]
        GM[guidance map<br/>s = s_min + 1-s_min · 1-L ^p] --> ZD[ZeroDCEpp.curve_map → A<br/>apply_curve x + s·A·x²-x, 8 iters]
        ZD --> PN2[post_noise_eps + denoise_guided]
        PN2 --> CC2[colour correction]
    end

    AP --> IL
    AP --> GM
    IL -->|L| GM
    IO --> ZD

    CC --> FU[hybrid.exposure_fusion<br/>Mertens weights: contrast × saturation × well-exposedness<br/>Laplacian pyramid]
    CC2 --> FU
    FU --> HG[highlight guard<br/>keep already-bright regions]
    HG --> SEL{mode == auto<br/>and selector enabled?}
    SEL -->|yes| QS[hybrid.quality_features + selector_score<br/>exposure err, clipping, colour cast,<br/>detail loss, noise; weights fitted on val]
    SEL -->|no| OUT
    QS --> OUT[Output + intermediates + stats + latency]
    OUT --> UI[Gradio UI: before/after, gallery of 11 stages, JSON params]
```

## Offline pipeline (training, tuning, evaluation)

```mermaid
flowchart LR
    D1[(LOL-v1 our485)] --> SP[scripts/make_splits.py<br/>seed 42]
    SP --> TR[train.txt 435]
    SP --> VA[val.txt 50]
    D2[(LOL-v1 eval15)] --> TE[test_lol 15]
    D3[(LOL-v2 Real test)] --> TE2[test_lolv2 100]
    TR --> FT[src/dl/train.py<br/>zero-ref losses + L1/SSIM]
    VA --> FT
    FT --> W[weights/zerodcepp_finetuned.pth<br/>best val epoch]
    W --> TU[scripts/tune_val.py<br/>coordinate descent on val]
    VA --> TU
    TU --> CFG[configs/final.yaml FROZEN]
    VA --> FS[scripts/fit_selector.py<br/>NNLS + leave-one-out]
    CFG --> FS --> CFG
    CFG --> EV[scripts/evaluate.py<br/>methods + ablations]
    TE --> EV
    TE2 --> EV
    EV --> R[results/*/summary.csv<br/>eval_P1_TeamID.md]
    EV -. test_lock.json .-> EV
```

## Module map
| File | Responsibility |
|---|---|
| `src/io_utils.py` | Safe decoding, format normalisation, size limits, user-facing errors |
| `src/analysis.py` | Image statistics, adaptive parameter rules, post-brightening noise estimate |
| `src/classical.py` | Guided filter, illumination estimation, Retinex, CLAHE, denoising, colour correction |
| `src/dl/zerodce.py` | Zero-DCE++ network (official layer names), strict loading, guided inference |
| `src/dl/train.py` | Fine-tuning on train split, val-based checkpoint selection |
| `src/hybrid.py` | Guidance, fusion, highlight guard, quality selector, `HybridEnhancer` |
| `src/metrics.py` | PSNR, SSIM, LPIPS, NIQE, noise σ, edge correlation / F1, timing |
| `app.py` | Gradio interface |
