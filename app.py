"""Gradio app: adaptive low-light image restoration (CV Hackathon 2026, P1).

Run locally:  python app.py   → open http://localhost:7860
Hugging Face Spaces runs this file directly (Docker SDK).
"""
from __future__ import annotations

import base64
import html
import os
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.analysis import luminance
from src.hybrid import HybridEnhancer
from src.io_utils import MAX_SIDE, ImageInputError, load_image, to_uint8

ROOT = Path(__file__).resolve().parent
CONFIG = os.environ.get("LOWLIGHT_CONFIG", str(ROOT / "configs/final.yaml"))
if not Path(CONFIG).exists():
    CONFIG = None  # falls back to DEFAULT_CFG
ENH = HybridEnhancer(CONFIG)
if not Path(ENH.cfg["weights"]).is_absolute():
    ENH = HybridEnhancer({**ENH.cfg, "weights": str(ROOT / ENH.cfg["weights"])})
OUT_DIR = Path(tempfile.gettempdir()) / "lowlight_outputs"
OUT_DIR.mkdir(exist_ok=True)

MODES = {
    "Hybrid — classical + deep learning": "hybrid",
    "Classical only — Retinex pipeline": "classical",
    "Deep learning only — Zero-DCE++": "dl",
    "Auto selector (experimental, off → hybrid)": "auto",
}
DEFAULT_MODE = "Hybrid — classical + deep learning"

# (key in intermediates, caption, is single-channel map)
STAGES = [
    ("input", "1 · Input", False),
    ("illumination", "2 · Illumination map L (max-RGB + guided filter)", True),
    ("reflectance", "3 · Reflectance R = I / L (Retinex)", False),
    ("retinex", "4 · Retinex brightened (adaptive γ on L)", False),
    ("denoised", "5 · Edge-preserving denoise (guided filter)", False),
    ("clahe", "6 · Local contrast (CLAHE on Lab-L)", False),
    ("classical", "7 · Classical output (+ colour correction)", False),
    ("guidance_map", "8 · DL guidance: curve strength from 1 − L", True),
    ("dl_raw", "9 · Zero-DCE++ output (illumination-guided)", False),
    ("fusion_weight_dl", "10 · Fusion weight given to DL branch", True),
    ("hybrid", "11 · Hybrid fused output", False),
]


# ---------------------------------------------------------------- helpers

def _vis(x, is_map):
    if is_map:
        return cv2.cvtColor(cv2.applyColorMap(to_uint8(np.clip(x, 0, 1)), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return to_uint8(x)


def _data_url(img: np.ndarray, quality=90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def slider_html(before: np.ndarray, after: np.ndarray, left="Before", right="After") -> str:
    return f"""
<div class="ba-wrap" style="--pos:50%">
  <img class="ba-after" src="{_data_url(after)}" alt="{right}">
  <div class="ba-before"><img src="{_data_url(before)}" alt="{left}"></div>
  <div class="ba-line"><span class="ba-knob">⟷</span></div>
  <span class="ba-tag ba-tag-l">{left}</span><span class="ba-tag ba-tag-r">{right}</span>
  <input class="ba-range" type="range" min="0" max="100" value="50" aria-label="Drag to compare">
</div>
<p class="ba-hint">Drag across the image to compare</p>"""


EMPTY_SLIDER = """
<div class="ba-empty">
  <div class="ba-empty-icon">🌙</div>
  <div><b>Upload a dark photo</b> or tap an example.</div>
  <div class="muted">It runs automatically. Typical processing time is about 1 s on CPU.</div>
</div>"""


def _card(label, value, sub=""):
    return (f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-value">{value}</div><div class="card-sub">{html.escape(sub)}</div></div>')


def stat_cards(img, out, info, ms, mode_label):
    s, p = info["stats"], info["params"]
    y_in, y_out = luminance(img).mean() * 100, luminance(out).mean() * 100
    cards = [
        _card("Processing time", f"{ms / 1000:.2f} s", f"{img.shape[1]}×{img.shape[0]} px · CPU"),
        _card("Brightness", f"{y_in:.0f}% → <b>{y_out:.0f}%</b>", "mean luminance, input → output"),
        _card("Noise (after brightening)", f"σ = {p.get('noise_after_brightening', s['noise_sigma']):.3f}",
              "denoising " + ("ON" if p.get("denoise_on") else "off") + f" · eps {p.get('denoise_eps', 0):.4f}"),
        _card("Uneven lighting", f"{s['non_uniformity']:.2f}",
              "high → bright areas protected" if s["non_uniformity"] > 0.5 else "fairly uniform"),
        _card("Colour cast", f"{s['color_cast']:.2f}", "white balance " + ("ON" if p.get("wb_on") else "off")),
        _card("Adaptive γ / CLAHE", f"{p['gamma']:.2f} / {p['clahe_clip']:.2f}",
              f"dark pixels {s['dark_ratio'] * 100:.0f}% · highlights {s['highlight_ratio'] * 100:.1f}%"),
    ]
    sel = info.get("selector")
    if sel:
        cards.append(_card("Selector", html.escape(str(sel.get("choice"))), sel.get("note", "fitted on validation")))
    return f'<div class="method-pill">{html.escape(mode_label)}</div><div class="cards">{"".join(cards)}</div>'


def _save(out) -> str:
    path = OUT_DIR / f"enhanced_{int(time.time() * 1000)}.png"
    Image.fromarray(to_uint8(out)).save(path)
    return str(path)


def _path(f):
    if isinstance(f, dict):  # older gradio file payloads
        return f.get("path") or f.get("name")
    return f


def _load(f):
    f = _path(f)
    if f is None:
        raise gr.Error("Please upload an image (or pick an example) first.")
    try:
        return load_image(f, MAX_SIDE)
    except ImageInputError as e:
        raise gr.Error(str(e))


# ---------------------------------------------------------------- callbacks

def enhance(file, mode_label, progress=gr.Progress()):
    try:
        progress(0.1, desc="Reading image")
        img = _load(file)
        mode = MODES.get(mode_label, "hybrid")
        progress(0.3, desc="Analysing and enhancing")
        t = time.perf_counter()
        out, inter, info = ENH.enhance(img, mode=mode, return_intermediates=True)
        ms = (time.perf_counter() - t) * 1000
        if mode == "classical":
            inter["classical"] = out  # show the final classical result as the last pipeline step
        progress(0.9, desc="Preparing views")
        gallery = [(_vis(inter[k], m), cap) for k, cap, m in STAGES if k in inter]
        details = {"image statistics": {k: round(float(v), 4) for k, v in info["stats"].items()},
                   "adaptive parameters": {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                                           for k, v in info["params"].items()}}
        if info.get("selector"):
            details["selector"] = info["selector"]
        return (slider_html(img, out), stat_cards(img, out, info, ms, mode_label),
                gr.update(value=_save(out), visible=True), gallery, details)
    except gr.Error:
        raise
    except Exception:
        traceback.print_exc()
        raise gr.Error("Something went wrong while processing this image. Please try a different file.")


def _example_label(p):
    stem = Path(p).stem  # e.g. lolv2_00720 -> "LOL-v2 #720", lol_eval_1 -> "LOL-v1 #1"
    num = stem.rsplit("_", 1)[-1].lstrip("0") or "0"
    return ("LOL-v2 #" if stem.startswith("lolv2") else "LOL-v1 #") + num


def pick_example(mode_label, evt: gr.SelectData, progress=gr.Progress()):
    path = EXAMPLES[evt.index]
    return (path, *enhance(path, mode_label, progress))


def compare_all(file, progress=gr.Progress()):
    try:
        img = _load(file)
        results, rows = [(to_uint8(img), "Input")], []
        for i, (name, mode) in enumerate([("Classical", "classical"), ("Deep learning (Zero-DCE++)", "dl"),
                                          ("Hybrid (ours)", "hybrid")]):
            progress(0.15 + 0.28 * i, desc=f"Running {name}")
            t = time.perf_counter()
            out = ENH.enhance(img, mode=mode)
            ms = (time.perf_counter() - t) * 1000
            results.append((to_uint8(out), f"{name} · {ms / 1000:.2f} s"))
            y = luminance(out)
            rows.append(f"| {name} | {ms / 1000:.2f} s | {y.mean() * 100:.0f}% | {(y > 0.98).mean() * 100:.1f}% | "
                        f"{np.abs(cv2.Laplacian(y, cv2.CV_32F)).mean() * 100:.2f} |")
        table = ("| Method | Time | Brightness | Clipped highlights | Detail (Laplacian energy) |\n|---|---|---|---|---|\n"
                 + "\n".join(rows) +
                 "\n\n<span class='muted'>No-reference indicators only: there is no ground truth for your photo. "
                 "Benchmark scores (PSNR / SSIM / LPIPS) are in the **About & results** tab.</span>")
        return results, table
    except gr.Error:
        raise
    except Exception:
        traceback.print_exc()
        raise gr.Error("Something went wrong while comparing methods. Please try a different file.")


# ---------------------------------------------------------------- about tab (numbers read from results/, not typed)

def results_markdown():
    try:
        import pandas as pd
    except ImportError:
        pd = None
    rows = [("classical (adaptive)", "Classical"), ("DL: Zero-DCE++ pretrained", "Zero-DCE++ (pretrained)"),
            ("DL: Zero-DCE++ fine-tuned", "Zero-DCE++ (fine-tuned)"), ("hybrid (fixed params)", "Hybrid, fixed params"),
            ("hybrid (adaptive) [ours]", "**Hybrid, adaptive (ours)**")]
    out = []
    for split, title in [("test_lol", "LOL-v1 eval15 (15 images)"), ("test_lolv2", "LOL-v2 Real test (100 images, unseen)")]:
        f = ROOT / "results" / split / "summary.csv"
        if pd is None or not f.exists():
            continue
        df = pd.read_csv(f).set_index("method")
        out.append(f"**{title}**\n\n| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |\n|---|---|---|---|---|")
        for key, name in rows:
            if key in df.index:
                r = df.loc[key]
                out.append(f"| {name} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | {r['niqe']:.2f} |")
        out.append("")
    if not out:
        return "_Benchmark tables: see `eval_P1_*.md` in the GitHub repository._"
    return "\n".join(out) + ("\nAll parameters were tuned on a 50-image validation split; test sets were evaluated once. "
                             "Classical wins on the unseen LOL-v2 set because fine-tuning over-fitted the network to "
                             "LOL-v1 exposure. Full analysis, ablation and failure cases are in the repository.")


# ---------------------------------------------------------------- UI

CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero {background: linear-gradient(120deg,#0b1026 0%,#1e1b4b 45%,#4338ca 100%); color:#fff; border-radius:18px;
       padding:26px 30px; margin-bottom:6px; position:relative; overflow:hidden;}
#hero:after {content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px; border-radius:50%;
       background: radial-gradient(circle,#fde68a 0%,#f59e0b55 40%,transparent 70%);}
#hero h1 {color:#fff; margin:0 0 6px; font-size:1.9rem; letter-spacing:-.02em;}
#hero p {color:#c7d2fe; margin:0; max-width:720px;}
.chips {margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1;}
.chip {background:#ffffff1a; border:1px solid #ffffff33; color:#e0e7ff; padding:4px 12px; border-radius:999px; font-size:.82rem;}
.step-label {font-weight:700; font-size:.95rem; margin:4px 0 -4px; color:var(--body-text-color);}
.muted {color: var(--body-text-color-subdued); font-size:.85rem;}
.ba-wrap {position:relative; width:100%; border-radius:14px; overflow:hidden; user-select:none; line-height:0;
          box-shadow:0 8px 30px #0003; background:#000;}
.ba-wrap img {width:100%; height:auto; display:block;}
.ba-before {position:absolute; inset:0; clip-path: inset(0 calc(100% - var(--pos)) 0 0);}
.ba-before img {width:100%; height:100%; object-fit:cover;}
.ba-line {position:absolute; top:0; bottom:0; left:var(--pos); width:3px; margin-left:-1.5px; background:#fff;
          box-shadow:0 0 8px #0008; pointer-events:none;}
.ba-knob {position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:38px; height:38px; border-radius:50%;
          background:#fff; color:#312e81; display:flex; align-items:center; justify-content:center; font-size:18px;
          line-height:1; box-shadow:0 2px 10px #0006;}
.ba-tag {position:absolute; top:12px; padding:4px 10px; border-radius:8px; font-size:.78rem; line-height:1.2;
         background:#000a; color:#fff; pointer-events:none;}
.ba-tag-l {left:12px;} .ba-tag-r {right:12px;}
.ba-range {position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:ew-resize; touch-action:pan-y;}
.ba-hint {text-align:center; margin:6px 0 0 !important; font-size:.8rem; color:var(--body-text-color-subdued);}
.ba-empty {border:2px dashed var(--border-color-primary); border-radius:14px; padding:70px 20px; text-align:center;
           line-height:1.7;}
.ba-empty-icon {font-size:2.6rem;}
.method-pill {display:inline-block; background:var(--color-accent-soft); color:var(--color-accent); font-weight:600;
              padding:3px 12px; border-radius:999px; font-size:.82rem; margin:4px 0 10px;}
.cards {display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px;}
.card {border:1px solid var(--border-color-primary); border-radius:12px; padding:12px 14px; background:var(--block-background-fill);}
.card-label {font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--body-text-color-subdued);}
.card-value {font-size:1.25rem; font-weight:700; margin:4px 0 2px;}
.card-sub {font-size:.76rem; color:var(--body-text-color-subdued);}
#enhance-btn {font-size:1.05rem; min-height:48px;}
footer {display:none !important;}
html, body {overflow-x:hidden;}
#examples .thumbnail-item {border-radius:10px;}
@media (max-width: 640px) {
  .gradio-container {padding-left:10px !important; padding-right:10px !important;}
  #hero {padding:18px; border-radius:14px;} #hero h1 {font-size:1.35rem;} #hero p {font-size:.9rem;}
  #hero:after {width:150px; height:150px; right:-50px; top:-50px;}
  .chip {font-size:.74rem; padding:3px 9px;}
  .cards {grid-template-columns:repeat(2,minmax(0,1fr));}
}
"""

# Drives the before/after slider (event delegation survives Gradio re-renders).
HEAD = """<script>
document.addEventListener('input', function (e) {
  if (e.target && e.target.classList && e.target.classList.contains('ba-range')) {
    e.target.parentElement.style.setProperty('--pos', e.target.value + '%');
  }
});
</script>"""

HERO = """
<div id="hero">
  <h1>🌙 Adaptive Low-Light Image Restoration</h1>
  <p>Brightens dark, noisy photos while keeping edges, texture and colour. The system first <b>measures</b> your
  image (brightness, noise, uneven lighting, colour cast), then combines a classical Retinex pipeline with a
  Zero-DCE++ neural network guided by the estimated illumination.</p>
  <div class="chips"><span class="chip">Retinex + CLAHE</span><span class="chip">Guided-filter denoising</span>
  <span class="chip">Zero-DCE++ (10K params)</span><span class="chip">Illumination-guided fusion</span>
  <span class="chip">CPU · ~1 s / image</span></div>
</div>"""

EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="amber", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="Low-Light Image Restoration", theme=THEME, css=CSS, head=HEAD) as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=260) as left_col:
            gr.HTML('<div class="step-label">① Your photo</div>')
            # gr.File (not gr.Image) so OUR loader validates uploads; gr.Image surfaces raw decode errors.
            inp = gr.File(label="Drop a JPG / PNG here, or click to browse", type="filepath", height=120,
                          file_types=["image", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"])
            gr.HTML('<div class="step-label">② Method</div>')
            mode = gr.Radio(list(MODES), value=DEFAULT_MODE, show_label=False)
            btn = gr.Button("✨ Enhance", variant="primary", elem_id="enhance-btn")
            gr.HTML(f'<div class="muted">Photos larger than {MAX_SIDE}px are downscaled. Uploads are not stored.</div>')
        with gr.Column(scale=7, min_width=260):
            with gr.Tabs():
                with gr.Tab("Result"):
                    slider = gr.HTML(EMPTY_SLIDER)
                    cards = gr.HTML()
                    download = gr.File(label="Download enhanced image (PNG)", interactive=False, visible=False)
                with gr.Tab("Compare all 3 methods"):
                    gr.Markdown("Runs **classical**, **deep learning** and **hybrid** on the same photo, side by side.")
                    cmp_btn = gr.Button("Run comparison", variant="secondary")
                    cmp_gallery = gr.Gallery(columns=2, height="auto", object_fit="contain", show_label=False,
                                             preview=False)
                    cmp_table = gr.Markdown()
                with gr.Tab("Pipeline steps"):
                    gr.Markdown("Every intermediate stage of the selected method, in order. Maps use the *inferno* "
                                "colour scale (dark = low, yellow = high).")
                    gallery = gr.Gallery(columns=3, height="auto", object_fit="contain", show_label=False)
                    with gr.Accordion("Raw measurements and chosen parameters", open=False):
                        details = gr.JSON(show_label=False)
                with gr.Tab("About & results"):
                    gr.Markdown(
                        "### How it works\n"
                        "1. **Analyse.** Measure brightness, dark and highlight pixels, noise, lighting unevenness and colour cast.\n"
                        "2. **Classical branch.** Estimate illumination (max-RGB + guided filter), apply Retinex with an "
                        "adaptive γ, denoise with a guided filter (strength from noise measured *after* brightening), apply CLAHE, "
                        "then white balance if a cast is detected.\n"
                        "3. **Deep-learning branch.** Zero-DCE++ predicts brightening curves. Their strength is scaled per pixel "
                        "by the illumination map, so lamps and windows don't blow out.\n"
                        "4. **Fuse.** Exposure fusion weights each branch per pixel by contrast, saturation and good exposure.\n\n"
                        "### Benchmark results\n" + results_markdown() + "\n\n"
                        "### Limitations and responsible use\n"
                        "- Tuned on the LOL datasets (mostly indoor). Unusual scenes may be over- or under-brightened.\n"
                        "- Very heavy noise can leave grain or look over-smoothed.\n"
                        "- Outputs are plausible enhancements, **not ground truth**. Do not use them as forensic or "
                        "evidential images.")
    outs = [slider, cards, download, gallery, details]
    if EXAMPLES:
        with left_col:  # re-enter the left column so examples sit under the controls
            ex = gr.Gallery([(p, _example_label(p)) for p in EXAMPLES], columns=4, rows=1,
                            height=110, allow_preview=False, label="Examples: unseen test images (tap one)",
                            elem_id="examples")
        ex.select(pick_example, [mode], [inp] + outs)

    btn.click(enhance, [inp, mode], outs)
    inp.upload(enhance, [inp, mode], outs)
    cmp_btn.click(compare_all, [inp], [cmp_gallery, cmp_table])

if __name__ == "__main__":
    print(f"\n>>> Open http://localhost:{os.environ.get('PORT', 7860)} in your browser "
          "(the 0.0.0.0 address printed below does not work in a browser on Windows)\n", flush=True)
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   show_error=True)
