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
from src.io_utils import MAX_SIDE as _IO_MAX_SIDE, ImageInputError, load_image, to_uint8

ROOT = Path(__file__).resolve().parent
# Hosts with little RAM (e.g. Render free, 512 MB) set APP_MAX_SIDE lower to cap memory;
# the evaluation pipeline is unaffected.
MAX_SIDE = min(_IO_MAX_SIDE, int(os.environ.get("APP_MAX_SIDE", _IO_MAX_SIDE)))
try:
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", torch.get_num_threads())))
except ImportError:
    pass
cv2.setNumThreads(int(os.environ.get("CV_THREADS", cv2.getNumThreads())))
REPO_URL = "https://github.com/sanjitdk/lowlight-restoration-p1"
CONFIG = os.environ.get("LOWLIGHT_CONFIG", str(ROOT / "configs/final.yaml"))
if not Path(CONFIG).exists():
    CONFIG = None  # falls back to DEFAULT_CFG
ENH = HybridEnhancer(CONFIG)
if not Path(ENH.cfg["weights"]).is_absolute():
    ENH = HybridEnhancer({**ENH.cfg, "weights": str(ROOT / ENH.cfg["weights"])})
OUT_DIR = Path(tempfile.gettempdir()) / "lowlight_outputs"
OUT_DIR.mkdir(exist_ok=True)

# The quality selector ("auto") is disabled in the frozen config (it did not beat
# always-hybrid on validation), so it is not offered in the UI. See docs/design.md §4.6.
MODES = {
    "Hybrid (classical + Zero-DCE++)": "hybrid",
    "Classical (Retinex pipeline)": "classical",
    "Deep learning (Zero-DCE++)": "dl",
}
DEFAULT_MODE = "Hybrid (classical + Zero-DCE++)"
MODE_SHORT = {"hybrid": "Hybrid", "classical": "Classical", "dl": "Zero-DCE++"}

# (key in intermediates, caption, is single-channel map)
STAGES = [
    ("input", "1. Input", False),
    ("illumination", "2. Illumination map L", True),
    ("reflectance", "3. Reflectance R = I / L", False),
    ("retinex", "4. Retinex, adaptive gamma", False),
    ("denoised", "5. Guided-filter denoising", False),
    ("clahe", "6. CLAHE (Lab L channel)", False),
    ("classical", "7. Classical output", False),
    ("guidance_map", "8. DL curve strength (from 1 − L)", True),
    ("dl_raw", "9. Zero-DCE++ output", False),
    ("fusion_weight_dl", "10. Fusion weight, DL branch", True),
    ("hybrid", "11. Hybrid output", False),
]


# ---------------------------------------------------------------- rendering helpers

def _vis(x, is_map):
    if is_map:
        return cv2.cvtColor(cv2.applyColorMap(to_uint8(np.clip(x, 0, 1)), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    return to_uint8(x)


def _data_url(img: np.ndarray, quality=90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def slider_html(before: np.ndarray, after: np.ndarray) -> str:
    return f"""
<div class="cmp" style="--pos:50%">
  <img class="cmp-after" src="{_data_url(after)}" alt="Enhanced">
  <div class="cmp-before"><img src="{_data_url(before)}" alt="Input"></div>
  <div class="cmp-line"><span class="cmp-handle"></span></div>
  <span class="cmp-tag cmp-tag-l">Input</span><span class="cmp-tag cmp-tag-r">Enhanced</span>
  <input class="cmp-range" type="range" min="0" max="100" value="50" aria-label="Comparison position">
</div>"""


EMPTY = """
<div class="empty">
  <div class="empty-title">No image loaded</div>
  <div class="empty-text">Upload a low-light photograph or select one of the examples.</div>
</div>"""


def _rows(pairs):
    return "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in pairs)


def measurements_html(img, out, info, ms, mode):
    s, p = info["stats"], info["params"]
    y_in, y_out = luminance(img).mean(), luminance(out).mean()
    noise = p.get("noise_after_brightening", s["noise_sigma"])
    run = [
        ("Method", MODE_SHORT[mode]),
        ("Processing time", f"{ms / 1000:.2f} s"),
        ("Resolution", f"{img.shape[1]} × {img.shape[0]}"),
        ("Mean luminance", f"{y_in:.3f} → {y_out:.3f}"),
    ]
    measured = [
        ("Dark pixels (Y < 0.05)", f"{s['dark_ratio'] * 100:.1f} %"),
        ("Highlights (Y > 0.95)", f"{s['highlight_ratio'] * 100:.2f} %"),
        ("Noise σ after brightening", f"{noise:.4f}"),
        ("Lighting non-uniformity", f"{s['non_uniformity']:.3f}"),
        ("Colour cast", f"{s['color_cast']:.3f}"),
    ]
    chosen = [
        ("Gamma on illumination", f"{p['gamma']:.3f}"),
        ("CLAHE clip limit", f"{p['clahe_clip']:.2f}"),
        ("Denoising", f"on, ε = {p['denoise_eps']:.4f}" if p.get("denoise_on") else "off"),
        ("White balance", "on" if p.get("wb_on") else "off"),
    ]
    return f"""
<div class="panels">
  <section><h4>Run</h4><table class="kv">{_rows(run)}</table></section>
  <section><h4>Measured on input</h4><table class="kv">{_rows(measured)}</table></section>
  <section><h4>Adaptive parameters</h4><table class="kv">{_rows(chosen)}</table></section>
</div>"""


def _save(out) -> str:
    """Full-resolution PNG for download (fast, light compression)."""
    path = OUT_DIR / f"enhanced_{time.time_ns()}.png"
    cv2.imwrite(str(path), cv2.cvtColor(to_uint8(out), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return str(path)


def _gc(max_age_s: int = 900):
    """Delete our temporary outputs older than 15 minutes."""
    now = time.time()
    for f in OUT_DIR.glob("*"):
        try:
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
        except OSError:
            pass


def _jpg(img: np.ndarray, max_side: int | None = None, quality: int = 88) -> str:
    """Preview image as a JPEG file. Gradio would otherwise PNG-encode every gallery
    array, which takes seconds on a small shared CPU."""
    img = to_uint8(img) if img.dtype != np.uint8 else img
    if max_side and max(img.shape[:2]) > max_side:
        k = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    path = OUT_DIR / f"v_{time.time_ns()}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return str(path)


def _load(f):
    if isinstance(f, dict):  # older gradio file payloads
        f = f.get("path") or f.get("name")
    if f is None:
        raise gr.Error("Upload an image or select an example first.")
    try:
        return load_image(_open_small(f), MAX_SIDE)
    except ImageInputError as e:
        raise gr.Error(str(e))


def _open_small(path):
    """Open an upload without decoding a huge photo at full size (a 12 MP phone JPEG
    is ~150 MB as float). JPEGs decode at a reduced scale (PIL draft); other formats
    are box-reduced. The result stays >= 2x the final size; load_image does the
    final resize. Unreadable files are passed through so load_image reports them."""
    try:
        pil = Image.open(path)
        if pil.width * pil.height > 60_000_000:
            return path  # load_image rejects it with a clear message
        target = 2 * MAX_SIDE
        if max(pil.size) > target:
            pil.draft("RGB", (target, target))
            pil.load()
            factor = max(pil.size) // target
            if factor >= 2:
                pil = pil.reduce(factor)
        return pil
    except Exception:  # corrupt/unsupported: let load_image raise ImageInputError
        return path


# ---------------------------------------------------------------- callbacks

def enhance(file, mode_label, progress=gr.Progress()):
    try:
        progress(0.1, desc="Reading image")
        _gc()
        img = _load(file)
        mode = MODES.get(mode_label, "hybrid")
        progress(0.3, desc="Enhancing")
        t = time.perf_counter()
        out, inter, info = ENH.enhance(img, mode=mode, return_intermediates=True)
        ms = (time.perf_counter() - t) * 1000
        if mode == "classical":
            inter["classical"] = out  # show the final classical result as the last pipeline step
        progress(0.9, desc="Rendering")
        gallery = [(_jpg(_vis(inter[k], m), 480), cap) for k, cap, m in STAGES if k in inter]
        details = {"image statistics": {k: round(float(v), 4) for k, v in info["stats"].items()},
                   "adaptive parameters": {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                                           for k, v in info["params"].items()}}
        return (slider_html(img, out), measurements_html(img, out, info, ms, mode),
                gr.update(value=_save(out), visible=True), gallery, details)
    except gr.Error:
        raise
    except Exception:
        traceback.print_exc()
        raise gr.Error("Processing failed for this image. Try a different file.")




def pick_example(mode_label, evt: gr.SelectData, progress=gr.Progress()):
    path = EXAMPLES[evt.index]
    return (path, *enhance(path, mode_label, progress))


def compare_all(file, progress=gr.Progress()):
    try:
        _gc()
        img = _load(file)
        tiles, rows = [(_jpg(img, 768), "Input")], []
        for i, (name, mode) in enumerate([("Classical", "classical"), ("Zero-DCE++", "dl"), ("Hybrid", "hybrid")]):
            progress(0.15 + 0.28 * i, desc=f"Running {name}")
            t = time.perf_counter()
            out = ENH.enhance(img, mode=mode)
            ms = (time.perf_counter() - t) * 1000
            tiles.append((_jpg(out, 768), f"{name}, {ms / 1000:.2f} s"))
            y = luminance(out)
            rows.append(f"| {name} | {ms / 1000:.2f} | {y.mean():.3f} | {(y > 0.98).mean() * 100:.2f} | "
                        f"{np.abs(cv2.Laplacian(y, cv2.CV_32F)).mean() * 100:.2f} |")
        table = ("| Method | Time (s) | Mean luminance | Clipped (%) | Laplacian energy |\n|:--|--:|--:|--:|--:|\n"
                 + "\n".join(rows)
                 + "\n\nNo-reference indicators only; there is no ground truth for an uploaded photo. "
                   "Reference-based benchmark scores are listed under *About*.")
        return tiles, table
    except gr.Error:
        raise
    except Exception:
        traceback.print_exc()
        raise gr.Error("Comparison failed for this image. Try a different file.")


# ---------------------------------------------------------------- about tab (numbers read from results/, not typed)

def results_markdown():
    try:
        import pandas as pd
    except ImportError:
        return f"Benchmark tables: see `eval_P1_*.md` in the [repository]({REPO_URL})."
    rows = [("classical (adaptive)", "Classical"), ("DL: Zero-DCE++ pretrained", "Zero-DCE++, pretrained"),
            ("DL: Zero-DCE++ fine-tuned", "Zero-DCE++, fine-tuned"), ("hybrid (fixed params)", "Hybrid, fixed parameters"),
            ("hybrid (adaptive) [ours]", "Hybrid, adaptive")]
    out = []
    for split, title in [("test_lol", "LOL-v1 eval15, 15 images"), ("test_lolv2", "LOL-v2 Real test, 100 images (unseen)")]:
        f = ROOT / "results" / split / "summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("method")
        out.append(f"**{title}**\n\n| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |\n|:--|--:|--:|--:|--:|")
        for key, name in rows:
            if key in df.index:
                r = df.loc[key]
                out.append(f"| {name} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | {r['niqe']:.2f} |")
        out.append("")
    if not out:
        return f"Benchmark tables: see `eval_P1_*.md` in the [repository]({REPO_URL})."
    return "\n".join(out)


ABOUT = f"""
### Method
1. **Analysis.** The input is measured for brightness, dark and highlight fractions, noise, lighting non-uniformity and colour cast. All enhancement parameters are derived from these measurements.
2. **Classical branch.** Illumination is estimated with max-RGB and refined with a guided filter. Retinex decomposition is followed by an adaptive gamma on the illumination. Guided-filter denoising is set from the noise level measured after brightening. CLAHE is applied to the Lab L channel, and gray-world white balance only when a cast is detected.
3. **Deep-learning branch.** Zero-DCE++ (10,561 parameters) predicts per-pixel brightening curves. Their strength is scaled by the classical illumination map, so already-bright regions are protected.
4. **Fusion.** The two branches are blended per pixel with exposure-fusion weights (contrast, saturation, well-exposedness) in a Laplacian pyramid.

### Benchmark results
All parameters were tuned on a 50-image validation split. Each test set was evaluated once with the frozen configuration.

{results_markdown()}
On the unseen LOL-v2 set the classical branch scores highest. The fine-tuned network over-fitted the exposure level of LOL-v1, and the hybrid inherits part of that bias. The ablation study, failure cases and limitations are documented in the [repository]({REPO_URL}).

### Limitations
- Tuned on the LOL datasets, which are mostly indoor scenes. Unusual scenes can be over- or under-exposed.
- Very heavy noise can remain visible, or be smoothed at the cost of texture.
- Outputs are enhancements, not measurements of the scene, and must not be used as forensic or evidential images.
"""


# ---------------------------------------------------------------- styling

CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto !important;}
table.kv, .cmp-tag, .gradio-container table {font-variant-numeric: tabular-nums;}

.topbar {display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap;
         padding:4px 2px 14px; border-bottom:1px solid var(--border-color-primary); margin-bottom:8px;}
.topbar .title {font-size:1.25rem; font-weight:600; letter-spacing:-0.01em; color:var(--body-text-color);}
.topbar .subtitle {font-size:.86rem; color:var(--body-text-color-subdued); margin-top:2px;}
.topbar nav {display:flex; gap:18px; font-size:.86rem;}
.topbar nav a {color:var(--body-text-color-subdued); text-decoration:none;}
.topbar nav a:hover {color:var(--body-text-color); text-decoration:underline;}

.section-label {font-size:.8rem; font-weight:600; color:var(--body-text-color-subdued); margin:6px 0 -6px;}
.note {font-size:.8rem; color:var(--body-text-color-subdued); line-height:1.5;}

.cmp {position:relative; width:100%; overflow:hidden; border-radius:6px; line-height:0; user-select:none;
      background:#0a0a0a; border:1px solid var(--border-color-primary);}
.cmp img {width:100%; height:auto; display:block;}
.cmp-before {position:absolute; inset:0; clip-path: inset(0 calc(100% - var(--pos)) 0 0);}
.cmp-before img {width:100%; height:100%; object-fit:cover;}
.cmp-line {position:absolute; top:0; bottom:0; left:var(--pos); width:2px; margin-left:-1px;
           background:rgba(255,255,255,.9); pointer-events:none;}
.cmp-handle {position:absolute; top:50%; left:50%; width:26px; height:26px; margin:-13px 0 0 -13px; border-radius:50%;
             background:#fff; box-shadow:0 1px 4px rgba(0,0,0,.4);}
.cmp-handle:before, .cmp-handle:after {content:""; position:absolute; top:8px; width:2px; height:10px; background:#525252;}
.cmp-handle:before {left:9px;} .cmp-handle:after {right:9px;}
.cmp-tag {position:absolute; top:10px; padding:3px 7px; border-radius:3px; font-size:.72rem; line-height:1.3;
          background:rgba(0,0,0,.6); color:#f5f5f5; pointer-events:none;}
.cmp-tag-l {left:10px;} .cmp-tag-r {right:10px;}
.cmp-range {position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:ew-resize; touch-action:pan-y;}

.empty {border:1px dashed var(--border-color-primary); border-radius:6px; padding:96px 24px; text-align:center;}
.empty-title {font-weight:600; color:var(--body-text-color);}
.empty-text {font-size:.88rem; color:var(--body-text-color-subdued); margin-top:4px;}

.panels {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:20px; margin-top:6px;}
.panels h4 {font-size:.8rem; font-weight:600; color:var(--body-text-color-subdued); margin:0 0 6px;
            padding-bottom:6px; border-bottom:1px solid var(--border-color-primary);}
table.kv {width:100%; border-collapse:collapse; font-size:.84rem; border:none !important; margin:0 !important;}
table.kv tr, table.kv th, table.kv td {border:none !important; background:transparent !important;}
table.kv th {text-align:left; font-weight:400; color:var(--body-text-color-subdued); padding:3px 8px 3px 0 !important;}
table.kv td {text-align:right; color:var(--body-text-color); padding:3px 0 !important; white-space:nowrap;}

#run-btn {min-height:42px;}
#examples .thumbnail-item {aspect-ratio: 3 / 2; border-radius:4px;}
#examples .grid-wrap {padding:6px !important; overflow:hidden !important;}
footer {display:none !important;}
html, body {overflow-x:hidden;}
@media (max-width: 900px) { .panels {grid-template-columns:1fr; gap:14px;} }
"""

# Drives the comparison slider (event delegation survives Gradio re-renders).
HEAD = """<script>
document.addEventListener('input', function (e) {
  var t = e.target;
  if (t && t.classList && t.classList.contains('cmp-range')) {
    t.parentElement.style.setProperty('--pos', t.value + '%');
  }
});
</script>"""

TOPBAR = f"""
<div class="topbar">
  <div>
    <div class="title">Low-Light Image Restoration</div>
    <div class="subtitle">Adaptive Retinex and Zero-DCE++ hybrid. CV Hackathon 2026, Problem 1.</div>
  </div>
  <nav><a href="{REPO_URL}" target="_blank" rel="noopener">Source</a>
       <a href="{REPO_URL}/blob/main/docs/design.md" target="_blank" rel="noopener">Design document</a>
       <a href="{REPO_URL}/blob/main/eval_P1_TXX.md" target="_blank" rel="noopener">Evaluation</a></nav>
</div>"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral, neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm, text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#fafafa", body_background_fill_dark="#0a0a0a",
    block_background_fill="#ffffff", block_background_fill_dark="#141414",
    block_border_width="1px", block_shadow="none", block_shadow_dark="none",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_border_width="0px", block_label_text_color="#525252", block_label_text_color_dark="#a3a3a3",
    button_primary_background_fill="#171717", button_primary_background_fill_hover="#404040",
    button_primary_text_color="#ffffff", button_primary_border_color="#171717",
    button_primary_background_fill_dark="#f5f5f5", button_primary_background_fill_hover_dark="#d4d4d4",
    button_primary_text_color_dark="#0a0a0a", button_primary_border_color_dark="#f5f5f5",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#1f1f1f",
    checkbox_label_background_fill="#f5f5f5", checkbox_label_background_fill_dark="#1c1c1c",
    checkbox_label_background_fill_hover="#e5e5e5", checkbox_label_background_fill_hover_dark="#262626",
    checkbox_label_background_fill_selected="#171717", checkbox_label_text_color_selected="#ffffff",
    checkbox_label_background_fill_selected_dark="#f5f5f5", checkbox_label_text_color_selected_dark="#0a0a0a",
    checkbox_background_color_selected="#ffffff", checkbox_border_color_selected="#ffffff",
    checkbox_background_color_selected_dark="#0a0a0a", checkbox_border_color_selected_dark="#0a0a0a",
    checkbox_border_color_focus="#737373", checkbox_border_color_focus_dark="#737373",
    input_background_fill="#ffffff", input_background_fill_dark="#141414",
    link_text_color="#404040", link_text_color_dark="#d4d4d4",
    link_text_color_hover="#0a0a0a", link_text_color_hover_dark="#ffffff",
    link_text_color_visited="#404040", link_text_color_visited_dark="#d4d4d4",
    link_text_color_active="#0a0a0a", link_text_color_active_dark="#ffffff",
)

EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

with gr.Blocks(title="Low-Light Image Restoration", theme=THEME, css=CSS, head=HEAD,
               delete_cache=(900, 1800)) as demo:
    gr.HTML(TOPBAR)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=260) as left_col:
            gr.HTML('<div class="section-label">Input</div>')
            # gr.File (not gr.Image) so our loader validates uploads; gr.Image surfaces raw decode errors.
            inp = gr.File(label="JPG, PNG, BMP, WEBP or TIFF", type="filepath", height=110,
                          file_types=["image", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"])
            gr.HTML('<div class="section-label">Method</div>')
            mode = gr.Radio(list(MODES), value=DEFAULT_MODE, show_label=False)
            btn = gr.Button("Enhance", variant="primary", elem_id="run-btn")
            gr.HTML(f'<div class="note">Images are processed on CPU and downscaled to at most {MAX_SIDE} px '
                    'on the long side. Uploaded files are not stored.</div>')
        with gr.Column(scale=8, min_width=260):
            with gr.Tabs():
                with gr.Tab("Result"):
                    slider = gr.HTML(EMPTY)
                    panels = gr.HTML()
                    download = gr.File(label="Enhanced image (PNG)", interactive=False, visible=False)
                with gr.Tab("Comparison"):
                    gr.Markdown("Runs the classical, deep-learning and hybrid methods on the current input.")
                    cmp_btn = gr.Button("Run comparison", variant="secondary")
                    cmp_gallery = gr.Gallery(columns=2, height="auto", object_fit="contain", show_label=False,
                                             preview=False)
                    cmp_table = gr.Markdown()
                with gr.Tab("Pipeline"):
                    gr.Markdown("Intermediate outputs of the selected method. Single-channel maps use the "
                                "viridis colour scale (dark = low, yellow = high).")
                    gallery = gr.Gallery(columns=3, height="auto", object_fit="contain", show_label=False)
                    with gr.Accordion("Raw values", open=False):
                        details = gr.JSON(show_label=False)
                with gr.Tab("About"):
                    gr.Markdown(ABOUT)

    outs = [slider, panels, download, gallery, details]
    if EXAMPLES:
        with left_col:  # re-enter the left column so examples sit under the controls
            gr.HTML('<div class="section-label">Examples</div>')
            ex = gr.Gallery(EXAMPLES, columns=4, rows=1, height=132, allow_preview=False, show_label=False,
                            show_share_button=False, show_download_button=False, elem_id="examples")
        ex.select(pick_example, [mode], [inp] + outs, concurrency_id="cpu")

    btn.click(enhance, [inp, mode], outs, concurrency_id="cpu")
    inp.upload(enhance, [inp, mode], outs, concurrency_id="cpu")
    cmp_btn.click(compare_all, [inp], [cmp_gallery, cmp_table], concurrency_id="cpu")

if __name__ == "__main__":
    print(f"\n>>> Open http://localhost:{os.environ.get('PORT', 7860)} in your browser "
          "(the 0.0.0.0 address printed below does not work in a browser on Windows)\n", flush=True)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   show_error=True)
