"""Gradio app: adaptive low-light image restoration (CV Hackathon 2026, P1).

Run locally:  python app.py   → http://localhost:7860
Hugging Face Spaces runs this file directly.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.hybrid import HybridEnhancer
from src.io_utils import MAX_SIDE, ImageInputError, load_image, to_uint8

ROOT = Path(__file__).resolve().parent
CONFIG = os.environ.get("LOWLIGHT_CONFIG", str(ROOT / "configs/final.yaml"))
if not Path(CONFIG).exists():
    CONFIG = None  # falls back to DEFAULT_CFG
ENH = HybridEnhancer(CONFIG)
if not Path(ENH.cfg["weights"]).is_absolute():
    ENH = HybridEnhancer({**ENH.cfg, "weights": str(ROOT / ENH.cfg["weights"])})

MODES = {
    "Hybrid (classical + deep learning)": "hybrid",
    "Auto selector (experimental; off by default, falls back to hybrid)": "auto",
    "Classical only": "classical",
    "Deep learning only (Zero-DCE++)": "dl",
}

# (key in intermediates, caption, is single-channel map)
STAGES = [
    ("input", "1. Input", False),
    ("illumination", "2. Estimated illumination L (max-RGB + guided filter)", True),
    ("reflectance", "3. Reflectance R = I / L (Retinex)", False),
    ("retinex", "4. Retinex enhanced (adaptive gamma on L)", False),
    ("denoised", "5. Edge-preserving denoise (guided filter)", False),
    ("clahe", "6. CLAHE on Lab-L", False),
    ("classical", "7. Classical output (+ colour correction)", False),
    ("guidance_map", "8. DL guidance: curve strength from (1 - L)", True),
    ("dl_raw", "9. Zero-DCE++ output (guided)", False),
    ("fusion_weight_dl", "10. Fusion weight of DL branch", True),
    ("hybrid", "11. Hybrid fused output", False),
]


def _vis(x, is_map):
    if is_map:
        m = np.clip(x, 0, 1)
        return cv2.cvtColor(cv2.applyColorMap(to_uint8(m), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return to_uint8(x)


def _fmt(d, nd=3):
    return {k: (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def run(image_path, mode_label, progress=gr.Progress()):
    if isinstance(image_path, dict):  # older gradio file payloads
        image_path = image_path.get("path") or image_path.get("name")
    if image_path is None:
        raise gr.Error("Please upload an image (or pick an example) first.")
    try:
        progress(0.1, desc="Loading image")
        img = load_image(image_path, MAX_SIDE)
        mode = MODES.get(mode_label, "hybrid")
        progress(0.3, desc="Enhancing")
        t = time.perf_counter()
        out, inter, info = ENH.enhance(img, mode=mode, return_intermediates=True)
        ms = (time.perf_counter() - t) * 1000
        progress(0.9, desc="Preparing visualisations")
        gallery = [(_vis(inter[k], m), cap) for k, cap, m in STAGES if k in inter]
        h, w = img.shape[:2]
        sel = info.get("selector", {})
        summary = (
            f"**Mode:** {mode_label} · **Latency:** {ms:.0f} ms (CPU) · **Processed size:** {w}×{h}"
            + (f" · **Selector chose:** {sel.get('choice')}" if sel else "")
        )
        details = {
            "image statistics": _fmt(info["stats"]),
            "adaptive parameters": _fmt({k: v for k, v in info["params"].items()
                                         if not isinstance(v, str)}, 5),
        }
        if sel:
            details["selector"] = sel
        return to_uint8(img), to_uint8(out), gallery, summary, details
    except ImageInputError as e:
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except Exception:
        traceback.print_exc()
        raise gr.Error("Something went wrong while processing this image. Please try a different file.")


EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

with gr.Blocks(title="Low-Light Image Restoration", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Adaptive Low-Light Image Restoration\n"
        "Upload a dark, noisy photo. The system measures the image (brightness, noise, uneven lighting, "
        "colour cast), then combines a **classical Retinex pipeline** with a **Zero-DCE++ network** guided by "
        "the estimated illumination. Every intermediate step is shown below.\n\n"
        f"Images larger than {MAX_SIDE}px are downscaled for speed."
    )
    with gr.Row():
        with gr.Column(scale=1):
            # gr.File (not gr.Image) so that OUR loader validates the file: gr.Image decodes uploads
            # itself and surfaces raw errors (incl. server paths) for corrupt files.
            inp = gr.File(label="Low-light image (JPG / PNG / BMP / WEBP)", type="filepath",
                          file_types=["image", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"])
            mode = gr.Radio(list(MODES), value="Hybrid (classical + deep learning)", label="Method")
            btn = gr.Button("Enhance", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Examples (unseen test images)")
        with gr.Column(scale=2):
            with gr.Row():
                before = gr.Image(label="Before", interactive=False)
                after = gr.Image(label="After", interactive=False, show_download_button=True)
            summary = gr.Markdown()
    gallery = gr.Gallery(label="Intermediate steps", columns=4, height="auto", object_fit="contain")
    with gr.Accordion("Image analysis and chosen parameters", open=False):
        details = gr.JSON()
    gr.Markdown(
        "**Limitations:** extreme noise (near-black RAW-like inputs) can leave residual grain; strong coloured "
        "light sources may be partially desaturated by white balance. Outputs are enhancements, not ground "
        "truth — do not use for forensic or evidential purposes."
    )
    btn.click(run, [inp, mode], [before, after, gallery, summary, details])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), show_error=True)
