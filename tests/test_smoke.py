"""Smoke tests that need no dataset. Run: python -m pytest -q tests  (or python tests/test_smoke.py)

1. A built-in sample photo is synthetically darkened + noised; the classical
   pipeline must improve PSNR vs. the dark input.
2. Bad inputs (black, white, grayscale, alpha PNG, tiny, corrupt bytes) must either
   process without crashing or raise ImageInputError with a clear message.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classical import enhance_classical  # noqa: E402
from src.io_utils import ImageInputError, load_image, to_float_rgb  # noqa: E402


def synthetic_low_light(seed=0):
    rng = np.random.default_rng(seed)
    gt = to_float_rgb(data.astronaut())
    dark = np.power(gt, 2.2) * 0.25 + rng.normal(0, 0.01, gt.shape)
    return gt, np.clip(dark, 0, 1).astype(np.float32)


def test_classical_improves_synthetic():
    gt, dark = synthetic_low_light()
    out, inter, info = enhance_classical(dark, return_intermediates=True)
    assert out.shape == gt.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    p_in, p_out = psnr(gt, dark, data_range=1), psnr(gt, out, data_range=1)
    print(f"synthetic PSNR: dark={p_in:.2f} dB  classical={p_out:.2f} dB  params={info['params']}")
    assert p_out > p_in
    for k in ("illumination", "reflectance", "retinex", "clahe", "color"):
        assert k in inter


def _png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def test_bad_inputs():
    cases = {
        "black": _png_bytes(np.zeros((64, 64, 3), np.uint8)),
        "white": _png_bytes(np.full((64, 64, 3), 255, np.uint8)),
        "gray": _png_bytes(np.full((64, 64), 20, np.uint8)),
        "alpha": _png_bytes(np.dstack([np.full((64, 64, 3), 30, np.uint8), np.full((64, 64), 128, np.uint8)])),
    }
    for name, b in cases.items():
        img = load_image(b)
        out = enhance_classical(img)
        assert np.isfinite(out).all(), name
    for name, b in {"tiny": _png_bytes(np.zeros((8, 8, 3), np.uint8)), "corrupt": b"not an image"}.items():
        try:
            load_image(b)
        except ImageInputError as e:
            print(f"{name}: rejected cleanly -> {e}")
        else:
            raise AssertionError(f"{name} should be rejected")


if __name__ == "__main__":
    test_classical_improves_synthetic()
    test_bad_inputs()
    print("smoke tests passed")
