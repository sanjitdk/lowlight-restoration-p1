"""Zero-DCE++ inference without PyTorch (OpenCV DNN + ONNX).

The deployed app uses this: PyTorch alone takes ~150 MB of RAM, which does not fit
a 512 MB free host. The curve-estimation network is exported once to ONNX
(`python -m src.dl.zerodce_onnx export`, needs torch + onnx). The resizing and the
curve LE(x) = x + A(x^2 - x) are done here in NumPy, matching `zerodce.py`.
`tests/test_onnx_backend.py` checks that the output matches the PyTorch version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _bilinear_axis(n_in: int, n_out: int, align_corners: bool, scale: float | None):
    """Source indices/weights along one axis, matching torch.nn.functional.interpolate."""
    d = np.arange(n_out, dtype=np.float64)
    if align_corners:
        src = d * ((n_in - 1) / (n_out - 1)) if n_out > 1 else np.zeros_like(d)
    else:
        ratio = scale if scale is not None else n_in / n_out
        src = np.maximum((d + 0.5) * ratio - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (src - i0).astype(np.float32)


def _resize(x: np.ndarray, h: int, w: int, align_corners: bool, scale: float | None = None):
    """Bilinear resize of an HxWxC array, same as torch interpolate (no antialias)."""
    y0, y1, ly = _bilinear_axis(x.shape[0], h, align_corners, scale)
    x0, x1, lx = _bilinear_axis(x.shape[1], w, align_corners, scale)
    ly, lx = ly[:, None, None], lx[None, :, None]
    rows = x[y0] * (1 - ly) + x[y1] * ly
    return (rows[:, x0] * (1 - lx) + rows[:, x1] * lx).astype(np.float32)


def _auto_scale(h: int, w: int) -> int:
    """Same rule as zerodce._auto_scale."""
    side = max(h, w)
    return 1 if side <= 600 else (2 if side <= 1400 else 4)


class ZeroDCEppONNX:
    def __init__(self, onnx_path: str | Path):
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def curve_map(self, img: np.ndarray, scale_factor: int = 1) -> np.ndarray:
        h, w = img.shape[:2]
        x = img.astype(np.float32)
        if scale_factor != 1:
            x = _resize(x, h // scale_factor, w // scale_factor, False, float(scale_factor))
        self.net.setInput(np.ascontiguousarray(x.transpose(2, 0, 1))[None])
        A = self.net.forward()[0].transpose(1, 2, 0)
        if A.shape[:2] != (h, w):
            A = _resize(np.ascontiguousarray(A), h, w, True)
        return A


def load_model_onnx(path: str | Path) -> ZeroDCEppONNX:
    return ZeroDCEppONNX(path)


def enhance_dl_onnx(model: ZeroDCEppONNX, img: np.ndarray, strength=1.0, iterations: int = 8,
                    return_curve: bool = False):
    """Same contract as zerodce.enhance_dl."""
    h, w = img.shape[:2]
    A = model.curve_map(img, _auto_scale(h, w))
    s = np.asarray(strength, dtype=np.float32)
    As = A * (s[..., None] if s.ndim == 2 else s)
    x = img.astype(np.float32)
    for _ in range(iterations):
        x = x + As * (x * x - x)
    out = np.clip(x, 0, 1).astype(np.float32)
    return (out, A) if return_curve else out


def export(pth: str | Path, out: str | Path | None = None) -> Path:
    """Export the curve network of a PyTorch checkpoint to ONNX (needs torch + onnx)."""
    import torch

    from .zerodce import load_model

    model = load_model(pth)
    model.scale_factor = 1

    class _Curve(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.curve_map(x)

    out = Path(out) if out else Path(pth).with_suffix(".onnx")
    torch.onnx.export(_Curve(model).eval(), torch.rand(1, 3, 400, 600), str(out),
                      input_names=["x"], output_names=["A"], opset_version=13,
                      dynamic_axes={"x": {2: "h", 3: "w"}, "A": {2: "h", 3: "w"}})
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "export":
    for p in sys.argv[2:] or ["weights/zerodcepp_finetuned.pth"]:
        print("wrote", export(p))
