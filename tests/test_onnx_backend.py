"""The ONNX/OpenCV backend used in deployment must match the PyTorch model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.dl.zerodce import enhance_dl, load_model  # noqa: E402
from src.dl.zerodce_onnx import enhance_dl_onnx, load_model_onnx  # noqa: E402


@pytest.mark.parametrize("shape", [(400, 600), (375, 501), (1000, 750), (1500, 1001)])
def test_onnx_matches_torch(shape):
    rng = np.random.default_rng(0)
    img = (rng.random((*shape, 3)) * 0.25).astype(np.float32)
    strength = rng.random(shape).astype(np.float32)
    a = enhance_dl(load_model("weights/zerodcepp_finetuned.pth"), img, strength)
    b = enhance_dl_onnx(load_model_onnx("weights/zerodcepp_finetuned.onnx"), img, strength)
    assert np.abs(a - b).max() < 1e-4
