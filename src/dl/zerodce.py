"""Zero-DCE++ (Li, Guo, Loy — TPAMI 2021), ported from the official repo
https://github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++/model.py).

Layer names match the official checkpoint so `load_state_dict(strict=True)` works.
The network predicts a 3-channel curve parameter map A in [-1, 1]; the image is
enhanced by applying the light-enhancement curve LE(x) = x + A * (x^2 - x)
iteratively (8 times). Expected parameter count: 10,561.

Extension used by the hybrid: `strength` (scalar or HxW map) scales A per pixel,
so the classical illumination map can protect bright regions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_PARAMS = 10561
OFFICIAL_WEIGHTS_URL = ("https://github.com/Li-Chongyi/Zero-DCE_extension/raw/main/"
                        "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth")


class CSDN_Tem(nn.Module):
    """Depthwise-separable conv (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    def __init__(self, scale_factor: int = 1, iterations: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        self.iterations = iterations
        nf = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = CSDN_Tem(3, nf)
        self.e_conv2 = CSDN_Tem(nf, nf)
        self.e_conv3 = CSDN_Tem(nf, nf)
        self.e_conv4 = CSDN_Tem(nf, nf)
        self.e_conv5 = CSDN_Tem(nf * 2, nf)
        self.e_conv6 = CSDN_Tem(nf * 2, nf)
        self.e_conv7 = CSDN_Tem(nf * 2, 3)

    def curve_map(self, x: torch.Tensor) -> torch.Tensor:
        """Predict curve parameters A (N,3,H,W) in [-1,1]."""
        h, w = x.shape[-2:]
        xd = x if self.scale_factor == 1 else F.interpolate(
            x, scale_factor=1 / self.scale_factor, mode="bilinear", align_corners=False)
        x1 = self.relu(self.e_conv1(xd))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        A = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if A.shape[-2:] != (h, w):
            A = F.interpolate(A, size=(h, w), mode="bilinear", align_corners=True)
        return A

    @staticmethod
    def apply_curve(x: torch.Tensor, A: torch.Tensor, iterations: int = 8) -> torch.Tensor:
        for _ in range(iterations):
            x = x + A * (x * x - x)
        return x

    def forward(self, x):
        A = self.curve_map(x)
        return self.apply_curve(x, A, self.iterations), A


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(weights: str | Path, scale_factor: int = 1, device: str = "cpu") -> ZeroDCEpp:
    """Load weights strictly (raises on any missing/unexpected key)."""
    model = ZeroDCEpp(scale_factor=scale_factor)
    sd = torch.load(str(weights), map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}  # strip DataParallel prefix
    model.load_state_dict(sd, strict=True)
    assert count_params(model) == EXPECTED_PARAMS, f"param count {count_params(model)} != {EXPECTED_PARAMS}"
    return model.to(device).eval()


def _auto_scale(h: int, w: int) -> int:
    """Downscale factor for curve estimation on large images (curve map is smooth)."""
    side = max(h, w)
    return 1 if side <= 600 else (2 if side <= 1400 else 4)


@torch.no_grad()
def enhance_dl(model: ZeroDCEpp, img: np.ndarray, strength=1.0, iterations: int = 8,
               return_curve: bool = False):
    """Enhance float RGB [0,1] numpy image. `strength` is a scalar or HxW array."""
    h, w = img.shape[:2]
    model.scale_factor = _auto_scale(h, w)
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()
    A = model.curve_map(x)
    s = torch.as_tensor(np.asarray(strength, dtype=np.float32))
    if s.ndim == 2:
        s = s[None, None]
    out = model.apply_curve(x, A * s, iterations).clamp(0, 1)
    out_np = out[0].permute(1, 2, 0).numpy().astype(np.float32)
    if return_curve:
        return out_np, A[0].permute(1, 2, 0).numpy()
    return out_np
