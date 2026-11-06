"""Safe image loading and validation.

All pipeline code works on float32 RGB images in [0, 1] with shape (H, W, 3).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_SIDE = 1024      # cap for the app (keeps CPU latency low)
MIN_SIDE = 16        # smaller than this is rejected
MAX_PIXELS = 60_000_000  # reject decompression bombs before resizing


class ImageInputError(ValueError):
    """Raised for inputs that cannot be processed; message is user-facing."""


def to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Convert any HxW, HxWx1/3/4 uint8/uint16/float array to float32 RGB [0,1]."""
    if img is None:
        raise ImageInputError("No image received.")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3:4].astype(np.float32)
        alpha /= 255.0 if arr.dtype == np.uint8 else (65535.0 if arr.dtype == np.uint16 else 1.0)
        arr = rgb * alpha  # composite on black: transparent areas count as dark
        arr = arr.astype(img.dtype) if np.issubdtype(img.dtype, np.integer) else arr
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ImageInputError(f"Unsupported image shape {arr.shape}.")

    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    else:
        out = arr.astype(np.float32)
        if out.max() > 1.0 + 1e-3:
            out = out / 255.0
    if not np.isfinite(out).all():
        raise ImageInputError("Image contains invalid (NaN/Inf) values.")
    return np.clip(out, 0.0, 1.0)


def validate_size(img: np.ndarray, max_side: int | None = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ImageInputError(f"Image is too large ({w}x{h}). Please upload under 60 megapixels.")
    if min(h, w) < MIN_SIDE:
        raise ImageInputError(f"Image is too small ({w}x{h}). Minimum side is {MIN_SIDE}px.")
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def load_image(src: Union[str, Path, bytes, np.ndarray, Image.Image],
               max_side: int | None = MAX_SIDE) -> np.ndarray:
    """Load from path / bytes / PIL / ndarray → float32 RGB [0,1], size-capped."""
    try:
        if isinstance(src, np.ndarray):
            arr = src
        else:
            if isinstance(src, (str, Path)):
                pil = Image.open(src)
            elif isinstance(src, bytes):
                pil = Image.open(io.BytesIO(src))
            elif isinstance(src, Image.Image):
                pil = src
            else:
                raise ImageInputError(f"Unsupported input type {type(src).__name__}.")
            if pil.width * pil.height > MAX_PIXELS:
                raise ImageInputError("Image is too large. Please upload under 60 megapixels.")
            pil.load()
            if pil.mode in ("I;16", "I;16B", "I"):
                arr = np.asarray(pil).astype(np.uint16)
            elif pil.mode in ("RGBA", "LA"):
                arr = np.asarray(pil.convert("RGBA"))
            else:
                arr = np.asarray(pil.convert("RGB"))
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageInputError("Could not read the file as an image (corrupt or unsupported format).") from e
    return validate_size(to_float_rgb(arr), max_side)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path)
