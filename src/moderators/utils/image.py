from __future__ import annotations

from pathlib import Path
from typing import Any


def preprocess_image_input(inputs: Any, min_side: int = 16) -> Any:
    """
    Open path-like inputs with PIL, convert to RGB, ensure a minimal spatial size,
    and return a PIL.Image.Image. If PIL is unavailable or input is unsupported, return original input.
    """
    try:
        from PIL import Image
    except ImportError:
        return inputs

    img = None
    if isinstance(inputs, (str, Path)):
        try:
            img = Image.open(str(inputs))
        except (FileNotFoundError, OSError):
            return inputs
    elif hasattr(inputs, "mode") and hasattr(inputs, "convert"):
        img = inputs
    else:
        return inputs

    try:
        w, h = img.size
        if w < min_side or h < min_side:
            scale = max(min_side / w, min_side / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            img = img.resize((new_w, new_h), resampling)
    except (ValueError, OSError):
        pass

    return img
