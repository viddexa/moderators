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
    except Exception:
        return inputs

    img = None
    if isinstance(inputs, (str, Path)):
        try:
            img = Image.open(str(inputs))
        except Exception:
            return inputs
    elif hasattr(inputs, "mode") and hasattr(inputs, "convert"):
        img = inputs
    else:
        return inputs

    try:
        if getattr(img, "mode", "") != "RGB":
            img = img.convert("RGB")
    except Exception:
        return inputs

    try:
        w, h = img.size
        if w < min_side or h < min_side:
            img = img.resize((max(min_side, w), max(min_side, h)), Image.BILINEAR)
    except Exception:
        pass

    return img
