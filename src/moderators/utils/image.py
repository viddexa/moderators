from __future__ import annotations

from pathlib import Path
from typing import Any


def preprocess_image_input(inputs: Any, min_side: int = 0) -> Any:
    """
    Preprocesses image inputs from Path, PIL Image, or a list/tuple (batch).

    - Opens the image if the input is a path.
    - Converts the image to RGB.
    - (Optional) If min_side > 0, proportionally scales up small images.
    Returns the input as is if Pillow is not installed or if the input type is unrecognized.
    """
    try:
        from PIL import Image
    except ImportError:
        return inputs

    def _process(obj: Any):
        # Path or string
        if isinstance(obj, (str, Path)):
            try:
                img = Image.open(str(obj))
            except (FileNotFoundError, OSError):
                return obj
        # PIL Image-like object
        elif hasattr(obj, "mode") and hasattr(obj, "convert"):
            img = obj
        else:
            return obj

        # Ensure the image is in RGB mode
        if img.mode != "RGB":
            try:
                img = img.convert("RGB")
            except Exception:
                return obj  # Return the original if conversion fails

        # Optional resizing
        if min_side and min_side > 0:
            try:
                w, h = img.size
                if w < min_side or h < min_side:
                    scale = max(min_side / w, min_side / h)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
                    img = img.resize((new_w, new_h), resample)
            except Exception:
                pass

        return img

    if isinstance(inputs, (list, tuple)):
        return [_process(x) for x in inputs]
    return _process(inputs)
