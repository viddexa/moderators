from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable


def auto_install(packages: list[str]) -> bool:
    """
    Try to auto-install required packages using 'uv' if available, otherwise fall back to 'pip'.

    Controlled by env var: MODERATORS_DISABLE_AUTO_INSTALL=1 to disable.
    """
    if str(os.environ.get("MODERATORS_DISABLE_AUTO_INSTALL", "")).lower() in ("1", "true", "yes"):
        return False

    uv = shutil.which("uv")
    cmd = [uv, "pip", "install", *packages] if uv else [sys.executable, "-m", "pip", "install", *packages]

    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False


def ensure_transformers(install_fn: Callable[[list[str]], bool]):
    """Ensure 'transformers' is importable; optionally auto-install and retry."""
    try:
        import transformers as _transformers  # noqa: F401

        return _transformers
    except Exception:
        if not install_fn(["transformers"]):
            raise
        import transformers as _transformers  # type: ignore

        return _transformers


def ensure_dl_framework(install_fn: Callable[[list[str]], bool]) -> str:
    """
    Ensure at least one DL framework is available.

    Preference: PyTorch ('pt'), TensorFlow ('tf'), JAX/Flax ('flax').
    Tries to auto-install torch first.
    """
    try:
        import torch  # noqa: F401

        return "pt"
    except Exception:
        if install_fn(["torch"]):
            try:
                import torch  # noqa: F401

                return "pt"
            except Exception:
                pass
    try:
        import tensorflow  # noqa: F401

        return "tf"
    except Exception:
        pass
    try:
        import jax  # noqa: F401

        return "flax"
    except Exception:
        pass
    raise ImportError(
        "A deep learning framework is required for transformers pipelines. Install PyTorch with: uv pip install torch"
    )


def ensure_pillow_for_task(task: str, install_fn: Callable[[list[str]], bool]) -> None:
    """For image tasks, ensure Pillow is available; auto-install if missing."""
    if "image" not in str(task).lower():
        return
    try:
        import PIL  # noqa: F401
    except Exception:
        if not install_fn(["Pillow"]):
            raise ImportError("This image task requires Pillow. Install with: uv pip install Pillow")
        import PIL  # noqa: F401
