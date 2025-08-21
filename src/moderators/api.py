import importlib
import json
from typing import Any, Dict, Optional
# Optional dependency management
import os
import sys
import subprocess
from importlib.util import find_spec
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub import ModelHubMixin

# Architecture -> optional dependencies (module names and pip packages)
_EXTRA_DEPS: Dict[str, Dict[str, Any]] = {
    "TransformersModerator": {
        "modules": ["torch", "transformers", "accelerate"],
        "pip": ["torch>=2.0.0", "transformers>=4.30.0", "accelerate"],
        "extra": "transformers",
    },
    "UltralyticsModerator": {
        "modules": ["ultralytics"],
        "pip": ["ultralytics>=8.0.0"],
        "extra": "ultralytics",
    },
    "OnnxModerator": {
        "modules": ["onnx", "onnxruntime"],
        "pip": ["onnx", "onnxruntime"],
        "extra": "onnx",
    },
}

def _read_settings() -> Dict[str, Any]:
    try:
        p = Path.home() / ".moderators" / "settings.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text())
    except Exception:
        return {}

def _should_auto_install(kwargs: Dict[str, Any]) -> bool:
    # Disable via environment variable
    if os.getenv("MODERATORS_DISABLE_AUTO_INSTALL", "").lower() in {"1", "true", "yes"}:
        return False
    # Per-call override
    if "auto_install" in kwargs:
        return bool(kwargs["auto_install"])
    # Read from settings (default True)
    settings = _read_settings()
    return bool(settings.get("autoinstall", True))

def _ensure_optional_deps(architecture_name: str, *, auto_install: bool) -> None:
    info = _EXTRA_DEPS.get(architecture_name)
    if not info:
        return
    missing = [m for m in info["modules"] if find_spec(m) is None]
    if not missing:
        return
    if not auto_install:
        extra = info.get("extra")
        raise ImportError(
            "Missing optional dependencies: "
            + ", ".join(missing)
            + ". Auto-install is disabled. Manual install: "
            + (f"pip install 'moderators[{extra}]'" if extra else "")
            + (f" or: pip install " + " ".join(info["pip"]))
        )
    # Try auto-install
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *info["pip"]])
    except Exception as e:
        extra = info.get("extra")
        raise ImportError(
            "Automatic installation of optional dependencies failed. "
            "Please install manually: "
            + (f"pip install 'moderators[{extra}]'" if extra else "")
            + (f" or: pip install " + " ".join(info["pip"]))
        ) from e

class Moderator(ModelHubMixin):
    """
    User-facing factory class to load the appropriate integration based on a
    model's configuration file hosted on the Hugging Face Hub.
    """

    _ARCH_TO_CLASS_PATH: Dict[str, str] = {
        # architecture name in config -> fully-qualified class path
        "TransformersModerator": "moderators.integrations.transformers_moderator.TransformersModerator",
        "UltralyticsModerator": "moderators.integrations.ultralytics_moderator.UltralyticsModerator",
        "OnnxModerator": "moderators.integrations.onnx_moderator.OnnxModerator",
    }

    # Known model presets for repos that do not ship a moderators-specific config
    _MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
        # Falconsai NSFW image classification model (Transformers ViT)
        "Falconsai/nsfw_image_detection": {
            "architecture": "TransformersModerator",
            "task": "image-classification",
        },
        # New: suko/nsfw runs with an ONNX model expecting NHWC
        "suko/nsfw": {
            "architecture": "OnnxModerator",
            "task": "image-classification",
            "input_layout": "NHWC",
            "input_size": [224, 224],
        },
    }

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        config_filename: str = "config.json",
        **kwargs: Any,
    ) -> Any:
        """
        Downloads the config from the Hub (or uses a preset/metadata inference), instantiates the
        correct integration, calls its load_model(), and returns the ready instance.
        """
        # Try to fetch config.json from the Hub. If missing, try metadata-based inference, then presets.
        config: Dict[str, Any] = {}
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename=config_filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            # Try to infer architecture/task from model metadata
            inferred = None
            try:
                from huggingface_hub import HfApi  # local import to avoid hard dep at import time
                info = HfApi().model_info(model_id, revision=revision, token=token)
                pipeline_tag = getattr(info, "pipeline_tag", None)
                library_name = (getattr(info, "library_name", None) or "") or ""
                tags = set(getattr(info, "tags", []) or [])
                # Heuristics
                if "transformers" in str(library_name).lower() and pipeline_tag:
                    inferred = {"architecture": "TransformersModerator", "task": pipeline_tag}
                elif "ultralytics" in str(library_name).lower() or "ultralytics" in {t.lower() for t in tags}:
                    inferred = {"architecture": "UltralyticsModerator", "task": "object-detection"}
                elif "onnx" in {t.lower() for t in tags} or "onnx" in str(library_name).lower():
                    # Default to NHWC for many TF-exported ONNX CV models
                    inferred = {"architecture": "OnnxModerator", "task": "image-classification", "input_layout": "NHWC"}
            except Exception:
                # Ignore metadata failures and fall back to presets or error below
                pass

            if inferred is not None:
                config = inferred
            elif model_id not in cls._MODEL_PRESETS:
                # No config, no inference, no preset => re-raise
                raise

        # Apply preset if architecture is not present and a preset exists
        if not config.get("architecture") and model_id in cls._MODEL_PRESETS:
            preset = cls._MODEL_PRESETS[model_id]
            for k, v in preset.items():
                config.setdefault(k, v)

        architecture_name = config.get("architecture")
        if not architecture_name:
            raise ValueError(
                "Config is missing required 'architecture' key (e.g., 'TransformersModerator')."
            )

        # Ensure optional deps for the chosen architecture (auto-install if enabled)
        auto_install = _should_auto_install(kwargs)
        _ensure_optional_deps(architecture_name, auto_install=auto_install)

        class_path = cls._ARCH_TO_CLASS_PATH.get(architecture_name)
        if not class_path:
            raise ValueError(
                f"Unsupported architecture '{architecture_name}'. "
                f"Known: {sorted(cls._ARCH_TO_CLASS_PATH.keys())}"
            )

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        integration_cls = getattr(module, class_name)

        predictor = integration_cls(config=config, model_id=model_id, **kwargs)
        predictor.load_model()
        return predictor
