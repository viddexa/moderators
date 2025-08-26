# python
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from huggingface_hub import ModelHubMixin  # do not import hf_hub_download here
except Exception:
    class ModelHubMixin:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls._from_pretrained(*args, **kwargs)

def _load_config(identifier: str, *, local_files_only: bool = False) -> Dict[str, Any]:
    p = Path(identifier)
    if p.exists():
        cfg_path = p / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in local folder: {cfg_path}")
        return json.loads(cfg_path.read_text())

    # Lazy import to avoid pulling heavy deps during module import
    from huggingface_hub import hf_hub_download

    cfg_fp = hf_hub_download(
        repo_id=identifier,
        filename="config.json",
        repo_type="model",
        local_files_only=local_files_only,
    )
    return json.loads(Path(cfg_fp).read_text())


def _is_transformers_cfg(cfg: Dict[str, Any]) -> bool:
    # `architectures` is not enough alone to identify a Transformers model
    has_tf_sig = any(
        k in cfg for k in ("transformers_version", "model_type", "id2label", "label2id")
    )
    has_arch_list = isinstance(cfg.get("architectures"), list)
    return has_arch_list and has_tf_sig


def _infer_task(cfg: Dict[str, Any]) -> Optional[str]:
    # get general task from architectures or problem_type
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    if any("classification" in a for a in archs):
        return "image-classification"
    prob = str(cfg.get("problem_type", "")).lower()
    if "classification" in prob:
        return "image-classification"
    return None


class AutoModerator(ModelHubMixin):
    def __init__(self, *args, **kwargs) -> None:
        raise EnvironmentError(
            "AutoModerator is a factory class and cannot be instantiated directly. "
            "Please use the `AutoModerator.from_pretrained('model_id')` method."
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        config: Optional[dict] = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ):
        cfg = dict(config or _load_config(model_id, local_files_only=local_files_only))

        architecture = cfg.get("architecture")
        if not architecture:
            if _is_transformers_cfg(cfg):
                cfg["architecture"] = "TransformersModerator"
                if not cfg.get("task"):
                    inferred = _infer_task(cfg)
                    if inferred:
                        cfg["task"] = inferred
                    else:
                        raise ValueError(
                            "Could not infer 'task' from the Transformers config. "
                            "Please specify 'task' in the model's config.json "
                            "(e.g. 'image-classification')."
                        )
            else:
                raise ValueError(
                    f"Could not determine 'architecture' from config.json for model '{model_id}'. "
                )

        architecture = cfg["architecture"]

        # For MVP, only TransformersModerator is implemented
        if architecture != "TransformersModerator":
            raise NotImplementedError(
                f"'{architecture}' is not yet supported in this version of Moderators. "
                "As of now, only 'TransformersModerator' is implemented."
            )

        module_name = architecture.replace("Moderator", "_moderator").lower()
        module_path = f"moderators.integrations.{module_name}"

        try:
            module = importlib.import_module(module_path)
            moderator_class = getattr(module, architecture)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not find or import the class '{architecture}'. "
                f"Please ensure it is defined in '{module_path}.py'. Error: {e}"
            )

        instance = moderator_class(model_id=model_id, config=cfg, **kwargs)
        instance.load_model()
        return instance

