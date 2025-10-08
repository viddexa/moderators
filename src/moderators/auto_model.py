# auto_model.py
"""
AutoModerator Factory.

This module contains the AutoModerator class, a factory that automatically selects and initializes the correct moderator
class based on a model identifier from the Hugging Face Hub.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

# We import BaseModerator only for type hinting.
# This avoids potential circular dependency issues.
from .integrations.base import BaseModerator


def _load_config(identifier: str, *, local_files_only: bool = False) -> dict[str, Any]:
    """Loads a config.json file from a local path or the Hugging Face Hub."""
    p = Path(identifier)
    if p.is_dir():
        cfg_path = p / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in local folder: {cfg_path}")
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    # We lazy-import huggingface_hub only when needed to reduce initial import time.
    from huggingface_hub import hf_hub_download

    cfg_fp = hf_hub_download(
        repo_id=identifier,
        filename="config.json",
        repo_type="model",
        local_files_only=local_files_only,
    )
    return json.loads(Path(cfg_fp).read_text(encoding="utf-8"))


def _is_transformers_cfg(cfg: dict[str, Any]) -> bool:
    """Checks if the given configuration belongs to a Transformers model."""
    # The `architectures` key alone is not enough; we confirm with other signatures.
    has_tf_sig = any(k in cfg for k in ("transformers_version", "model_type", "id2label", "label2id"))
    has_arch_list = isinstance(cfg.get("architectures"), list)
    return has_arch_list and has_tf_sig


def _infer_task(cfg: dict[str, Any]) -> str | None:
    """Attempts to infer the model's task by inspecting its architecture or problem_type."""
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    if any("classification" in a for a in archs):
        return "image-classification"

    prob = str(cfg.get("problem_type", "")).lower()
    if "classification" in prob:
        return "image-classification"

    return None


class AutoModerator:
    """
    A factory class that loads the correct moderator using the `from_pretrained` method.

    This class cannot be instantiated directly (its `__init__` method will raise an error).
    Instead, it should be used like:
    `AutoModerator.from_pretrained('username/my-model')`
    """

    def __init__(self, *args, **kwargs) -> None:
        """AutoModerator cannot be instantiated directly."""
        raise OSError(
            "AutoModerator is a factory class and cannot be instantiated directly. "
            "Please use the `AutoModerator.from_pretrained('model_id')` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config: dict | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> BaseModerator:
        """
        Loads the appropriate moderator from a model ID on the Hub or a local path.

        This method reads the `config.json` file, determines the model's architecture,
        dynamically loads the corresponding moderator class, and returns an initialized instance of it.

        Args:
            model_id (str): The Hugging Face Hub ID of the model to load or a path to a
                local directory.
            config (dict, optional): If provided, this config will be used instead of
                downloading one from the Hub.
            local_files_only (bool, optional): If True, will not attempt to download files
                and will only look at local cached files. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the moderator
                class's `__init__` method.

        Returns:
            BaseModerator: A loaded and ready-to-use moderator object.
        """
        # Step 1: Load the configuration
        cfg = dict(config or _load_config(model_id, local_files_only=local_files_only))

        # Step 2: Determine the model architecture
        architecture = cfg.get("architecture")
        if not architecture:
            # If architecture is not specified, try to infer if it's a Transformers model
            if _is_transformers_cfg(cfg):
                cfg["architecture"] = "TransformersModerator"
                # If the task is also not specified, try to infer it
                if not cfg.get("task"):
                    inferred_task = _infer_task(cfg)
                    if inferred_task:
                        cfg["task"] = inferred_task
                    else:
                        raise ValueError(
                            "Could not infer 'task' from the Transformers config. "
                            "Please specify 'task' in the model's config.json "
                            "(e.g. 'image-classification')."
                        )
            else:
                raise ValueError(
                    f"Could not determine 'architecture' from config.json for model '{model_id}'. "
                    "Please specify 'architecture' in the config file."
                )

        architecture = cfg["architecture"]

        # Step 3: Dynamically load the correct moderator class based on the architecture
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

        # Step 4: Initialize the moderator class and load its model
        instance = moderator_class(model_id=model_id, config=cfg, **kwargs)
        instance.load_model()

        return instance
