from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from moderators.utils import (
    auto_install,
    ensure_dl_framework,
    ensure_pillow_for_task,
    ensure_transformers,
    preprocess_image_input,
)

from .base import BaseModerator, PredictionResult


class TransformersModerator(BaseModerator):
    """Moderator implementation using HuggingFace Transformers."""

    def load_model(self) -> None:
        """
        Build a transformers pipeline deterministically:
        - Validate task.
        - Ensure deps (transformers, DL framework, Pillow for image tasks).
        - Try AutoProcessor (if local `preprocessor_config.json` exists).
        - Fallback order: AutoImageProcessor -> AutoFeatureExtractor.
        - Also try AutoTokenizer when relevant.
        - Pass only successfully loaded components (processor / image_processor / feature_extractor / tokenizer).
        """
        task = self.config.get("task")
        if not task:
            raise ValueError("TransformersModerator requires 'task' in config.json")

        # Ensure transformers is available
        try:
            _transformers = ensure_transformers(auto_install)
        except Exception as e:
            raise ImportError(
                "TransformersModerator requires the 'transformers' package. "
                "Install with: uv pip install -e '.[transformers]' or: uv pip install transformers"
            ) from e

        pipeline = getattr(_transformers, "pipeline")

        # Ensure a DL framework (pt/tf/flax)
        framework = ensure_dl_framework(auto_install)

        # Ensure Pillow for image tasks
        ensure_pillow_for_task(task, auto_install)

        model_id = self.model_id

        processor = None
        image_processor = None
        feature_extractor = None
        tokenizer = None

        # Check local preprocessor_config.json
        try:
            p = Path(model_id)
            has_local_preprocessor = p.is_dir() and (p / "preprocessor_config.json").exists()
        except Exception:
            has_local_preprocessor = False

        # AutoProcessor (generic unified) first if local config hints it exists
        if has_local_preprocessor:
            try:
                AutoProcessor = getattr(_transformers, "AutoProcessor", None)
                if AutoProcessor:
                    processor = AutoProcessor.from_pretrained(model_id)
            except Exception:
                processor = None  # soft fallback

        # If no unified processor, attempt vision processors explicitly
        if processor is None:
            # Newer API
            try:
                AutoImageProcessor = getattr(_transformers, "AutoImageProcessor", None)
                if AutoImageProcessor:
                    image_processor = AutoImageProcessor.from_pretrained(model_id)
            except Exception:
                image_processor = None
            # Legacy feature extractor
            if image_processor is None:
                try:
                    AutoFeatureExtractor = getattr(_transformers, "AutoFeatureExtractor", None)
                    if AutoFeatureExtractor:
                        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
                except Exception:
                    feature_extractor = None

        # Tokenizer (independent of vision processors)
        try:
            AutoTokenizer = getattr(_transformers, "AutoTokenizer", None)
            if AutoTokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tokenizer = None

        pipe_kwargs = {}
        if processor is not None:
            pipe_kwargs["processor"] = processor
        else:
            if image_processor is not None:
                pipe_kwargs["image_processor"] = image_processor
            elif feature_extractor is not None:
                pipe_kwargs["feature_extractor"] = feature_extractor
            if tokenizer is not None:
                pipe_kwargs["tokenizer"] = tokenizer

        self._pipe = pipeline(
            task,
            model=model_id,
            framework=framework,
            **pipe_kwargs,
        )

    def _preprocess(self, inputs: Any) -> Any:
        task = str(self.config.get("task", "")).lower()
        if "image" in task:
            return preprocess_image_input(inputs, min_side=2)
        return inputs

    def _predict(self, processed_inputs: Any) -> Any:
        return self._pipe(processed_inputs)

    def _postprocess(self, model_outputs: Any) -> list[PredictionResult]:
        # Pipelines typically return dict or list[dict]
        outputs = model_outputs
        if isinstance(outputs, dict):
            outputs = [outputs]

        results: list[PredictionResult] = []
        for out in outputs:
            classifications: dict[str, float] = {}
            label = out.get("label")
            score = out.get("score")
            if label is not None and score is not None:
                classifications[str(label)] = float(score)

            results.append(
                PredictionResult(
                    source_path=str(self.config.get("source", "")),
                    classifications=classifications,
                    detections=[],
                    raw_output=out,
                )
            )
        return results

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> str:
        """Saves model + tokenizer + (processor / image_processor / feature_extractor) and refreshes/creates a
        config.json with required moderator metadata.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        pipe = getattr(self, "_pipe", None)

        model = getattr(pipe, "model", None) if pipe is not None else None
        tokenizer = getattr(pipe, "tokenizer", None) if pipe is not None else None

        # Unified vision processor resolution order:
        # processor (generic) -> image_processor (newer HF) -> feature_extractor (legacy)
        vision_processor = None
        if pipe is not None:
            vision_processor = (
                getattr(pipe, "processor", None)
                or getattr(pipe, "image_processor", None)
                or getattr(pipe, "feature_extractor", None)
            )

        if model and hasattr(model, "save_pretrained"):
            model.save_pretrained(out_dir)
        if tokenizer and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_dir)
        if vision_processor and hasattr(vision_processor, "save_pretrained"):
            vision_processor.save_pretrained(out_dir)

        cfg_path = out_dir / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        except Exception:
            cfg = {}

        cfg["architecture"] = "TransformersModerator"
        if self.config.get("task"):
            cfg["task"] = self.config["task"]

        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_dir)
