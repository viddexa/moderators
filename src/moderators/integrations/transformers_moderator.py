from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseModerator, PredictionResult
from moderators.utils import (
    auto_install,
    ensure_transformers,
    ensure_dl_framework,
    ensure_pillow_for_task,
    preprocess_image_input,
)


class TransformersModerator(BaseModerator):
    def load_model(self) -> None:
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
        pipeline = _transformers.pipeline

        # Ensure a DL framework (pt/tf/flax)
        framework = ensure_dl_framework(auto_install)

        # Ensure Pillow for image tasks
        ensure_pillow_for_task(task, auto_install)

        # Build pipeline
        self._pipe = pipeline(task, model=self.model_id, framework=framework)

    def _preprocess(self, inputs: Any) -> Any:
        task = str(self.config.get("task", "")).lower()
        if "image" in task:
            return preprocess_image_input(inputs)
        return inputs

    def _predict(self, processed_inputs: Any) -> Any:
        return self._pipe(processed_inputs)

    def _postprocess(self, model_outputs: Any) -> List[PredictionResult]:
        # Pipelines typically return dict or list[dict]
        outputs = model_outputs
        if isinstance(outputs, dict):
            outputs = [outputs]

        results: List[PredictionResult] = []
        for out in outputs:
            classifications: Dict[str, float] = {}
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
        """
        - Saves model, tokenizer, processor (if any) to `save_directory`.
        - Also saves/updates `config.json` with architecture and task info.
        Returns the `save_directory` path.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        pipe = getattr(self, "_pipe", None)
        model = getattr(pipe, "model", None) if pipe is not None else None
        tokenizer = getattr(pipe, "tokenizer", None) if pipe is not None else None
        processor = getattr(pipe, "processor", None) if pipe is not None else getattr(pipe, "feature_extractor", None)

        if model and hasattr(model, "save_pretrained"):
            model.save_pretrained(out_dir)
        if tokenizer and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_dir)
        if processor and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(out_dir)

        # config.json'u garanti altına al ve özel alanları ekle
        cfg_path = out_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        cfg["architecture"] = "TransformersModerator"
        if self.config.get("task"):
            cfg["task"] = self.config["task"]
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_dir)

