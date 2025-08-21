from __future__ import annotations

from typing import Any, List

from transformers import pipeline

from .base import BaseModerator, PredictionResult


class TransformersModerator(BaseModerator):
    def load_model(self) -> None:
        task = self.config.get("task")
        if not task:
            raise ValueError("TransformersModerator requires 'task' in config")
        self.pipeline = pipeline(task, model=self.model_id)

    def _preprocess(self, inputs: Any) -> Any:
        return inputs

    def _predict(self, processed_inputs: Any) -> Any:
        return self.pipeline(processed_inputs)

    def _postprocess(self, model_outputs: Any) -> List[PredictionResult]:
        # Minimal MVP: handle image/text classification-like outputs
        results: List[PredictionResult] = []
        # transformers pipelines may return dict or list-of-dicts depending on input
        outputs = model_outputs
        if isinstance(outputs, dict):
            outputs = [outputs]

        for out in outputs:
            classifications = {}
            # common keys: label/score
            label = out.get("label")
            score = out.get("score")
            if label is not None and score is not None:
                classifications[label] = float(score)

            results.append(
                PredictionResult(
                    source_path=str(self.config.get("source", "")),
                    classifications=classifications,
                    raw_output=out,
                )
            )
        return results
