from __future__ import annotations

from abc import ABC, abstractmethod  # added
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import ModelHubMixin


@dataclass
class Box:
    # xyxy: [x1, y1, x2, y2]
    xyxy: list[float]
    label: str
    score: float


@dataclass
class PredictionResult:
    # Context about the source (file path, URL, etc.)
    source_path: str = ""
    # Probability map for classification
    classifications: dict[str, float] = field(default_factory=dict)
    # Detection results
    detections: list[Box] = field(default_factory=list)
    # Raw output specific to models/integrations
    raw_output: Any = None


class BaseModerator(ABC, ModelHubMixin):
    def __init__(self, config: dict[str, Any], model_id: str, **kwargs: Any) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self.model_id: str = model_id
        self.config.setdefault("model_id", self.model_id)
        self.callbacks = self.get_default_callbacks()

    @abstractmethod
    def load_model(self) -> None:
        """Load model/pipeline and any processors if present."""
        pass

    # Inference flow
    def __call__(self, source: Any, **kwargs: Any):
        self.run_callbacks("on_predict_start")
        processed_inputs = self._preprocess(source)
        model_outputs = self._predict(processed_inputs)
        results = self._postprocess(model_outputs)
        self.run_callbacks("on_predict_end")
        return results

    @abstractmethod
    def _preprocess(self, inputs: Any) -> Any:
        """Convert inputs to model-ready format."""
        pass

    @abstractmethod
    def _predict(self, processed_inputs: Any) -> Any:
        """Run model inference."""
        pass

    @abstractmethod
    def _postprocess(self, model_outputs: Any) -> Any:
        """Convert outputs to PredictionResult format."""
        pass

    @abstractmethod
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> str:
        """Save model and any processors to the given directory."""
        raise NotImplementedError

    # Callback system (simple MVP)
    def get_default_callbacks(self) -> dict[str, list]:
        from moderators.utils.callbacks import DEFAULT_CALLBACKS

        return {k: list(v) for k, v in DEFAULT_CALLBACKS.items()}


    def run_callbacks(self, event_name: str) -> None:
        for func in self.callbacks.get(event_name, []):
            try:
                func(self)
            except Exception:
                # Do not break inference flow due to a callback failure
                pass