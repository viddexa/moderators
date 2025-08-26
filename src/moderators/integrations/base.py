from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from abc import ABC, abstractmethod  # added


@dataclass
class Box:
    # xyxy: [x1, y1, x2, y2]
    xyxy: List[float]
    label: str
    score: float


@dataclass
class PredictionResult:
    # Context about the source (file path, URL, etc.)
    source_path: str = ""
    # Probability map for classification
    classifications: Dict[str, float] = field(default_factory=dict)
    # Detection results
    detections: List[Box] = field(default_factory=list)
    # Raw output specific to models/integrations
    raw_output: Any = None


class BaseModerator(ABC):  # derives from ABC
    def __init__(self, config: Dict[str, Any], model_id: str, **kwargs: Any) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.model_id: str = model_id

    @abstractmethod
    def load_model(self) -> None:
        """Load model/pipeline and any processors if present."""
        pass

    # Inference flow
    def predict(self, source: Any, **kwargs: Any):
        # self.run_callbacks("on_predict_start")
        processed_inputs = self._preprocess(source)
        model_outputs = self._predict(processed_inputs)
        results = self._postprocess(model_outputs)
        # self.run_callbacks("on_predict_end")
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
