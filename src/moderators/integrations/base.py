from __future__ import annotations

from abc import ABC, abstractmethod  # added
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import ModelHubMixin


@dataclass
class Box:
    """
    Represents a bounding box detection result.

    Attributes:
        xyxy: Bounding box coordinates as [x1, y1, x2, y2]
        label: Classification label for the detected object
        score: Confidence score for the detection
    """

    # xyxy: [x1, y1, x2, y2]
    xyxy: list[float]
    label: str
    score: float


@dataclass
class PredictionResult:
    """
    Represents the output of a moderation prediction.

    Attributes:
        source_path: Context about the source (file path, URL, etc.)
        classifications: Probability map for classification tasks
        detections: List of bounding box detections
        raw_output: Raw model output specific to the integration
    """

    # Context about the source (file path, URL, etc.)
    source_path: str = ""
    # Probability map for classification
    classifications: dict[str, float] = field(default_factory=dict)
    # Detection results
    detections: list[Box] = field(default_factory=list)
    # Raw output specific to models/integrations
    raw_output: Any = None


class BaseModerator(ABC, ModelHubMixin):
    """
    Base class for all moderator implementations.

    Provides the core prediction flow and callback system for content moderation.
    """

    def __init__(self, config: dict[str, Any], model_id: str, **kwargs: Any) -> None:
        """
        Initialize the moderator.

        Args:
            config: Configuration dictionary for the moderator
            model_id: Model identifier (HuggingFace Hub ID or local path)
            **kwargs: Additional keyword arguments
        """
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
        """
        Execute the full prediction pipeline.

        Args:
            source: Input source (text, image path, PIL Image, etc.)
            **kwargs: Additional keyword arguments

        Returns:
            List of PredictionResult objects
        """
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
        """
        Get the default callback configuration.

        Returns:
            Dictionary mapping event names to lists of callback functions
        """
        from moderators.utils.callbacks import DEFAULT_CALLBACKS

        return {k: list(v) for k, v in DEFAULT_CALLBACKS.items()}

    def run_callbacks(self, event_name: str) -> None:
        """
        Execute all callbacks for a given event.

        Args:
            event_name: Name of the event to trigger callbacks for
        """
        for func in self.callbacks.get(event_name, []):
            try:
                func(self)
            except Exception:
                # Do not break inference flow due to a callback failure
                pass
