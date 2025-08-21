from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Box:
    xyxy: List[float]  # [xmin, ymin, xmax, ymax]
    label: str
    score: float


@dataclass
class PredictionResult:
    source_path: str
    classifications: Dict[str, float] = field(default_factory=dict)
    detections: List[Box] = field(default_factory=list)
    raw_output: Any = None


class BaseModerator(ABC):
    def __init__(self, config: Dict[str, Any], model_id: str, **kwargs: Any) -> None:
        self.config = config
        self.model_id = model_id
        self.backend: str = str(kwargs.get("backend", config.get("backend", "auto")))
        self.backend_config: Dict[str, Any] = dict(kwargs.get("backend_config", config.get("backend_config", {})))
        self.callbacks = self.get_default_callbacks()
        self.add_integration_callbacks()

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model, processors, tokenizers, etc., into memory."""

    @abstractmethod
    def _preprocess(self, inputs: Any) -> Any:
        """Prepares raw inputs for the model."""

    @abstractmethod
    def _predict(self, processed_inputs: Any) -> Any:
        """Runs the core model inference."""

    @abstractmethod
    def _postprocess(self, model_outputs: Any) -> List[PredictionResult]:
        """Converts raw outputs to standardized PredictionResult objects."""

    def predict(self, source: Any, **kwargs: Any) -> List[PredictionResult]:
        self.run_callbacks("on_predict_start")
        processed_inputs = self._preprocess(source)
        # Normalize input layout if requested by config (e.g., ONNX expecting NHWC)
        processed_inputs = self._maybe_fix_input_layout(processed_inputs)
        model_outputs = self._predict(processed_inputs)
        results = self._postprocess(model_outputs)
        # Ensure source_path is populated if integration did not set it
        try:
            source_str = getattr(source, "filename", None) or str(source)
        except Exception:
            source_str = ""
        for r in results:
            if not getattr(r, "source_path", None):
                r.source_path = source_str
        self.run_callbacks("on_predict_end")
        return results

    # Callback system (simple MVP)
    def get_default_callbacks(self) -> Dict[str, List]:
        from moderators.utils.callbacks import DEFAULT_CALLBACKS

        return {k: list(v) for k, v in DEFAULT_CALLBACKS.items()}

    def add_integration_callbacks(self) -> None:
        try:
            from moderators.integrations.hub_callbacks import HUB_CALLBACKS

            for event_name, funcs in HUB_CALLBACKS.items():
                self.callbacks.setdefault(event_name, []).extend(funcs)
        except Exception:
            # Silently ignore missing optional callbacks at MVP stage
            pass

    def run_callbacks(self, event_name: str) -> None:
        for func in self.callbacks.get(event_name, []):
            try:
                func(self)
            except Exception:
                # Do not break inference flow due to a callback failure
                pass

    # Backend helpers (MVP for ONNX Runtime)
    def _create_onnx_session(self, model_path: str, providers: List[str] | None = None):  # type: ignore[name-defined]
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is not installed. Auto-install occurs when loading an ONNX model if enabled. "
                "Disable auto-install with MODERATORS_DISABLE_AUTO_INSTALL=1 or CLI: `moderators settings autoinstall=false`. "
                "Manual install: `pip install moderators[onnx]` or `pip install onnx onnxruntime`."
            ) from e
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers or ["CPUExecutionProvider"])  # type: ignore

    # Layout normalization util
    def _maybe_fix_input_layout(self, arr: Any) -> Any:
        """Transpose between NCHW and NHWC based on config['input_layout'] if needed."""
        try:
            import numpy as np  # type: ignore
        except Exception:
            return arr
        expected = str(self.config.get("input_layout", "")).upper()
        if expected not in ("NHWC", "NCHW") or not isinstance(arr, np.ndarray):
            return arr
        # 4D tensors
        if arr.ndim == 4:
            # NCHW -> NHWC
            if expected == "NHWC" and arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                return np.transpose(arr, (0, 2, 3, 1))
            # NHWC -> NCHW
            if expected == "NCHW" and arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
                return np.transpose(arr, (0, 3, 1, 2))
        # 3D tensors (single image without batch)
        if arr.ndim == 3:
            # CHW -> HWC
            if expected == "NHWC" and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                return np.transpose(arr, (1, 2, 0))
            # HWC -> CHW
            if expected == "NCHW" and arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                return np.transpose(arr, (2, 0, 1))
        return arr

    # Persistence (MVP)
    def save_pretrained(self, save_directory: str) -> None:
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def push_to_hub(self, repo_id: str, *, token: str | None = None, commit_message: str = "Add model", private: bool | None = None) -> None:
        from huggingface_hub import HfApi
        import tempfile
        import os

        api = HfApi()
        # Create repo if needed
        api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        # Save to temp dir and upload folder
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(folder_path=tmpdir, repo_id=repo_id, token=token, commit_message=commit_message)

