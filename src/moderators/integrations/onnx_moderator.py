from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .base import BaseModerator, PredictionResult


class OnnxModerator(BaseModerator):
    """
    Minimal ONNX Runtime image classification moderator.
    Expects an ONNX model that takes a batched image tensor and outputs class scores.
    Config keys:
      - model_path: local path or hub-downloaded path to onnx model
      - input_size: [H, W]
      - mean: [R, G, B]
      - std: [R, G, B]
      - labels: [label0, label1, ...]
      - input_name: optional explicit input tensor name
      - output_name: optional explicit output tensor name
    """

    def load_model(self) -> None:
        model_path = self.config.get("model_path") or self.model_id
        # If model_path looks like a repo id, download the filename specified (default model.onnx)
        if "/" in str(model_path) and not str(model_path).endswith(".onnx"):
            from huggingface_hub import hf_hub_download

            hub_file = self.config.get("hub_weights", "model.onnx")
            model_path = hf_hub_download(repo_id=str(model_path), filename=str(hub_file))
        providers = self.backend_config.get("providers")
        self.session = self._create_onnx_session(model_path, providers=providers)
        self.input_name, self.output_name = self._resolve_io_names()

    def _resolve_io_names(self) -> Tuple[str, str]:
        input_name = self.config.get("input_name")
        output_name = self.config.get("output_name")
        if input_name and output_name:
            return input_name, output_name

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        in_name = input_name or inputs[0].name
        out_name = output_name or outputs[0].name
        return in_name, out_name

    def _preprocess(self, inputs: Any) -> np.ndarray:
        if isinstance(inputs, (str, bytes)):
            image = Image.open(inputs).convert("RGB")
        elif isinstance(inputs, Image.Image):
            image = inputs.convert("RGB")
        else:
            raise ValueError("Unsupported input type for OnnxModerator")

        size = self.config.get("input_size", [224, 224])
        image = image.resize((size[1], size[0]))
        array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array(self.config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(self.config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
        array = np.expand_dims(array, 0)  # NCHW
        return array

    def _predict(self, processed_inputs: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = self.session.run([self.output_name], {self.input_name: processed_inputs})
        return {self.output_name: outputs[0]}

    def _postprocess(self, model_outputs: Dict[str, np.ndarray]) -> List[PredictionResult]:
        scores = model_outputs[self.output_name]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]

        labels = self.config.get("labels", [])
        classifications: Dict[str, float] = {}
        if len(labels) == len(scores):
            for label, score in zip(labels, scores.tolist()):
                classifications[str(label)] = float(score)
        else:
            # Fallback: map by index
            for i, s in enumerate(scores.tolist()):
                classifications[str(i)] = float(s)

        return [
            PredictionResult(
                source_path=str(self.config.get("source", "")),
                classifications=classifications,
                raw_output=model_outputs,
            )
        ]

