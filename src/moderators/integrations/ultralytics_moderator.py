from __future__ import annotations

from typing import Any, List

from .base import BaseModerator, PredictionResult, Box


class UltralyticsModerator(BaseModerator):
    def load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics dependency not installed. Install with `pip install moderators[ultralytics]`."
            ) from e

        # Allow overriding model path via config or fallback to model_id (Hub path)
        model_path = self.config.get("model_path", self.model_id)
        # If the path looks like a repo asset, try to download from Hub
        if not str(model_path).endswith((".pt", ".onnx")) and "/" in str(model_path):
            # assume it's a repo id and use default file
            from huggingface_hub import hf_hub_download

            hub_file = self.config.get("hub_weights", "yolov8n.pt")
            model_path = hf_hub_download(repo_id=str(model_path), filename=str(hub_file))
        self.model = YOLO(model_path)

    def _preprocess(self, inputs: Any) -> Any:
        return inputs

    def _predict(self, processed_inputs: Any) -> Any:
        return self.model(processed_inputs)

    def _postprocess(self, model_outputs: Any) -> List[PredictionResult]:
        # Convert Ultralytics results to standardized format
        results: List[PredictionResult] = []
        outputs = model_outputs
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            detections: List[Box] = []
            names = out.names if hasattr(out, "names") else {}
            for box in getattr(out, "boxes", []):
                xyxy = [float(x) for x in box.xyxy[0].tolist()] if hasattr(box, "xyxy") else []
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                label = names.get(cls_id, str(cls_id))
                detections.append(Box(xyxy=xyxy, label=label, score=conf))

            results.append(
                PredictionResult(
                    source_path=str(getattr(out, "path", "")),
                    detections=detections,
                    raw_output=out,
                )
            )
        return results

