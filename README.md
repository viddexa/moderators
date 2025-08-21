# moderators

An open-source Python toolkit for multi-domain content moderation.

## Quickstart

```python
from moderators import Moderator

# Use a public placeholder model until you publish your own
model = Moderator.from_pretrained("Falconsai/nsfw_image_detection")
results = model.predict("path/to/image.jpg")
for res in results:
    print(res)
```

### Save and push to Hub (MVP)

```python
inst = Moderator.from_pretrained("your-org/your-model")
inst.save_pretrained("./out")
inst.push_to_hub("your-org/your-model-copy")
```

## Automatic Optional Dependency Installation

- The library detects the selected model architecture and attempts to install required optional dependencies automatically on first use.
- If auto-install fails or is disabled, you will get a clear error with manual install commands.

Disable auto-install:
- Environment variable: `MODERATORS_DISABLE_AUTO_INSTALL=1`
- CLI: `moderators settings autoinstall=false`

Manual install (fallback):
- Transformers: `pip install 'moderators[transformers]'` or `pip install torch>=2.0.0 transformers>=4.30.0 accelerate`
- Ultralytics: `pip install 'moderators[ultralytics]'` or `pip install ultralytics>=8.0.0`
- ONNX: `pip install 'moderators[onnx]'` or `pip install onnx onnxruntime`

## ONNX Runtime Backend

Add an ONNX model to the Hub with a config:

```json
{
  "architecture": "OnnxModerator",
  "task": "image-classification",
  "model_path": "model.onnx",
  "input_size": [224, 224],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "labels": ["nsfw", "sfw"]
}
```

Run:

```python
from moderators import Moderator
model = Moderator.from_pretrained("your-org/onnx-model-repo")
res = model.predict("image.jpg")
```

## Ultralytics Backend

```json
{
  "architecture": "UltralyticsModerator",
  "task": "object-detection",
  "model_path": "your-org/your-yolo-repo",
  "hub_weights": "yolov8n.pt"
}
```

## Benchmarks

Simple timing utility:

```bash
python examples/benchmarks.py your-org/your-model image.jpg --warmup 3 --repeats 20
```
