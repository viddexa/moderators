# Examples

This folder contains small, practical examples to get you productive with Moderators in minutes.

## Install
Pick one and stick to it:

```bash
pip install moderators
```

Or with uv:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv add moderators
```

## Quickstart (Image, Python)
```python
from moderators.auto_model import AutoModerator

# NSFW image classification model from the Hub
moderator = AutoModerator.from_pretrained("viddexa/nsfw-mini")

# Run on a local image path
result = moderator("/path/to/image.jpg")
print(result)
```

## Quickstart (Image, CLI)
```bash
moderators viddexa/nsfw-mini /path/to/image.jpg
```

Tip: Add `--local-files-only` to force offline usage if the files are already cached.

## Batch processing (folder of images)
Process a directory of images and print the top result per file.

```python
from pathlib import Path
from moderators.auto_model import AutoModerator

images_dir = Path("/path/to/images")
model = AutoModerator.from_pretrained("viddexa/nsfw-mini")

for img_path in images_dir.glob("**/*"):
    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif"}:
        out = model(str(img_path))
        print(img_path.name, "->", out)
```

## Text classification example
You can also load a text classifier.

Python:
```python
from moderators.auto_model import AutoModerator

text_model = AutoModerator.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(text_model("I love this!"))
```

CLI:
```bash
moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
```

## Benchmark script
`examples/benchmarks.py` measures per-inference latency with warmup and repeats.

Usage:
```bash
python examples/benchmarks.py <model_id> <image_path> [--warmup N] [--repeats N] [--backend onnx]
```

Examples:
```bash
# Default backend (auto-detected)
python examples/benchmarks.py viddexa/nsfw-mini /path/to/image.jpg --warmup 3 --repeats 20

```

Expected output (sample):
```
Model: viddexa/nsfw-mini
Backend: auto
Runs: 20, avg: 12.34 ms, p50: 11.80 ms, p90: 14.10 ms
```

## Tips
- If you see missing packages (Pillow/torch/transformers), install the package: `pip install moderators`. Prefer manual control? Use extras: `pip install "moderators[transformers]"`.
- For private Hub repos, ensure you’re logged in with `huggingface-cli login` or have valid tokens set in your environment.
- GPU usage depends on your underlying framework install (e.g., CUDA-enabled PyTorch).

## Troubleshooting
- ImportError: No module named 'PIL' (or missing torch/transformers)
  - Install the package (`pip install moderators`) or manage dependencies manually with extras: `pip install "moderators[transformers]"`.
- OSError: Couldn’t find config.json / model files
  - Verify your model id or local folder path; it must contain a `config.json`.
- HTTP errors from the Hub
  - Check connectivity and permissions, or run with `--local-files-only` if already cached.
