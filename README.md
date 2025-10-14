# Moderators

[![Moderators PYPI](https://img.shields.io/pypi/v/moderators?color=blue)](https://pypi.org/project/moderators/)
[![Moderators HuggingFace Space](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/viddexa/moderators)
[![Moderators CI](https://github.com/viddexa/moderators/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/viddexa/moderators/actions/workflows/ci.yml)
[![Moderators License](https://img.shields.io/pypi/l/moderators)](https://github.com/viddexa/moderators/blob/main/LICENSE)

Run open‑source content moderation models (NSFW, toxicity, etc.) with one line — from Python or the CLI. Works with Hugging Face models or local folders. Outputs are normalized and app‑ready.

- One simple API and CLI
- Use any compatible Transformers model from the Hub or disk
- Normalized JSON output you can plug into your app
- Optional auto‑install of dependencies for a smooth first run

Note: Today we ship a Transformers-based integration for image/text classification.


## Who is this for?
Developers and researchers/academics who want to quickly evaluate or deploy moderation models without wiring different runtimes or dealing with model‑specific output formats.


## Installation
Pick one option:

Using pip (recommended):
```bash
pip install moderators
```

Using uv:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv add moderators
```

From source (cloned repo):
```bash
uv sync --extra transformers
```

Requirements:
- Python 3.10+
- For image tasks, Pillow and a DL framework (PyTorch preferred). Moderators can auto‑install these.


## Quickstart
Run a model in a few lines.

Python API:
```python
from moderators.auto_model import AutoModerator

# Load from the Hugging Face Hub (e.g., NSFW image classifier)
moderator = AutoModerator.from_pretrained("viddexa/nsfw-mini")

# Run on a local image path
result = moderator("/path/to/image.jpg")
print(result)
```

CLI:
```bash
moderators viddexa/nsfw-mini /path/to/image.jpg
```

Text example (sentiment/toxicity):
```bash
moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
```


## What do results look like?
You get a list of normalized prediction entries. In Python, they’re dataclasses; in the CLI, you get JSON.

Python shape (pretty-printed):
```text
[
  PredictionResult(
    source_path='',
    classifications={'NSFW': 0.9821},
    detections=[],
    raw_output={'label': 'NSFW', 'score': 0.9821}
  ),
  ...
]
```

JSON shape (CLI output):
```json
[
  {
    "source_path": "",
    "classifications": {"NSFW": 0.9821},
    "detections": [],
    "raw_output": {"label": "NSFW", "score": 0.9821}
  }
]
```

Tip (Python):
```python
from dataclasses import asdict
from moderators.auto_model import AutoModerator

moderator = AutoModerator.from_pretrained("viddexa/nsfw-mini")
result = moderator("/path/to/image.jpg")
json_ready = [asdict(r) for r in result]
print(json_ready)
```


## Example: Real output on a sample image
Image source:

![Example input image](https://img.freepik.com/free-photo/front-view-woman-doing-exercises_23-2148498678.jpg?t=st=1760435237~exp=1760438837~hmac=9a0a0a56f83d8fa52f424c7acdf4174dffc3e4d542e189398981a13af3f82b40&w=360)

Raw model scores:
```json
[
  { "normal": 0.9999891519546509 },
  { "nsfw": 0.000010843970812857151 }
]
```

Moderators normalized JSON shape:
```json
[
  { "source_path": "", "classifications": {"normal": 0.9999891519546509}, "detections": [], "raw_output": {"label": "normal", "score": 0.9999891519546509} },
  { "source_path": "", "classifications": {"nsfw": 0.000010843970812857151}, "detections": [], "raw_output": {"label": "nsfw", "score": 0.000010843970812857151} }
]
```


## Comparison at a glance
The table below places Moderators next to the raw Transformers `pipeline()` usage.

| Feature | Transformers.pipeline() | Moderators |
|---|---|---|
| Usage | `pipeline("task", model=...)` | `AutoModerator.from_pretrained(...)` |
| Model configuration | Manual or model-specific | Automatic via `config.json` (task inference when possible) |
| Output format | Varies by model/pipe | Standardized `PredictionResult` / JSON |
| Requirements | Manual dependency setup | Optional automatic `pip/uv` install |
| CLI | None or project-specific | Built-in `moderators` CLI (JSON to stdout) |
| Extensibility | Mostly one ecosystem | Open to new integrations (same interface) |
| Error messages | Vary by model | Consistent, task/integration-guided |
| Task detection | User-provided | Auto-inferred from config when possible |


## Pick a model
- From the Hub: pass a model id like `viddexa/nsfw-mini` or any compatible Transformers model.
- From disk: pass a local folder that contains a `config.json` next to your weights.

Moderators detects the task and integration from the config when possible, so you don’t have to specify pipelines manually.


## Command line usage
Run models from your terminal and get normalized JSON to stdout.

Usage:
```bash
moderators <model_id_or_local_dir> <input> [--local-files-only]
```

Examples:
- Text classification:
  ```bash
  moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
  ```
- Image classification (local image):
  ```bash
  moderators viddexa/nsfw-mini /path/to/image.jpg
  ```

Tips:
- `--local-files-only` forces offline usage if files are cached.
- The CLI prints a single JSON array (easy to pipe or parse).


## Examples
- Small demos and benchmarking script: `examples/README.md`, `examples/benchmarks.py`


## FAQ
- Which tasks are supported?
  - Image and text classification via Transformers (e.g., NSFW, sentiment/toxicity). More can be added over time.
- Does it need a GPU?
  - No. CPU is fine for small models. If your framework has CUDA installed, it will use it.
- How are dependencies handled?
  - If something is missing (e.g., `torch`, `transformers`, `Pillow`), Moderators can auto‑install via `uv` or `pip` unless you disable it. To disable:
    ```bash
    export MODERATORS_DISABLE_AUTO_INSTALL=1
    ```
- Can I run offline?
  - Yes. Use `--local-files-only` in the CLI or `local_files_only=True` in Python after you have the model cached.
- What does “normalized output” mean?
  - Regardless of the underlying pipeline, you always get the same result schema (classifications/detections/raw_output), so your app code stays simple.


## Roadmap
What’s planned:
- Ultralytics integration (YOLO family) via `UltralyticsModerator`
- Optional ONNX Runtime backend where applicable
- Simple backend switch (API/CLI flag, e.g., `--backend onnx|torch`)
- Expanded benchmarks: latency, throughput, memory on common tasks
- Documentation and examples to help you pick the right option


## Troubleshooting
- ImportError (PIL/torch/transformers):
  - Install the package (`pip install moderators`) or let auto‑install run (ensure `MODERATORS_DISABLE_AUTO_INSTALL` is unset). If you prefer manual dependency control, install extras: `pip install "moderators[transformers]"`.
- OSError: couldn’t find `config.json` / model files:
  - Check your model id or local folder path; ensure `config.json` is present.
- HTTP errors when pulling from the Hub:
  - Verify connectivity and auth (if private). Use offline mode if already cached.
- GPU not used:
  - Ensure your framework is installed with CUDA support.


## License
Apache-2.0. See `LICENSE`.
