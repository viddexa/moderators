# Moderators

[![Moderators PYPI](https://img.shields.io/pypi/v/moderators?color=blue)](https://pypi.org/project/moderators/)
[![Moderators HuggingFace Space](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/viddexa/moderators)
[![Moderators CI](https://github.com/viddexa/moderators/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/viddexa/moderators/actions/workflows/ci.yml)
[![Moderators License](https://img.shields.io/pypi/l/moderators)](https://github.com/viddexa/moderators/blob/main/LICENSE)

# TODO: refactor readme to target users instead of maintainers

This repository provides an extensible core skeleton for content moderation. Phase 1 includes:
- Standard data classes (Box, PredictionResult)
- BaseModerator flow (predict → _preprocess → _predict → _postprocess)
- ModelHubMixin-based `AutoModerator` factory (reads `config.json` from HF Hub or local)
- CLI: `moderators` (load and run inference)

First integration: Transformers.

## Installation

Create Python environment (Python 3.10+ recommended):

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Install with pip:

```bash
pip install moderators[transformers]
```

Install with uv:

```bash
uv add "moderators[transformers]"
```

Install from source:

```bash
uv sync --extra transformers
```

## Quick Start

```python
from moderators.auto_model import AutoModerator

moderator = AutoModerator.from_pretrained("org/model")  # or a local folder path
results = moderator("some input")
print(results)
```

`config.json` example (Transformers):
```json
{
  "architecture": "TransformersModerator",
  "task": "image-classification"
}
```

- Naming convention: the `XyzModerator` class must be defined in `moderators/integrations/xyz_moderator.py`.
- Note: `AutoModerator` is a factory class; it returns the actual integration instance.

## Automatic dependency installation
When using the Transformers integration, the library may auto-install missing dependencies at runtime:
- transformers
- A deep learning framework (PyTorch preferred: torch)
- Pillow (for image tasks)

It uses `uv` if available, otherwise falls back to `pip`. Disable auto-install via:
```
export MODERATORS_DISABLE_AUTO_INSTALL=1
```

## Usage Overview
`AutoModerator.from_pretrained("org/model")` dynamically loads the correct integration class based on the `"architecture"` field in `config.json`.

## Command Line (CLI)
Run models directly from the terminal.

Usage:
```
moderators <model_id_or_local_dir> <input> [--local-files-only]
```

Examples:
- Text classification:
```
moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
```

- Image classification (Falconsai/nsfw_image_detection) with a local image:
```
moderators Falconsai/nsfw_image_detection /path/to/image.jpg
```

Notes:
- The CLI prints JSON to stdout.
- Use `--local-files-only` to force offline usage if all files are already cached.

## Transformers config inference
If `"architecture"` is missing but the config looks like a Transformers model (e.g., has `architectures`, `transformers_version`, `id2label`/`label2id`), the factory assumes:
- `architecture = "TransformersModerator"`
- It tries to infer `"task"` (e.g., classification). If it cannot infer, you must specify `"task"` explicitly (e.g., `"image-classification"`).

## Callbacks
Moderators run a minimal callback system around prediction:
- `on_predict_start(moderator)` is called before prediction.
- `on_predict_end(moderator)` is called after prediction.

By default, `on_predict_start` enqueues a lightweight analytics event (see below). You can customize per-instance callbacks:
```python
mod = AutoModerator.from_pretrained("org/model")
# Disable all start callbacks (including analytics)
mod.callbacks["on_predict_start"].clear()
# Or add your own callback
def my_callback(m):
    print("Starting inference for", m.model_id)
mod.callbacks["on_predict_start"].append(my_callback)
```

## Anonymous Telemetry

We believe in providing our users with full control over their data. By default, our package is configured to collect analytics to help improve the experience for all users. However, we respect that some users may prefer to opt out of this data collection.

To opt out of sending analytics, you can simply create `~/.moderators/settings.json` file with `"sync": false`. This ensures that no data is transmitted from your machine to our analytics tools.

## Limitations (Phase 1)
- Only `TransformersModerator` is supported; other architectures raise `NotImplementedError`.
- Image tasks require Pillow and at least one DL framework (preferably PyTorch). The library may attempt auto-install, otherwise it will raise an error.

## Integrations
- Transformers integration
