# Installation Guide

## Installation Options

Choose the method that works best for your workflow:

### Using pip (recommended)

```bash
pip install moderators
```

### Using uv

```bash
uv venv --python 3.10
source .venv/bin/activate
uv add moderators
```

### From source (cloned repo)

```bash
uv sync --extra transformers
```

## Requirements

- **Python**: 3.10+
- **For image tasks**: Pillow and a deep learning framework (PyTorch preferred)
    - Moderators can auto-install these dependencies when needed

## Dependency Auto-Installation

If something is missing (e.g., `torch`, `transformers`, `Pillow`), Moderators can automatically install it via `uv` or `pip` unless you disable this feature.

To disable auto-installation:

```bash
export MODERATORS_DISABLE_AUTO_INSTALL=1
```

## Manual Dependency Control

If you prefer to manage dependencies manually, install with extras:

```bash
pip install "moderators[transformers]"
```

## Offline Mode

After caching models, you can run completely offline:

**CLI:**

```bash
moderators <model_id> <input> --local-files-only
```

**Python API:**

```python
moderator = AutoModerator.from_pretrained("model-id", local_files_only=True)
```

## GPU Support

Moderators works on CPU by default. If your deep learning framework (e.g., PyTorch) is installed with CUDA support, GPU acceleration will be used automatically.

To ensure GPU usage:

- Install PyTorch with CUDA support following [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- Verify CUDA availability in your environment
