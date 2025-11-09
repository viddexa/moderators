<div align="center">
  <img src="https://github.com/viddexa/moderators/releases/download/v0.1.1/logo-v2.jpeg" width="400" alt="Moderators Logo">

#

[![Moderators PYPI](https://img.shields.io/pypi/v/moderators?color=blue)](https://pypi.org/project/moderators/)
[![Moderators HuggingFace Space](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/viddexa/moderators)
[![Moderators CI](https://github.com/viddexa/moderators/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/viddexa/moderators/actions/workflows/ci.yml)
[![Moderators License](https://img.shields.io/pypi/l/moderators)](https://github.com/viddexa/moderators/blob/main/LICENSE)

Run open‑source content moderation models (NSFW, nudity, etc.) with one line — from Python or the CLI.

</div>

## ✨ Key Highlights

- One simple API and CLI
- Use any compatible Transformers model from the Hub or disk
- Normalized JSON output you can plug into your app
- Optional auto‑install of dependencies for a smooth first run

## 🚀 Performance

NSFW image detection performance of `nsfw-detector-mini` compared with [Azure Content Safety AI](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety) and [Falconsai](https://huggingface.co/Falconsai/nsfw_image_detection).

**F_safe** and **F_nsfw** below are class-wise F1 scores for safe and nsfw classes, respectively. Results show that `nsfw-detector-mini` performs better than Falconsai and Azure AI with fewer parameters.

| Model                                                                                |     F_safe |     F_nsfw |  Params |
| ------------------------------------------------------------------------------------ | ---------: | ---------: | ------: |
| [nsfw-detector-nano](https://huggingface.co/viddexa/nsfw-detection-nano)             |     96.91% |     96.87% |      4M |
| **[nsfw-detector-mini](https://huggingface.co/viddexa/nsfw-detector-mini)**          | **97.90%** | **97.89%** | **17M** |
| [Azure AI](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety) |     96.79% |     96.57% |     N/A |
| [Falconsai](https://huggingface.co/Falconsai/nsfw_image_detection)                   |     89.52% |     89.32% |     85M |

## 📦 Installation

```bash
pip install moderators
```

For detailed installation options, see the [Installation Guide](docs/INSTALLATION.md).

## 🚀 Quickstart

**Python API:**

```python
from moderators import AutoModerator

# Load from the Hugging Face Hub (e.g., NSFW image classifier)
moderator = AutoModerator.from_pretrained("viddexa/nsfw-detector-mini")

# Run on a local image path
result = moderator("/path/to/image.jpg")
print(result)
```

**CLI:**

```bash
# Image classification
moderators viddexa/nsfw-detector-mini /path/to/image.jpg

# Text classification
moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
```

## 📊 Real Output Example

![Example input image](https://img.freepik.com/free-photo/front-view-woman-doing-exercises_23-2148498678.jpg?t=st=1760435237~exp=1760438837~hmac=9a0a0a56f83d8fa52f424c7acdf4174dffc3e4d542e189398981a13af3f82b40&w=360)

Moderators normalized JSON output:

```json
[
  {
    "source_path": "",
    "classifications": { "safe": 0.9999891519546509 },
    "detections": [],
    "raw_output": { "label": "safe", "score": 0.9999891519546509 }
  },
  {
    "source_path": "",
    "classifications": { "nsfw": 0.000010843970812857151 },
    "detections": [],
    "raw_output": { "label": "nsfw", "score": 0.000010843970812857151 }
  }
]
```

## 🔍 Comparison at a Glance

| Feature             | Transformers.pipeline()       | Moderators                                                 |
| ------------------- | ----------------------------- | ---------------------------------------------------------- |
| Usage               | `pipeline("task", model=...)` | `AutoModerator.from_pretrained(...)`                       |
| Model configuration | Manual or model-specific      | Automatic via `config.json` (task inference when possible) |
| Output format       | Varies by model/pipe          | Standardized `PredictionResult` / JSON                     |
| Requirements        | Manual dependency setup       | Optional automatic `pip/uv` install                        |
| CLI                 | None or project-specific      | Built-in `moderators` CLI (JSON to stdout)                 |
| Extensibility       | Mostly one ecosystem          | Open to new integrations (same interface)                  |
| Error messages      | Vary by model                 | Consistent, task/integration-guided                        |
| Task detection      | User-provided                 | Auto-inferred from config when possible                    |

## 🎯 Pick a Model

- **From the Hub**: Pass a model ID like `viddexa/nsfw-detector-mini` or any compatible Transformers model
- **From disk**: Pass a local folder that contains a `config.json` next to your weights

Moderators detects the task and integration from the config when possible, so you don't have to specify pipelines manually.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed installation options and requirements
- [CLI Reference](docs/CLI.md) - Complete command-line usage guide
- [API Documentation](docs/API.md) - Python API reference and output formats
- [FAQ](docs/FAQ.md) - Frequently asked questions
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 📝 Examples

Small demos and benchmarking script: `examples/README.md`, `examples/benchmarks.py`

## 🗺️ Roadmap

- Ultralytics integration (YOLO family) via `UltralyticsModerator`
- Optional ONNX Runtime backend where applicable
- Simple backend switch (API/CLI flag, e.g., `--backend onnx|torch`)
- Expanded benchmarks: latency, throughput, memory on common tasks

## 📖 Citation

If you use this package in your work, please cite:

```bibtex
@article{akyon2023nudity,
  title={State-of-the-Art in Nudity Classification: A Comparative Analysis},
  author={Akyon, Fatih Cagatay and Temizel, Alptekin},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2023},
  organization={IEEE},
  doi={10.1109/ICASSPW59220.2023.10193621},
  url={https://ieeexplore.ieee.org/abstract/document/10193621/}
}
```

```bibtex
@article{akyon2022contentmoderation,
  title={Deep Architectures for Content Moderation and Movie Content Rating},
  author={Akyon, Fatih Cagatay and Temizel, Alptekin},
  journal={arXiv preprint arXiv:2212.04533},
  year={2022},
  doi={10.48550/arXiv.2212.04533},
  url={https://arxiv.org/abs/2212.04533}
}
```

## 📄 License

Apache-2.0. See [LICENSE](LICENSE).
