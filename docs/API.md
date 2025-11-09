# API Reference

## Output Format

Moderators provides normalized, consistent output regardless of the underlying model or framework.

### Python API

Results are returned as a list of `PredictionResult` dataclass instances:

```python
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

### JSON Format (CLI)

The CLI outputs the same structure as JSON:

```json
[
    {
        "source_path": "",
        "classifications": { "NSFW": 0.9821 },
        "detections": [],
        "raw_output": { "label": "NSFW", "score": 0.9821 }
    }
]
```

## Converting Python Results to JSON

Use `dataclasses.asdict()` to convert Python results to JSON-ready dictionaries:

```python
from dataclasses import asdict
from moderators import AutoModerator

moderator = AutoModerator.from_pretrained("viddexa/nsfw-detector-mini")
result = moderator("/path/to/image.jpg")
json_ready = [asdict(r) for r in result]
print(json_ready)
```

## PredictionResult Fields

- **`source_path`** (str): Path to the input file, or empty string for text/direct input
- **`classifications`** (dict): Normalized classification results as `{label: score}` pairs
- **`detections`** (list): Object detection results (empty for classification tasks)
- **`raw_output`** (dict): Original model output for reference

## AutoModerator API

### Loading Models

**From Hugging Face Hub:**

```python
from moderators import AutoModerator

moderator = AutoModerator.from_pretrained("viddexa/nsfw-detector-mini")
```

**From local directory:**

```python
moderator = AutoModerator.from_pretrained("/path/to/model")
```

**With offline mode:**

```python
moderator = AutoModerator.from_pretrained("model-id", local_files_only=True)
```

### Running Inference

**Image input:**

```python
result = moderator("/path/to/image.jpg")
```

**Text input:**

```python
result = moderator("Text to classify")
```

## Task Detection

Moderators automatically detects the task type from the model's `config.json` when possible, so you don't need to specify the task manually.

Supported tasks:

- Image classification (e.g., NSFW detection)
- Text classification (e.g., sentiment analysis, toxicity detection)

## Model Selection

- **From the Hub**: Pass a model ID like `viddexa/nsfw-detector-mini` or any compatible Transformers model
- **From disk**: Pass a local folder that contains a `config.json` next to your model weights

The system automatically infers the task and integration from the config when possible.
