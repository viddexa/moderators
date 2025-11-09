# Command Line Reference

Run moderation models from your terminal and get normalized JSON output to stdout.

## Usage

```bash
moderators <model_id_or_local_dir> <input> [--local-files-only]
```

### Arguments

- `<model_id_or_local_dir>`: Hugging Face model ID (e.g., `viddexa/nsfw-detector-mini`) or path to local model directory
- `<input>`: Input data - either a file path (for images) or text string (for text models)
- `--local-files-only` (optional): Force offline mode using cached files only

## Examples

### Text Classification

```bash
moderators distilbert/distilbert-base-uncased-finetuned-sst-2-english "I love this!"
```

### Image Classification

```bash
moderators viddexa/nsfw-detector-mini /path/to/image.jpg
```

### Offline Mode

```bash
moderators viddexa/nsfw-detector-mini /path/to/image.jpg --local-files-only
```

## Output Format

The CLI prints a JSON array to stdout, making it easy to pipe or parse:

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

## Tips

- The output is a single JSON array per execution
- Use `--local-files-only` to ensure no network requests are made
- Pipe output to `jq` for advanced JSON processing:
    ```bash
    moderators viddexa/nsfw-detector-mini image.jpg | jq '.[0].classifications'
    ```
- Redirect output to a file for batch processing:
    ```bash
    moderators viddexa/nsfw-detector-mini image.jpg > results.json
    ```
