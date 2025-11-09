# Frequently Asked Questions

## Which tasks are supported?

Image and text classification via Transformers (e.g., NSFW detection, sentiment/toxicity analysis). More tasks can be added over time.

## Does it need a GPU?

No. CPU is fine for small models. If your framework has CUDA installed, it will automatically use GPU acceleration.

## How are dependencies handled?

If something is missing (e.g., `torch`, `transformers`, `Pillow`), Moderators can auto-install via `uv` or `pip` unless you disable it.

To disable auto-installation:

```bash
export MODERATORS_DISABLE_AUTO_INSTALL=1
```

For manual dependency control:

```bash
pip install "moderators[transformers]"
```

## Can I run offline?

Yes. Use `--local-files-only` in the CLI or `local_files_only=True` in Python after you have the model cached.

**CLI:**

```bash
moderators model-id input.jpg --local-files-only
```

**Python:**

```python
moderator = AutoModerator.from_pretrained("model-id", local_files_only=True)
```

## What does "normalized output" mean?

Regardless of the underlying pipeline, you always get the same result schema (`PredictionResult` with classifications/detections/raw_output), so your application code stays simple and consistent across different models.

## Can I use my own custom models?

Yes! As long as your model has a `config.json` file and is compatible with Transformers, you can use it with Moderators. Just point to the model directory or Hugging Face model ID.

## How do I contribute or request features?

Check out the [GitHub repository](https://github.com/viddexa/moderators) to open issues or submit pull requests. Feature requests and contributions are welcome!
