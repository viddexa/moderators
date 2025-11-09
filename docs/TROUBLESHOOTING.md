# Troubleshooting Guide

## ImportError (PIL/torch/transformers)

**Problem**: Missing dependencies when trying to run Moderators.

**Solution**:

- Install the package: `pip install moderators`
- Let auto-install run (ensure `MODERATORS_DISABLE_AUTO_INSTALL` is unset)
- For manual control: `pip install "moderators[transformers]"`

## OSError: couldn't find `config.json` / model files

**Problem**: Model configuration or files not found.

**Solution**:

- Check your model ID or local folder path
- Ensure `config.json` is present in the model directory
- For Hugging Face models, verify the model ID is correct
- Try downloading the model first to verify it exists:
    ```python
    from transformers import AutoConfig
    AutoConfig.from_pretrained("your-model-id")
    ```

## HTTP errors when pulling from the Hub

**Problem**: Network errors or authentication failures when downloading models.

**Solution**:

- Verify internet connectivity
- For private models, ensure you're authenticated:
    ```bash
    huggingface-cli login
    ```
- Use offline mode if the model is already cached:
    ```bash
    moderators model-id input.jpg --local-files-only
    ```

## GPU not used

**Problem**: Model running on CPU despite having a GPU available.

**Solution**:

- Ensure your framework is installed with CUDA support
- For PyTorch, reinstall with CUDA:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```
- Verify CUDA availability:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

## Model inference is slow

**Problem**: Inference taking longer than expected.

**Suggestions**:

- Use GPU acceleration (see "GPU not used" above)
- Try smaller models (e.g., `nsfw-detector-nano` instead of larger variants)
- Consider batch processing for multiple inputs
- Check if auto-installation is downloading dependencies (first run only)

## Output format unexpected

**Problem**: Results don't match expected format.

**Solution**:

- Check the API documentation for the correct output schema
- Use `asdict()` to convert Python results to dictionaries:
    ```python
    from dataclasses import asdict
    json_ready = [asdict(r) for r in result]
    ```
- Verify you're using the correct input type (image path vs text string)

## Need More Help?

If you're still experiencing issues:

- Check the [GitHub Issues](https://github.com/viddexa/moderators/issues)
- Review the examples in the `examples/` folder
- Open a new issue with details about your environment and error messages
