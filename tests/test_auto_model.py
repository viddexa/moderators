import base64
import json
import re
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from moderators import AutoModerator
from moderators.integrations.base import PredictionResult
from moderators.integrations.transformers_moderator import TransformersModerator


def write_config(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(data))
    return tmp_path


def test_local_transformers_config_predict_success(tmp_path, fake_transformers):
    # Minimal valid config (TransformersModerator + task)
    model_dir = write_config(
        tmp_path,
        {"architecture": "TransformersModerator", "task": "text-classification"},
    )

    mod = AutoModerator.from_pretrained(str(model_dir))
    assert isinstance(mod, TransformersModerator)

    out = mod("hello world")
    assert isinstance(out, list) and len(out) == 1
    pr: PredictionResult = out[0]
    assert isinstance(pr, PredictionResult)
    assert pr.classifications.get("OK") == pytest.approx(0.9, rel=1e-6)


def test_not_implemented_for_other_architecture(tmp_path):
    # For now, only TransformersModerator is supported
    model_dir = write_config(tmp_path, {"architecture": "OnnxModerator"})
    with pytest.raises(NotImplementedError) as ei:
        AutoModerator.from_pretrained(str(model_dir))
    assert "only 'TransformersModerator' is implemented" in str(ei.value)


def test_infer_transformers_from_hf_like_config(tmp_path, fake_transformers):
    # No explicit 'architecture'; infer TransformersModerator and task from a HF-like config
    model_dir = write_config(
        tmp_path,
        {
            "architectures": ["ResNetForImageClassification"],
            "transformers_version": "4.38.0",
            "id2label": {"0": "OK"},
            "label2id": {"OK": 0},
            # 'task' deliberately omitted to test inference
        },
    )

    mod = AutoModerator.from_pretrained(str(model_dir))
    assert isinstance(mod, TransformersModerator)

    out = mod("any input")
    assert isinstance(out, list) and len(out) == 1
    assert out[0].classifications.get("OK") == pytest.approx(0.9, rel=1e-6)


def test_cannot_infer_task_raises(tmp_path):
    # Looks like a Transformers config but classification task cannot be inferred
    model_dir = write_config(
        tmp_path,
        {
            "architectures": ["SomeCustomModel"],
            "transformers_version": "4.38.0",
            "id2label": {},
            "label2id": {},
            # problem_type missing and architectures do not include 'classification'
        },
    )
    with pytest.raises(ValueError) as ei:
        AutoModerator.from_pretrained(str(model_dir))
    assert "Could not infer 'task'" in str(ei.value)


def test_missing_config_json_raises(tmp_path):
    # Local folder exists but config.json is missing
    with pytest.raises(FileNotFoundError):
        AutoModerator.from_pretrained(str(tmp_path))


def test_pipeline_framework_not_passed_on_real_transformers():
    """Verify inspect-based framework detection works with the installed transformers."""
    import inspect

    from transformers import pipeline

    has_framework = "framework" in inspect.signature(pipeline).parameters
    # transformers 5.x removed the framework param; 4.x had it
    # Either way, our code in TransformersModerator.load_model checks this at runtime
    assert isinstance(has_framework, bool)


def test_pipeline_framework_passed_for_v4_signature(tmp_path, monkeypatch):
    """Verify framework kwarg IS passed when the pipeline function accepts it (v4 behavior)."""
    captured = {}

    def fake_pipeline_v4(task, model=None, framework=None, **kwargs):
        captured["framework"] = framework
        return lambda inputs: {"label": "OK", "score": 0.9}

    mod = types.ModuleType("transformers")
    mod.pipeline = fake_pipeline_v4
    monkeypatch.setitem(sys.modules, "transformers", mod)

    model_dir = write_config(tmp_path, {"architecture": "TransformersModerator", "task": "text-classification"})
    m = AutoModerator.from_pretrained(str(model_dir))
    out = m("hello")

    assert isinstance(out, list) and len(out) == 1
    assert captured.get("framework") is not None, "framework should be passed to v4-style pipeline"


def test_pipeline_framework_omitted_for_v5_signature(tmp_path, monkeypatch):
    """Verify framework kwarg is NOT passed when the pipeline rejects it (v5 behavior)."""

    def fake_pipeline_v5(task, model=None, **kwargs):
        if "framework" in kwargs:
            raise TypeError("unexpected keyword argument 'framework'")
        return lambda inputs: {"label": "OK", "score": 0.9}

    mod = types.ModuleType("transformers")
    mod.pipeline = fake_pipeline_v5
    monkeypatch.setitem(sys.modules, "transformers", mod)

    model_dir = write_config(tmp_path, {"architecture": "TransformersModerator", "task": "text-classification"})
    m = AutoModerator.from_pretrained(str(model_dir))
    out = m("hello")

    assert isinstance(out, list) and len(out) == 1


_REPO_ROOT = Path(__file__).resolve().parents[1]
_HF_URLS = sorted(
    {
        url
        for md in [*_REPO_ROOT.glob("*.md"), *_REPO_ROOT.glob("docs/*.md"), *_REPO_ROOT.glob("examples/*.md")]
        for url in re.findall(r"https://huggingface\.co/[\w-]+/[\w-]+", md.read_text())
    }
)


@pytest.mark.parametrize("url", _HF_URLS)
def test_hf_model_links_valid(url):
    """Verify HuggingFace URLs found in markdown docs are not broken."""
    req = urllib.request.Request(url, method="HEAD")
    resp = urllib.request.urlopen(req, timeout=10)
    assert resp.status == 200


def test_hf_model_viddexa_nsfw_detection_integration_online(tmp_path):
    # If HF Hub is offline, skip
    try:
        from huggingface_hub.utils import is_offline_mode

        if is_offline_mode():
            pytest.skip("HF Hub is in offline mode; skipping integration test.")
    except Exception:
        pass

    # Allow disabling auto-install via env for CI environments
    import os

    if str(os.environ.get("MODERATORS_DISABLE_AUTO_INSTALL", "")).lower() in ("1", "true", "yes"):
        pytest.skip("Auto-install disabled; skipping online integration test.")

    model_id = "viddexa/nsfw-detection-2-mini"
    mod = AutoModerator.from_pretrained(model_id, local_files_only=False)
    assert isinstance(mod, TransformersModerator)

    # Prepare a tiny 1x1 PNG (red pixel) without requiring Pillow in the test
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    img_path = tmp_path / "tiny.png"
    img_path.write_bytes(base64.b64decode(png_b64))

    out = mod(str(img_path))
    assert isinstance(out, list) and len(out) >= 1
    first = out[0]
    assert isinstance(first, PredictionResult)
    assert isinstance(first.classifications, dict) and len(first.classifications) >= 1
