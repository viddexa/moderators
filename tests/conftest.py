import sys
import types
from pathlib import Path

import pytest

# Add the src directory to sys.path so tests can import the package without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def fake_transformers(monkeypatch):
    """
    Replace 'from transformers import pipeline' with a fake pipeline.

    Allows testing AutoModerator.load_model flow without any network/download.
    """
    mod = types.ModuleType("transformers")

    def fake_pipeline(task, model=None, **kwargs):
        def runner(inputs):
            # Produce a simple deterministic output
            return {"label": "OK", "score": 0.9}

        return runner

    mod.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod
