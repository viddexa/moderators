import json
from pathlib import Path

from moderators.auto_model import AutoModerator


class _FakeTempDir:
    def __init__(self, prefix: str = "moderators_push_"):
        import tempfile

        self.name = tempfile.mkdtemp(prefix=prefix)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        # Klasörü bilerek silmiyoruz; test sonrasında pytest tmp cleanup zaten yapar.
        return False


def write_config(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(data))
    return tmp_path


def test_push_to_hub_offline(tmp_path, monkeypatch, fake_transformers):
    # Ağ ve ağır bağımlılıkları devre dışı bırak
    monkeypatch.setenv("MODERATORS_DISABLE_AUTO_INSTALL", "1")
    monkeypatch.setattr(
        "moderators.integrations.transformers_moderator.ensure_dl_framework",
        lambda installer: "pt",
        raising=True,
    )

    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, private=None, exist_ok=True, **kw):
            calls["create_repo"] = {
                "repo_id": repo_id,
                "repo_type": "model",
                "private": private,
                "exist_ok": exist_ok,
            }
            from types import SimpleNamespace

            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, repo_id, repo_type, folder_path, commit_message=None, token=None, **kw):
            calls["upload_folder"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "folder_path": folder_path,
                "commit_message": commit_message,
            }
            return {"ok": True}

    # Doğru namespace: hub_mixin içindeki HfApi sembolünü patch'le
    monkeypatch.setattr(
        "huggingface_hub.hub_mixin.HfApi",
        lambda *a, **k: FakeApi(),
        raising=True,
    )

    # Geçici klasörün silinmesini engelle: SoftTemporaryDirectory'yi patch'le
    monkeypatch.setattr(
        "huggingface_hub.hub_mixin.SoftTemporaryDirectory",
        lambda *a, **k: _FakeTempDir(prefix="moderators_push_"),
        raising=True,
    )

    # Yerel bir TransformersModerator konfigürasyonu ile yükle
    model_dir = write_config(tmp_path, {"architecture": "TransformersModerator", "task": "text-classification"})
    mod = AutoModerator.from_pretrained(str(model_dir))

    # push_to_hub çağrısı (ağ yok, FakeApi çalışacak)
    repo_id = "user/repo-for-tests"
    mod.push_to_hub(repo_id, commit_message="test commit", token="fake-token")

    # upload_folder çağrıldı mı?
    assert "upload_folder" in calls
    up = calls["upload_folder"]
    folder_path = Path(up["folder_path"])
    assert folder_path.exists()

    # Kaydedilen config.json doğrula
    cfg = json.loads((folder_path / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("architecture") == "TransformersModerator"
    assert cfg.get("task") == "text-classification"
