from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests


def _settings_path() -> Path:
    base = Path.home() / ".moderators"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def _read_settings() -> Dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {"sync": True}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"sync": True}


class Events:
    """
    Handles the collection and transmission of anonymous usage analytics.
    Implemented as a queue with background sending and rate limiting.
    """

    def __init__(self) -> None:
        # Placeholder GA4 endpoint. Replace with real credentials for production use.
        self.url = (
            "https://www.google-analytics.com/mp/collect?measurement_id=G-XXXX&api_secret=XXXX"
        )
        self.events: List[Dict[str, Any]] = []
        self.rate_limit_seconds = 30.0
        self.last_sent_ts = 0.0
        self.metadata = self._get_metadata()
        self.enabled = self._is_enabled()

    def __call__(self, cfg: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if len(self.events) > 25:
            return
        event_data = {
            "name": cfg.get("task", "unknown_task"),
            "params": {**self.metadata, "model_id": cfg.get("model_id")},
        }
        self.events.append(event_data)

        if (time.time() - self.last_sent_ts) > self.rate_limit_seconds:
            self.send_events()

    def send_events(self) -> None:
        if not self.events:
            return

        data_payload = {"client_id": self.metadata["user_id"], "events": self.events}
        self.events = []
        self.last_sent_ts = time.time()
        threading.Thread(
            target=self._make_request, args=(data_payload,), daemon=True
        ).start()

    def _make_request(self, json_data: Dict[str, Any]) -> None:
        try:
            requests.post(self.url, json=json_data, timeout=5)
        except Exception:
            pass

    def _get_metadata(self) -> Dict[str, Any]:
        user_id = self._get_or_create_user_id()
        return {
            "user_id": user_id,
            "library": "moderators",
            "library_version": self._get_version(),
        }

    def _get_or_create_user_id(self) -> str:
        path = Path.home() / ".moderators" / "user.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if "user_id" in data:
                    return str(data["user_id"])
            except Exception:
                pass
        uid = str(uuid.uuid4())
        try:
            path.write_text(json.dumps({"user_id": uid}))
        except Exception:
            pass
        return uid

    def _is_enabled(self) -> bool:
        # Environment variable override
        if os.getenv("MODERATORS_DISABLE_ANALYTICS", "").lower() in {"1", "true", "yes"}:
            return False
        settings = _read_settings()
        return bool(settings.get("sync", True))

    def _get_version(self) -> str:
        try:
            from moderators import __version__  # type: ignore

            return str(__version__)
        except Exception:
            return "0"


# Global singleton instance
events = Events()

