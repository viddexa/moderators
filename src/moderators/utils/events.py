from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _robust_post_request(
    url: str,
    json_data: dict[str, Any],
    retries: int = 3,
    initial_wait: float = 1.0,
    timeout: int = 5,
) -> None:
    """
    Sends a POST request with JSON data to a URL, with retries on transient errors using an exponential backoff
    strategy.

    Args:
        url (str): The URL to send the request to.
        json_data (dict[str, Any]): The JSON data to send.
        retries (int): The maximum number of retries for a failed request.
        initial_wait (float): The initial wait time between retries in seconds.
        timeout (int): The timeout for each individual request in seconds.
    """
    wait_time = initial_wait
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            # 2xx status codes indicate success, exit the function.
            if 200 <= response.status_code < 300:
                return
            # 4xx client errors are not worth retrying, break the loop.
            if 400 <= response.status_code < 500:
                break
            # 5xx server errors are worth retrying.
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # In case of a timeout or connection error, proceed to the next attempt.
            pass

        # Wait only if this is not the last attempt.
        if attempt < retries:
            time.sleep(wait_time)
            wait_time *= 2  # Double the wait time for the next attempt.


def _settings_path() -> Path:
    base = Path.home() / ".moderators"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def _read_settings() -> dict[str, Any]:
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
        """Initialize the Events analytics handler."""
        self.url = (
            "https://www.google-analytics.com/mp/collect?measurement_id=G-XDJCD0WJDW&api_secret=xgz_lUC6SK-u4-EDAUFcFg"
        )
        self.events: list[dict[str, Any]] = []
        self.rate_limit_seconds = 30.0
        self.last_sent_ts = 0.0
        self.metadata = self._get_metadata()
        self.enabled = self._is_enabled()

    def __call__(self, cfg: dict[str, Any]) -> None:
        """
        Enqueue an analytics event.

        Args:
            cfg: Configuration dictionary containing task and model_id
        """
        if not self.enabled:
            return
        if len(self.events) > 25:
            return

        # Clean the event name by replacing hyphens with underscores
        original_task_name = cfg.get("task", "unknown_task")
        sanitized_event_name = original_task_name.replace("-", "_")

        event_data = {
            "name": sanitized_event_name,
            "params": {**self.metadata, "model_id": cfg.get("model_id")},
        }
        self.events.append(event_data)

        if (time.time() - self.last_sent_ts) > self.rate_limit_seconds:
            self.send_events()

    def send_events(self) -> None:
        """Send queued analytics events in a background thread."""
        if not self.events:
            return

        data_payload = {"client_id": self.metadata["user_id"], "events": self.events}
        self.events = []
        self.last_sent_ts = time.time()
        threading.Thread(target=self._make_request, args=(data_payload,), daemon=True).start()

    def _make_request(self, json_data: dict[str, Any]) -> None:
        """
        Makes a robust network request to send analytics data.

        This method now uses a helper function with retry logic.
        """
        # We are now using the more robust helper function.
        _robust_post_request(self.url, json_data)

    def _get_metadata(self) -> dict[str, Any]:
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
