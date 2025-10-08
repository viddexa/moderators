import importlib
from types import SimpleNamespace


def _reload_events_and_callbacks(monkeypatch, tmp_path):
    # Redirect HOME to a temporary path, so analytics remains enabled by default.
    monkeypatch.setenv("HOME", str(tmp_path))
    # Do not clear the env; tests can set it themselves if needed.

    # Reload modules in a clean environment.
    import moderators.utils.callbacks as callbacks
    import moderators.utils.events as events

    importlib.reload(events)
    importlib.reload(callbacks)
    return events, callbacks


def _inline_thread(monkeypatch, events_module):
    # Stub threading.Thread to run synchronously for testing.
    class _InlineThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args or ()
            self._kwargs = kwargs or {}
            self.daemon = daemon

        def start(self):
            # Run the target function immediately.
            if self._target:
                self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(events_module.threading, "Thread", _InlineThread, raising=True)


def test_events_sends_payload_via_requests_post(tmp_path, monkeypatch):
    events, _ = _reload_events_and_callbacks(monkeypatch, tmp_path)
    _inline_thread(monkeypatch, events)

    calls = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.setdefault("posts", []).append({"url": url, "json": json, "timeout": timeout})

        class _Resp:
            status_code = 200

        return _Resp()

    # Patch requests.post.
    monkeypatch.setattr(events.requests, "post", fake_post, raising=True)

    # Send an event.
    cfg = {"task": "text_classification", "model_id": "model_foo"}
    events.events(cfg)

    # A single post request should have been made.
    assert len(calls.get("posts", [])) == 1
    sent = calls["posts"][0]

    # The URL should be the correct endpoint.
    assert sent["url"] == events.events.url

    # Verify payload structure and fields.
    payload = sent["json"]
    assert isinstance(payload, dict)
    assert "client_id" in payload and isinstance(payload["client_id"], str)
    assert "events" in payload and isinstance(payload["events"], list) and len(payload["events"]) == 1

    ev = payload["events"][0]
    assert ev.get("name") == "text_classification"
    assert isinstance(ev.get("params"), dict)
    params = ev["params"]
    # model_id is carried in the params.
    assert params.get("model_id") == "model_foo"
    # Library metadata should be present.
    assert params.get("library") == "moderators"
    assert "library_version" in params


def test_robust_post_request_retries_and_success(tmp_path, monkeypatch):
    events, _ = _reload_events_and_callbacks(monkeypatch, tmp_path)

    call_count = {"n": 0}

    def fake_post(url, json=None, timeout=None, **kwargs):
        call_count["n"] += 1

        class _Resp:
            status_code = 500 if call_count["n"] < 3 else 200

        return _Resp()

    monkeypatch.setattr(events.requests, "post", fake_post, raising=True)

    # Set waits to zero to speed up the test.
    events._robust_post_request(
        url="https://example.com/collect",
        json_data={"x": 1},
        retries=3,
        initial_wait=0.0,
        timeout=1,
    )

    # 2 failures + 1 success -> exactly 3 calls.
    assert call_count["n"] == 3


def test_robust_post_request_no_retry_on_4xx(tmp_path, monkeypatch):
    events, _ = _reload_events_and_callbacks(monkeypatch, tmp_path)

    call_count = {"n": 0}

    def fake_post(url, json=None, timeout=None, **kwargs):
        call_count["n"] += 1

        class _Resp:
            status_code = 400

        return _Resp()

    monkeypatch.setattr(events.requests, "post", fake_post, raising=True)

    events._robust_post_request(
        url="https://example.com/collect",
        json_data={"x": 1},
        retries=5,
        initial_wait=0.0,
        timeout=1,
    )

    # No retry on 4xx status codes -> exactly 1 call.
    assert call_count["n"] == 1


def test_callbacks_integration_triggers_event_and_rate_limit(tmp_path, monkeypatch):
    events, callbacks = _reload_events_and_callbacks(monkeypatch, tmp_path)
    _inline_thread(monkeypatch, events)

    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls["n"] += 1

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr(events.requests, "post", fake_post, raising=True)

    predictor = SimpleNamespace(config={"task": "image-classification"}, model_id="mid-123")

    # The first call should trigger a post request.
    callbacks.on_predict_start(predictor)
    assert calls["n"] == 1

    # Rate limit is active -> a second call soon after should not make an extra post.
    callbacks.on_predict_start(predictor)
    assert calls["n"] == 1
