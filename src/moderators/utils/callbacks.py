from moderators.utils.events import events


def on_predict_start(predictor):
    """
    Callback executed before prediction starts.

    Enqueues analytics event with task and model_id information.
    """
    # Add analytics event
    try:
        cfg = {
            "task": predictor.config.get("task", "unknown_task"),
            "model_id": getattr(predictor, "model_id", None),
        }
        events(cfg)
    except Exception:
        # Callback should not break inference flow
        pass


def on_predict_end(predictor):
    """Callback executed after prediction ends."""
    pass


DEFAULT_CALLBACKS = {
    "on_predict_start": [on_predict_start],
    "on_predict_end": [on_predict_end],
}
