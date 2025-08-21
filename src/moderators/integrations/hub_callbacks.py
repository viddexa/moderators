try:
    # Optional analytics bridge (not implemented in MVP yet)
    from moderators.utils.analytics import events  # type: ignore

    def on_predict_start_for_analytics(predictor):
        try:
            events(cfg={"task": predictor.config.get("task"), "model_id": predictor.model_id})
        except Exception:
            pass

    HUB_CALLBACKS = {
        "on_predict_start": [on_predict_start_for_analytics],
    }
except Exception:
    HUB_CALLBACKS = {}
