def test_callbacks_default_dict():
    from moderators.utils.callbacks import DEFAULT_CALLBACKS

    assert "on_predict_start" in DEFAULT_CALLBACKS
    assert "on_predict_end" in DEFAULT_CALLBACKS
