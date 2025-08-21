def on_predict_start(predictor):
    pass


def on_predict_end(predictor):
    pass


DEFAULT_CALLBACKS = {
    "on_predict_start": [on_predict_start],
    "on_predict_end": [on_predict_end],
}
