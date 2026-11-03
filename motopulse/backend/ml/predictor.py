"""
MotoPulse — ML inference wrapper.

Loads the trained Random Forest bundle once at import time and exposes
`predict_from_features()` for the simulator / API layer. This is the ONLY
place in the backend that calls the model — every sensor tick's features
pass through here, so predictions are genuine model output, not hard-coded.
"""
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                "model.joblib not found. Run `python ml/train_model.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_from_features(feature_dict: dict) -> dict:
    """
    feature_dict must contain every key in config.FEATURE_NAMES.
    Returns {"label": "SAFE"|"AGGRESSIVE", "label_id": 0|1, "confidence": float,
             "probabilities": {"SAFE": p0, "AGGRESSIVE": p1}}
    """
    bundle = _load()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]

    x = np.array([[feature_dict[name] for name in feature_names]])
    proba = model.predict_proba(x)[0]
    label_id = int(np.argmax(proba))
    return {
        "label": class_names[label_id],
        "label_id": label_id,
        "confidence": float(proba[label_id]),
        "probabilities": {class_names[i]: float(p) for i, p in enumerate(proba)},
    }


def is_ready() -> bool:
    return MODEL_PATH.exists()
