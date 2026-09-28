"""
Model selection helpers — read metrics logs and pick the best checkpoint for serving.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app import config

logger = logging.getLogger(__name__)

_SERVEABLE_MODELS = (
    "transformer",
    "lstm",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
)
_DEFAULT_PRIORITY = _SERVEABLE_MODELS


def load_latest_metrics() -> Dict[str, Any]:
    """Return the newest metrics JSON payload, or an empty dict."""
    if not config.LOG_DIR.exists():
        return {}

    files = sorted(config.LOG_DIR.glob("metrics_*.json"), reverse=True)
    if not files:
        return {}

    try:
        with open(files[0], "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.error("Failed to read metrics file %s: %s", files[0], exc)
        return {}


def get_best_model_name() -> Optional[str]:
    """Return the model with the lowest validation RMSE from the latest metrics run."""
    payload = load_latest_metrics()
    explicit = payload.get("best_model")
    if isinstance(explicit, str) and explicit in _SERVEABLE_MODELS:
        return explicit

    models: Dict[str, Dict[str, float]] = payload.get("models", {})
    candidates: list[tuple[str, float]] = []

    for name in _SERVEABLE_MODELS:
        metrics = models.get(name, {})
        val_rmse = metrics.get("val_rmse")
        if val_rmse is None:
            val_rmse = metrics.get("rmse")
        if isinstance(val_rmse, (int, float)) and val_rmse == val_rmse:
            candidates.append((name, float(val_rmse)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1])
    return candidates[0][0]


def checkpoint_exists(name: str) -> bool:
    """Return True when a known checkpoint file exists for *name*."""
    mapping = {
        "transformer": "transformer.pt",
        "lstm": "lstm.pt",
        "xgboost": "xgboost.joblib",
        "lightgbm": "lightgbm.joblib",
        "catboost": "catboost.joblib",
        "ensemble": "ensemble.joblib",
    }
    filename = mapping.get(name)
    if not filename:
        return False
    return (config.CHECKPOINT_DIR / filename).exists()


def readiness_check() -> Dict[str, Any]:
    """Summarise whether the API is ready to serve forecasts."""
    data_ready = config.FEATURES_FILE.exists() or config.RAW_DATA_FILE.exists()
    scaler_ready = config.SCALER_FILE.exists()
    model_ready = any(checkpoint_exists(name) for name in _SERVEABLE_MODELS)
    metrics_ready = bool(load_latest_metrics().get("models"))

    if data_ready and scaler_ready and model_ready:
        status = "healthy"
    elif model_ready and data_ready:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "data": data_ready,
        "scaler": scaler_ready,
        "model_checkpoint": model_ready,
        "metrics_logged": metrics_ready,
        "best_model": get_best_model_name(),
    }


def resolve_serving_model() -> Tuple[str, str]:
    """Pick the model name and kind (``sequence`` | ``tabular``) used for /forecast."""
    best = get_best_model_name()
    if best and checkpoint_exists(best):
        if best == "ensemble":
            return best, "ensemble"
        kind = "sequence" if best in ("transformer", "lstm") else "tabular"
        return best, kind

    for name in _DEFAULT_PRIORITY:
        if checkpoint_exists(name):
            if name == "ensemble":
                return name, "ensemble"
            kind = "sequence" if name in ("transformer", "lstm") else "tabular"
            logger.info("No metrics file — falling back to checkpoint priority: %s", name)
            return name, kind

    raise RuntimeError("No trained model checkpoint available.")
