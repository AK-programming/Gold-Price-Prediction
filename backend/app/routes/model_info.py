"""
Gold Price Forecasting System — Model Info & Explainability Routes

FastAPI router for inspecting trained models, retrieving metrics,
SHAP feature importance, attention heatmaps, and triggering retraining.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix=config.API_PREFIX, tags=["models"])


# ------------------------------------------------------------------ #
# Pydantic response models
# ------------------------------------------------------------------ #


class ModelSummary(BaseModel):
    """Summary of a single trained model."""

    name: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_exists: bool = False


class ModelsListResponse(BaseModel):
    """Response schema for ``GET /models``."""

    models: List[ModelSummary]
    latest_run: Optional[str] = None


class MetricsDetailResponse(BaseModel):
    """Detailed metrics for a single model."""

    name: str
    metrics: Dict[str, float]
    per_horizon: Optional[Dict[str, Dict[str, float]]] = None


class ShapResponse(BaseModel):
    """SHAP feature importance payload."""

    feature_importance: List[Dict[str, Any]] = Field(default_factory=list)
    num_samples: int = 0
    num_features: int = 0


class AttentionResponse(BaseModel):
    """Attention heatmap payload."""

    temporal_importance: List[float] = Field(default_factory=list)
    sequence_length: int = 0
    attention_shape: List[int] = Field(default_factory=list)
    input_window: int = config.INPUT_SEQUENCE_LENGTH


class RetrainResponse(BaseModel):
    """Acknowledgement that retraining has been queued."""

    status: str = "queued"
    message: str = "Retraining started in background."


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_KNOWN_MODELS = [
    "transformer",
    "lstm",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
]


def _load_latest_metrics() -> Dict[str, Any]:
    """Find and parse the newest ``metrics_*.json`` in ``config.LOG_DIR``.

    Returns
    -------
    dict
        Parsed JSON payload, or empty dict if none found.
    """
    log_dir = config.LOG_DIR
    if not log_dir.exists():
        return {}

    files = sorted(log_dir.glob("metrics_*.json"), reverse=True)
    if not files:
        return {}

    try:
        with open(files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded latest metrics from %s", files[0].name)
        return data
    except Exception as exc:
        logger.error("Error reading metrics file %s: %s", files[0], exc)
        return {}


def _run_retraining() -> None:
    """Background task: execute full training pipeline."""
    logger.info("Background retraining started.")
    try:
        from app.utils.trainer import TrainingOrchestrator

        orchestrator = TrainingOrchestrator()
        orchestrator.run_full_pipeline()
        logger.info("Background retraining finished successfully.")
    except Exception as exc:
        logger.error("Background retraining failed: %s", exc, exc_info=True)


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """List all known models and their latest evaluation metrics."""
    latest = _load_latest_metrics()
    model_metrics: Dict[str, Dict[str, float]] = latest.get("models", {})
    latest_run = latest.get("timestamp")

    summaries: List[ModelSummary] = []
    for name in _KNOWN_MODELS:
        ckpt_exists = any(
            (config.CHECKPOINT_DIR / f"{name}{ext}").exists()
            for ext in (".joblib", ".pt", ".pth", ".ckpt")
        )
        summaries.append(
            ModelSummary(
                name=name,
                metrics=model_metrics.get(name, {}),
                checkpoint_exists=ckpt_exists,
            )
        )

    return ModelsListResponse(models=summaries, latest_run=latest_run)


@router.get("/models/{name}/metrics", response_model=MetricsDetailResponse)
async def model_metrics(name: str) -> MetricsDetailResponse:
    """Return detailed metrics for a specific model.

    Parameters
    ----------
    name : str
        Model identifier (e.g. ``transformer``, ``xgboost``, ``ensemble``).
    """
    if name not in _KNOWN_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{name}'. Known: {_KNOWN_MODELS}",
        )

    latest = _load_latest_metrics()
    model_metrics_map: Dict[str, Dict[str, float]] = latest.get("models", {})

    if name not in model_metrics_map:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics recorded for model '{name}'. Train first.",
        )

    return MetricsDetailResponse(
        name=name,
        metrics=model_metrics_map[name],
    )


@router.get("/explainability/shap", response_model=ShapResponse)
async def shap_importance() -> ShapResponse:
    """Return SHAP-based feature importance for the best tree model."""
    try:
        import joblib
        import numpy as np

        from app.explainability.shap_explainer import ShapExplainer

        # Try to load a pre-computed SHAP result
        shap_cache = config.LOG_DIR / "shap_results.json"
        if shap_cache.exists():
            with open(shap_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return ShapResponse(**cached)

        # Otherwise compute on the fly for the first available tree model
        for name, filename in [
            ("xgboost", "xgboost.joblib"),
            ("lightgbm", "lightgbm.joblib"),
            ("catboost", "catboost.joblib"),
        ]:
            ckpt = config.CHECKPOINT_DIR / filename
            if ckpt.exists():
                model = joblib.load(ckpt)
                break
        else:
            raise HTTPException(
                status_code=503,
                detail="No tree model checkpoint found for SHAP analysis.",
            )

        # Load a sample of data
        import pandas as pd

        if not config.FEATURES_FILE.exists():
            raise HTTPException(status_code=503, detail="Feature data not available.")

        df = pd.read_parquet(config.FEATURES_FILE)
        feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]
        X_sample = df[feature_cols].dropna().tail(200).values.astype(np.float32)

        explainer = ShapExplainer(feature_names=feature_cols)
        explainer.explain_tree_model(model, X_sample)

        result = explainer.to_json()

        # Cache for future requests
        with open(shap_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return ShapResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("SHAP endpoint failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/explainability/attention", response_model=AttentionResponse)
async def attention_heatmap() -> AttentionResponse:
    """Return temporal attention importance from the Transformer model."""
    try:
        import numpy as np

        from app.explainability.attention_viz import AttentionVisualizer

        # Try pre-computed cache
        attn_cache = config.LOG_DIR / "attention_results.json"
        if attn_cache.exists():
            with open(attn_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return AttentionResponse(**cached)

        # Load Transformer checkpoint
        transformer_ckpt = config.CHECKPOINT_DIR / "transformer.pt"
        if not transformer_ckpt.exists():
            transformer_ckpt = config.CHECKPOINT_DIR / "transformer.pth"
        if not transformer_ckpt.exists():
            raise HTTPException(
                status_code=503,
                detail="Transformer checkpoint not found for attention analysis.",
            )

        import torch

        from app.models.transformer import TransformerForecaster

        model = torch.load(transformer_ckpt, map_location="cpu")
        model.eval()

        # Load sample input
        import pandas as pd

        if not config.FEATURES_FILE.exists():
            raise HTTPException(status_code=503, detail="Feature data not available.")

        df = pd.read_parquet(config.FEATURES_FILE)
        feature_cols = [c for c in df.columns if c != config.TARGET_COLUMN]
        recent = df[feature_cols].dropna().tail(config.INPUT_SEQUENCE_LENGTH)
        X_sample = recent.values.astype(np.float32)[np.newaxis, :, :]

        viz = AttentionVisualizer()
        viz.extract_attention(model, X_sample)
        viz.get_temporal_importance()

        result = viz.to_json()

        # Cache
        with open(attn_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return AttentionResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Attention endpoint failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retrain", response_model=RetrainResponse)
async def retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    """Trigger full retraining pipeline in the background."""
    logger.info("Retraining requested via API.")
    background_tasks.add_task(_run_retraining)
    return RetrainResponse()
