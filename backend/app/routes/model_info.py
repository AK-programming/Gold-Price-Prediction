"""
Model metadata and explainability routes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix=config.API_PREFIX, tags=["models"])


class ModelSummary(BaseModel):
    name: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_exists: bool = False


class ModelsListResponse(BaseModel):
    models: List[ModelSummary]
    latest_run: Optional[str] = None
    best_model: Optional[str] = None


class MetricsDetailResponse(BaseModel):
    name: str
    metrics: Dict[str, float]
    per_horizon: Optional[Dict[str, Dict[str, float]]] = None


class ShapResponse(BaseModel):
    feature_importance: List[Dict[str, Any]] = Field(default_factory=list)
    num_samples: int = 0
    num_features: int = 0


class AttentionResponse(BaseModel):
    temporal_importance: List[float] = Field(default_factory=list)
    sequence_length: int = 0
    attention_shape: List[int] = Field(default_factory=list)
    input_window: int = config.INPUT_SEQUENCE_LENGTH


class RetrainResponse(BaseModel):
    status: str = "queued"
    message: str = "Retraining started in background."


_KNOWN_MODELS = [
    "transformer",
    "lstm",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
]


def _normalize_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    """Map persisted metric keys to the API shape expected by the dashboard."""
    if not raw:
        return {}

    def pick(*keys: str) -> float | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, (int, float)) and value == value:
                return float(value)
        return None

    normalized: Dict[str, float] = {}
    for key, value in (
        ("rmse", pick("test_rmse", "rmse")),
        ("mae", pick("test_mae", "mae")),
        ("mape", pick("test_mape", "mape")),
        ("r2", pick("test_r2", "r2")),
        ("directional_accuracy", pick("test_directional_accuracy", "directional_accuracy")),
        ("val_rmse", pick("val_rmse")),
    ):
        if value is not None:
            normalized[key] = value

    return normalized


def _load_latest_metrics() -> Dict[str, Any]:
    if not config.LOG_DIR.exists():
        return {}

    files = sorted(config.LOG_DIR.glob("metrics_*.json"), reverse=True)
    if not files:
        return {}

    try:
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Error reading metrics file %s: %s", files[0], exc)
        return {}


def _flat_feature_names(feature_cols: list[str]) -> list[str]:
    return [
        f"{col}_t{step}"
        for step in range(config.INPUT_SEQUENCE_LENGTH)
        for col in feature_cols
    ]


def _load_tree_checkpoint() -> tuple[str, Any, list[str] | None]:
    import joblib

    for name, filename in [
        ("xgboost", "xgboost.joblib"),
        ("lightgbm", "lightgbm.joblib"),
        ("catboost", "catboost.joblib"),
    ]:
        ckpt = config.CHECKPOINT_DIR / filename
        if not ckpt.exists():
            continue

        payload = joblib.load(ckpt)
        if isinstance(payload, dict) and "model" in payload:
            return name, payload["model"], payload.get("feature_names")
        return name, payload, None

    raise HTTPException(
        status_code=503,
        detail="No tree model checkpoint found for SHAP analysis.",
    )


def _run_retraining() -> None:
    logger.info("Background retraining started.")
    try:
        from app.utils.trainer import TrainingOrchestrator

        orchestrator = TrainingOrchestrator()
        orchestrator.run_full_pipeline()
        logger.info("Background retraining finished successfully.")
    except Exception as exc:
        logger.error("Background retraining failed: %s", exc, exc_info=True)


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
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
                metrics=_normalize_metrics(model_metrics.get(name, {})),
                checkpoint_exists=ckpt_exists,
            )
        )

    return ModelsListResponse(
        models=summaries,
        latest_run=latest_run,
        best_model=latest.get("best_model"),
    )


@router.get("/models/{name}/metrics", response_model=MetricsDetailResponse)
async def model_metrics(name: str) -> MetricsDetailResponse:
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
        metrics=_normalize_metrics(model_metrics_map[name]),
    )


@router.get("/explainability/shap", response_model=ShapResponse)
async def shap_importance() -> ShapResponse:
    try:
        import numpy as np
        import pandas as pd

        from app.explainability.shap_explainer import ShapExplainer
        from app.pipeline.preprocessor import create_tabular_dataset, preprocess_data

        shap_cache = config.LOG_DIR / "shap_results.json"
        if shap_cache.exists():
            with open(shap_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("feature_importance"):
                return ShapResponse(**cached)

        model_name, model, saved_feature_names = _load_tree_checkpoint()

        if not config.FEATURES_FILE.exists():
            raise HTTPException(status_code=503, detail="Feature data not available.")

        features_df = pd.read_parquet(config.FEATURES_FILE)
        _, _, test_scaled, feature_cols, target_idx, _ = preprocess_data(features_df)
        X_sample, _ = create_tabular_dataset(test_scaled, target_idx=target_idx)
        X_sample = X_sample[-200:].astype(np.float32)

        feature_names = (
            saved_feature_names
            if saved_feature_names and len(saved_feature_names) == X_sample.shape[1]
            else _flat_feature_names(feature_cols)
        )

        explainer = ShapExplainer(feature_names=feature_names)
        explainer.explain_tree_model(model, X_sample)
        result = explainer.to_json()

        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(shap_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info("SHAP response generated from %s", model_name)
        return ShapResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("SHAP endpoint failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/explainability/attention", response_model=AttentionResponse)
async def attention_heatmap() -> AttentionResponse:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import torch

        from app.explainability.attention_viz import AttentionVisualizer
        from app.models import GoldTransformer

        attn_cache = config.LOG_DIR / "attention_results.json"
        if attn_cache.exists():
            with open(attn_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("temporal_importance"):
                return AttentionResponse(**cached)

        transformer_ckpt = config.CHECKPOINT_DIR / "transformer.pt"
        if not transformer_ckpt.exists():
            raise HTTPException(
                status_code=503,
                detail="Transformer checkpoint not found for attention analysis.",
            )
        if not config.FEATURES_FILE.exists():
            raise HTTPException(status_code=503, detail="Feature data not available.")
        if not config.SCALER_FILE.exists():
            raise HTTPException(status_code=503, detail="Scaler not available.")

        checkpoint = torch.load(transformer_ckpt, map_location="cpu")
        feature_cols = checkpoint.get("feature_cols")
        num_features = int(checkpoint["num_features"])

        df = pd.read_parquet(config.FEATURES_FILE)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        if not feature_cols:
            feature_cols = list(df.columns)
            feature_cols.remove(config.TARGET_COLUMN)
            feature_cols.append(config.TARGET_COLUMN)

        recent = df[feature_cols].tail(config.INPUT_SEQUENCE_LENGTH)
        scaler = joblib.load(config.SCALER_FILE)
        X_sample = scaler.transform(recent.values.astype(np.float64))
        X_sample = X_sample.astype(np.float32)[np.newaxis, :, :]

        model = GoldTransformer(num_features=num_features)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        viz = AttentionVisualizer()
        viz.extract_attention(model, X_sample)
        viz.get_temporal_importance()
        result = viz.to_json()

        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    logger.info("Retraining requested via API.")
    background_tasks.add_task(_run_retraining)
    return RetrainResponse()
