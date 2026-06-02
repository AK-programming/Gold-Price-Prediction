"""
Gold Price Forecasting System — Prediction Routes

FastAPI router for serving gold price forecasts and historical data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix=config.API_PREFIX, tags=["predictions"])


# ------------------------------------------------------------------ #
# Pydantic response models
# ------------------------------------------------------------------ #


class ForecastResponse(BaseModel):
    """Response schema for the ``/forecast`` endpoint."""

    dates: List[str] = Field(..., description="ISO-format forecast dates")
    predictions: List[float] = Field(..., description="Point predictions (median)")
    confidence_low: List[float] = Field(
        ..., description="Lower bound (10th percentile)"
    )
    confidence_high: List[float] = Field(
        ..., description="Upper bound (90th percentile)"
    )
    current_price: float = Field(..., description="Most recent actual gold price")
    model_used: str = Field("ensemble", description="Model that produced the forecast")
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC timestamp of forecast generation",
    )


class HistoricalPoint(BaseModel):
    """Single historical data point."""

    date: str
    actual: float
    predicted: Optional[float] = None


class HistoricalResponse(BaseModel):
    """Response schema for the ``/historical`` endpoint."""

    data: List[HistoricalPoint]
    count: int


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _load_latest_model() -> Any:
    """Load the most recently saved ensemble (or best single model).

    Returns
    -------
    model
        A fitted model with a ``predict`` method.

    Raises
    ------
    HTTPException
        If no trained model checkpoint exists.
    """
    try:
        from app.models.ensemble import EnsembleForecaster

        model = EnsembleForecaster.load(config.CHECKPOINT_DIR / "ensemble.joblib")
        logger.info("Loaded ensemble model from checkpoint.")
        return model
    except Exception:
        logger.debug("Ensemble checkpoint not found — trying individual models.")

    # Fallback order: XGBoost → LightGBM → CatBoost
    for name, filename in [
        ("xgboost", "xgboost.joblib"),
        ("lightgbm", "lightgbm.joblib"),
        ("catboost", "catboost.joblib"),
    ]:
        ckpt = config.CHECKPOINT_DIR / filename
        if ckpt.exists():
            try:
                import joblib

                model = joblib.load(ckpt)
                logger.info("Loaded fallback model '%s' from %s.", name, ckpt)
                return model
            except Exception as exc:
                logger.warning("Failed to load %s: %s", ckpt, exc)

    raise HTTPException(
        status_code=503,
        detail="No trained model available. Please run training first.",
    )


def _load_recent_prices(days: int = 365) -> np.ndarray:
    """Load the most recent *days* of actual gold closing prices.

    Returns
    -------
    np.ndarray
        Shape ``(days,)`` or shorter if data is unavailable.
    """
    try:
        import pandas as pd

        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
        elif config.RAW_DATA_FILE.exists():
            df = pd.read_parquet(config.RAW_DATA_FILE)
        else:
            return np.array([])

        col = config.TARGET_COLUMN
        if col not in df.columns:
            logger.warning("Target column '%s' not in data.", col)
            return np.array([])

        prices = df[col].dropna().values
        return prices[-days:]
    except Exception as exc:
        logger.error("Failed to load recent prices: %s", exc)
        return np.array([])


def _generate_forecast_dates(n_days: int) -> List[str]:
    """Create a list of upcoming trading-day date strings (skipping weekends)."""
    dates: List[str] = []
    current = datetime.utcnow()
    while len(dates) < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon=0 … Fri=4
            dates.append(current.strftime("%Y-%m-%d"))
    return dates


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast() -> ForecastResponse:
    """Generate a 10-day gold price forecast using the latest trained model.

    Returns the median point prediction together with 10th/90th percentile
    confidence bands when quantile outputs are available.
    """
    model = _load_latest_model()

    # Prepare the most recent input window
    prices = _load_recent_prices(days=config.INPUT_SEQUENCE_LENGTH + 50)
    if prices.size < config.INPUT_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Insufficient price history ({prices.size} days available, "
                f"{config.INPUT_SEQUENCE_LENGTH} required)."
            ),
        )

    current_price = float(prices[-1])

    try:
        # Try full pipeline input (scaled features + sequences)
        import pandas as pd

        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
            # Take the last INPUT_SEQUENCE_LENGTH rows as the model input
            recent = df.iloc[-config.INPUT_SEQUENCE_LENGTH:]
            feature_cols = [c for c in recent.columns if c != config.TARGET_COLUMN]
            X_input = recent[feature_cols].values.astype(np.float32)

            # Reshape for sequential models: (1, seq_len, n_features)
            X_seq = X_input[np.newaxis, :, :]

            raw_pred = model.predict(X_seq)
        else:
            raw_pred = model.predict(prices[-config.INPUT_SEQUENCE_LENGTH:][np.newaxis, :])

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    raw_pred = np.asarray(raw_pred).squeeze()

    # Handle quantile output (low, median, high) vs point output
    if raw_pred.ndim == 2 and raw_pred.shape[0] == 3:
        preds_low = raw_pred[0].tolist()
        preds_mid = raw_pred[1].tolist()
        preds_high = raw_pred[2].tolist()
    elif raw_pred.ndim == 2 and raw_pred.shape[1] == 3:
        preds_low = raw_pred[:, 0].tolist()
        preds_mid = raw_pred[:, 1].tolist()
        preds_high = raw_pred[:, 2].tolist()
    else:
        # Point predictions — generate synthetic confidence bands
        preds_mid = raw_pred.ravel().tolist()
        spread = np.std(prices[-60:]) * 0.5
        preds_low = (raw_pred.ravel() - spread).tolist()
        preds_high = (raw_pred.ravel() + spread).tolist()

    horizon = config.OUTPUT_SEQUENCE_LENGTH
    dates = _generate_forecast_dates(horizon)

    # Trim / pad to horizon length
    preds_mid = preds_mid[:horizon]
    preds_low = preds_low[:horizon]
    preds_high = preds_high[:horizon]

    return ForecastResponse(
        dates=dates,
        predictions=preds_mid,
        confidence_low=preds_low,
        confidence_high=preds_high,
        current_price=current_price,
    )


@router.get("/historical", response_model=HistoricalResponse)
async def get_historical() -> HistoricalResponse:
    """Return the last 365 days of actual gold prices (and past predictions if available)."""
    try:
        import pandas as pd

        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
        elif config.RAW_DATA_FILE.exists():
            df = pd.read_parquet(config.RAW_DATA_FILE)
        else:
            raise HTTPException(status_code=503, detail="No historical data available.")

        col = config.TARGET_COLUMN
        if col not in df.columns:
            raise HTTPException(
                status_code=503,
                detail=f"Column '{col}' not found in data."
            )

        recent = df[[col]].dropna().tail(365)

        # Build date list from index or range
        if hasattr(recent.index, "strftime"):
            dates = recent.index.strftime("%Y-%m-%d").tolist()
        else:
            base = datetime.utcnow() - timedelta(days=len(recent))
            dates = [
                (base + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(len(recent))
            ]

        points = [
            HistoricalPoint(date=d, actual=float(v))
            for d, v in zip(dates, recent[col].values)
        ]

        return HistoricalResponse(data=points, count=len(points))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Historical data retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
