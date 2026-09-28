"""
Prediction routes for gold price forecasts and historical data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.pipeline.preprocessor import inverse_target_column, prepare_inference_window
from app.utils.model_selection import resolve_serving_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix=config.API_PREFIX, tags=["predictions"])


class ForecastResponse(BaseModel):
    dates: List[str] = Field(..., description="ISO-format forecast dates")
    predictions: List[float] = Field(..., description="Point predictions")
    confidence_low: List[float] = Field(..., description="Lower confidence bound")
    confidence_high: List[float] = Field(..., description="Upper confidence bound")
    current_price: float = Field(..., description="Most recent actual gold price")
    model_used: str = Field(..., description="Model that produced the forecast")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HistoricalPoint(BaseModel):
    date: str
    actual: float
    predicted: Optional[float] = None


class HistoricalResponse(BaseModel):
    data: List[HistoricalPoint]
    count: int


def _load_sequence_model(name: str) -> Any:
    import torch

    from app.models import GoldLSTM, GoldTransformer

    filename = "transformer.pt" if name == "transformer" else "lstm.pt"
    ckpt = config.CHECKPOINT_DIR / filename
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    data = torch.load(ckpt, map_location="cpu")
    num_features = int(data["num_features"])
    model = (
        GoldTransformer(num_features=num_features)
        if name == "transformer"
        else GoldLSTM(num_features=num_features)
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    logger.info("Loaded %s checkpoint from %s", name, ckpt)
    return model


def _load_tabular_model(name: str) -> Any:
    from app.models import CatBoostModel, LightGBMModel, XGBoostModel

    wrappers = {
        "xgboost": (XGBoostModel, "xgboost.joblib"),
        "lightgbm": (LightGBMModel, "lightgbm.joblib"),
        "catboost": (CatBoostModel, "catboost.joblib"),
    }
    if name not in wrappers:
        raise ValueError(f"Unknown tabular model: {name}")

    cls, filename = wrappers[name]
    ckpt = config.CHECKPOINT_DIR / filename
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    model = cls().load(ckpt)
    logger.info("Loaded %s checkpoint from %s", name, ckpt)
    return model


def _load_ensemble_model() -> Any:
    from app.models.ensemble import load_stacking_ensemble

    ensemble = load_stacking_ensemble()
    logger.info("Loaded stacking ensemble")
    return ensemble


def _predict_ensemble_horizon(
    ensemble: Any,
    recent_scaled: np.ndarray,
    horizon: int,
) -> np.ndarray:
    x_seq = recent_scaled[np.newaxis, :, :].astype(np.float32)
    x_tab = recent_scaled.reshape(1, -1).astype(np.float32)
    preds = ensemble.predict(x_seq, x_tab)
    return np.asarray(preds, dtype=np.float32).reshape(-1)[:horizon]


def _load_serving_model() -> tuple[str, str, Any]:
    """Load the best model per metrics, with checkpoint priority fallback."""
    try:
        name, kind = resolve_serving_model()
        if kind == "ensemble":
            return name, kind, _load_ensemble_model()
        if kind == "sequence":
            return name, kind, _load_sequence_model(name)
        return name, kind, _load_tabular_model(name)
    except RuntimeError:
        pass
    except Exception as exc:
        logger.warning("Failed to load selected model: %s", exc)

    for name in ("ensemble", "transformer", "lstm", "xgboost", "lightgbm", "catboost"):
        try:
            if name == "ensemble":
                return name, "ensemble", _load_ensemble_model()
            if name in ("transformer", "lstm"):
                return name, "sequence", _load_sequence_model(name)
            return name, "tabular", _load_tabular_model(name)
        except (FileNotFoundError, Exception) as exc:
            logger.warning("Could not load %s: %s", name, exc)

    raise HTTPException(
        status_code=503,
        detail="No trained model available. Run train_and_setup.py or eval_models.py.",
    )


def _load_recent_prices(days: int = 365) -> np.ndarray:
    try:
        import pandas as pd

        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
        elif config.RAW_DATA_FILE.exists():
            df = pd.read_parquet(config.RAW_DATA_FILE)
        else:
            return np.array([])

        if config.TARGET_COLUMN not in df.columns:
            return np.array([])

        return df[config.TARGET_COLUMN].dropna().values[-days:]
    except Exception as exc:
        logger.error("Failed to load recent prices: %s", exc)
        return np.array([])


def _load_scaled_recent_window() -> tuple[np.ndarray, list[str], int, Any, float]:
    """Load the latest feature window with an inference scaler fit on full history."""
    try:
        import pandas as pd

        if not config.FEATURES_FILE.exists():
            raise HTTPException(status_code=503, detail="Feature data not available.")

        df = pd.read_parquet(config.FEATURES_FILE)
        try:
            return prepare_inference_window(df)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to prepare forecast features: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature preparation error: {exc}")


def _predict_tabular_horizon(
    model: Any,
    recent_scaled: np.ndarray,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    window = recent_scaled.copy()
    preds: list[float] = []

    for _ in range(horizon):
        pred = float(np.asarray(model.predict(window.reshape(1, -1))).ravel()[0])
        preds.append(pred)

        next_row = window[-1].copy()
        next_row[target_idx] = pred
        window = np.vstack([window[1:], next_row])

    return np.asarray(preds, dtype=np.float32)


def _generate_forecast_dates(n_days: int) -> List[str]:
    dates: List[str] = []
    current = datetime.now(timezone.utc)

    while len(dates) < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))

    return dates


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast() -> ForecastResponse:
    model_name, model_kind, model = _load_serving_model()

    prices = _load_recent_prices(days=config.INPUT_SEQUENCE_LENGTH + 50)
    if prices.size < config.INPUT_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Insufficient price history ({prices.size} days available, "
                f"{config.INPUT_SEQUENCE_LENGTH} required)."
            ),
        )

    try:
        recent_scaled, feature_cols, target_idx, scaler, current_price = (
            _load_scaled_recent_window()
        )
        horizon = config.OUTPUT_SEQUENCE_LENGTH

        if model_kind == "ensemble":
            scaled_mid = _predict_ensemble_horizon(
                model, recent_scaled, horizon
            )
            scaled_low = None
            scaled_high = None
        elif model_kind == "sequence":
            import torch

            x_tensor = torch.tensor(
                recent_scaled[np.newaxis, :, :],
                dtype=torch.float32,
            )
            with torch.no_grad():
                output = model(x_tensor)

            if isinstance(output, dict):
                scaled_low = output.get("q10")
                scaled_mid = output.get("q50")
                scaled_high = output.get("q90")
                if scaled_low is None or scaled_mid is None or scaled_high is None:
                    raise ValueError("Transformer output is missing q10/q50/q90.")
                scaled_low = scaled_low.cpu().numpy().squeeze()
                scaled_mid = scaled_mid.cpu().numpy().squeeze()
                scaled_high = scaled_high.cpu().numpy().squeeze()
            else:
                scaled_mid = output.cpu().numpy().squeeze()
                scaled_low = None
                scaled_high = None
        else:
            scaled_mid = _predict_tabular_horizon(
                model,
                recent_scaled,
                target_idx,
                horizon,
            )
            scaled_low = None
            scaled_high = None
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    preds_mid_arr = inverse_target_column(
        np.asarray(scaled_mid).ravel(),
        scaler,
        target_idx,
    )

    if scaled_low is not None and scaled_high is not None:
        preds_low_arr = inverse_target_column(
            np.asarray(scaled_low).ravel(),
            scaler,
            target_idx,
        )
        preds_high_arr = inverse_target_column(
            np.asarray(scaled_high).ravel(),
            scaler,
            target_idx,
        )
    else:
        spread = float(np.std(prices[-60:]) * 0.5)
        preds_low_arr = preds_mid_arr - spread
        preds_high_arr = preds_mid_arr + spread

    horizon = config.OUTPUT_SEQUENCE_LENGTH

    return ForecastResponse(
        dates=_generate_forecast_dates(horizon),
        predictions=preds_mid_arr[:horizon].tolist(),
        confidence_low=preds_low_arr[:horizon].tolist(),
        confidence_high=preds_high_arr[:horizon].tolist(),
        current_price=current_price,
        model_used=model_name,
    )


@router.get("/historical", response_model=HistoricalResponse)
async def get_historical() -> HistoricalResponse:
    try:
        import pandas as pd

        if config.FEATURES_FILE.exists():
            df = pd.read_parquet(config.FEATURES_FILE)
        elif config.RAW_DATA_FILE.exists():
            df = pd.read_parquet(config.RAW_DATA_FILE)
        else:
            raise HTTPException(status_code=503, detail="No historical data available.")

        if config.TARGET_COLUMN not in df.columns:
            raise HTTPException(
                status_code=503,
                detail=f"Column '{config.TARGET_COLUMN}' not found in data.",
            )

        recent = df[[config.TARGET_COLUMN]].dropna().tail(365)
        if hasattr(recent.index, "strftime"):
            dates = recent.index.strftime("%Y-%m-%d").tolist()
        else:
            base = datetime.now(timezone.utc) - timedelta(days=len(recent))
            dates = [
                (base + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(len(recent))
            ]

        points = [
            HistoricalPoint(date=date, actual=float(value))
            for date, value in zip(dates, recent[config.TARGET_COLUMN].values)
        ]

        return HistoricalResponse(data=points, count=len(points))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Historical data retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
