"""
Market data status and refresh routes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix=config.API_PREFIX, tags=["data"])


class DataStatusResponse(BaseModel):
    last_data_date: str = Field(..., description="Last trading day in features file (ISO)")
    last_close_usd: float = Field(..., description="Last gold close in USD/oz")
    ticker: str = Field(default="GC=F")
    source_label: str = Field(
        default="COMEX Gold Futures via Yahoo Finance (USD per troy oz)"
    )
    days_behind: int = Field(..., description="Calendar days since last data date")
    live_quote_usd: float | None = Field(
        None, description="Latest available GC=F close from Yahoo (may differ slightly)"
    )
    live_quote_date: str | None = None
    features_ready: bool = True


class DataRefreshResponse(BaseModel):
    status: str = "ok"
    message: str
    last_data_date: str
    last_close_usd: float
    rows_downloaded: int


def _read_dataset_status() -> tuple[str, float]:
    import pandas as pd

    if not config.FEATURES_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Feature data not found. Run training or refresh data first.",
        )

    df = pd.read_parquet(config.FEATURES_FILE)
    if config.TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=503,
            detail=f"Column '{config.TARGET_COLUMN}' missing from features.",
        )

    series = df[config.TARGET_COLUMN].dropna()
    if series.empty:
        raise HTTPException(status_code=503, detail="No gold price rows in features file.")

    last_ts = series.index[-1]
    if hasattr(last_ts, "strftime"):
        last_date = last_ts.strftime("%Y-%m-%d")
    else:
        last_date = str(last_ts)[:10]

    return last_date, float(series.iloc[-1])


def _fetch_live_quote() -> tuple[float | None, str | None]:
    try:
        import pandas as pd
        import yfinance as yf

        end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        df = yf.download(
            config.TICKERS["gold"],
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None, None

        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna()
        last_ts = close.index[-1]
        last_date = (
            last_ts.strftime("%Y-%m-%d")
            if hasattr(last_ts, "strftime")
            else str(last_ts)[:10]
        )
        return float(close.iloc[-1]), last_date
    except Exception as exc:
        logger.warning("Live quote fetch failed: %s", exc)
        return None, None


@router.get("/data/status", response_model=DataStatusResponse)
async def data_status() -> DataStatusResponse:
    """Return freshness of stored gold data and an optional live Yahoo quote."""
    last_date, last_close = _read_dataset_status()
    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    days_behind = max(0, (today - last_dt).days)

    live_price, live_date = _fetch_live_quote()

    return DataStatusResponse(
        last_data_date=last_date,
        last_close_usd=round(last_close, 2),
        ticker=config.TICKERS["gold"],
        days_behind=days_behind,
        live_quote_usd=round(live_price, 2) if live_price is not None else None,
        live_quote_date=live_date,
        features_ready=config.FEATURES_FILE.exists(),
    )


@router.post("/data/refresh", response_model=DataRefreshResponse)
async def refresh_market_data() -> DataRefreshResponse:
    """Re-download Yahoo data through today and rebuild the features file."""
    try:
        from app.pipeline.data_ingestion import download_all_data
        from app.pipeline.feature_engineering import engineer_features

        end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        logger.info("Refreshing market data through %s", end_date)

        raw_df = download_all_data(end_date=end_date)
        features_df = engineer_features(raw_df)

        last_date, last_close = _read_dataset_status()

        return DataRefreshResponse(
            status="ok",
            message=(
                "Data refreshed. Forecasts use the latest download; "
                "retrain models for best accuracy after large price moves."
            ),
            last_data_date=last_date,
            last_close_usd=round(last_close, 2),
            rows_downloaded=len(features_df),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Data refresh failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data refresh failed: {exc}")
