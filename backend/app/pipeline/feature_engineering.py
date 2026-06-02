"""
Gold Price Forecasting System — Feature Engineering Module

Transforms raw merged OHLCV data into 50+ modelling features:

* **Technical indicators** (``ta`` library): SMA, EMA, RSI, MACD,
  Bollinger Bands, ATR.
* **Price ratios**: gold/silver, gold/oil, gold/S&P 500.
* **Rolling statistics**: volatility (std of returns), rolling mean / min /
  max of returns.
* **Log returns** for every asset.
* **Lagged values** using ``config.LAG_PERIODS``.
* **Calendar features**: day_of_week, month, quarter, is_month_start,
  is_month_end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import ta

from app import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Technical-indicator helpers
# ------------------------------------------------------------------

def _add_sma(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """Append Simple Moving Averages for *column* at every window size."""
    for w in windows:
        col_name = f"{column}_sma_{w}"
        df[col_name] = df[column].rolling(window=w, min_periods=1).mean()
    return df


def _add_ema(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """Append Exponential Moving Averages for *column* at every window size."""
    for w in windows:
        col_name = f"{column}_ema_{w}"
        df[col_name] = df[column].ewm(span=w, adjust=False, min_periods=1).mean()
    return df


def _add_rsi(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Append RSI computed from *column*."""
    col_name = f"{column}_rsi_{period}"
    df[col_name] = ta.momentum.RSIIndicator(
        close=df[column], window=period
    ).rsi()
    return df


def _add_macd(
    df: pd.DataFrame,
    column: str,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """Append MACD line, signal line, and histogram."""
    macd_ind = ta.trend.MACD(
        close=df[column],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    )
    df[f"{column}_macd"] = macd_ind.macd()
    df[f"{column}_macd_signal"] = macd_ind.macd_signal()
    df[f"{column}_macd_hist"] = macd_ind.macd_diff()
    return df


def _add_bollinger_bands(
    df: pd.DataFrame,
    column: str,
    window: int,
    std_dev: int,
) -> pd.DataFrame:
    """Append Bollinger Bands (high, mid, low, bandwidth, %b)."""
    bb = ta.volatility.BollingerBands(
        close=df[column], window=window, window_dev=std_dev
    )
    df[f"{column}_bb_high"] = bb.bollinger_hband()
    df[f"{column}_bb_mid"] = bb.bollinger_mavg()
    df[f"{column}_bb_low"] = bb.bollinger_lband()
    df[f"{column}_bb_bandwidth"] = bb.bollinger_wband()
    df[f"{column}_bb_pctb"] = bb.bollinger_pband()
    return df


def _add_atr(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    period: int,
    prefix: str,
) -> pd.DataFrame:
    """Append Average True Range."""
    atr = ta.volatility.AverageTrueRange(
        high=df[high_col],
        low=df[low_col],
        close=df[close_col],
        window=period,
    )
    df[f"{prefix}_atr_{period}"] = atr.average_true_range()
    return df


# ------------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------------

def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for the gold (target) series."""
    gold_close = "gold_Close"

    # SMA
    df = _add_sma(df, gold_close, config.SMA_WINDOWS)

    # EMA
    df = _add_ema(df, gold_close, config.EMA_WINDOWS)

    # RSI
    df = _add_rsi(df, gold_close, config.RSI_PERIOD)

    # MACD
    df = _add_macd(
        df,
        gold_close,
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )

    # Bollinger Bands
    df = _add_bollinger_bands(
        df,
        gold_close,
        window=config.BOLLINGER_WINDOW,
        std_dev=config.BOLLINGER_STD,
    )

    # ATR (requires High, Low, Close)
    df = _add_atr(
        df,
        high_col="gold_High",
        low_col="gold_Low",
        close_col=gold_close,
        period=config.ATR_PERIOD,
        prefix="gold",
    )

    return df


def _add_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add inter-asset price ratios."""
    # Guard against division by zero with small epsilon
    eps = 1e-10

    if "silver_Close" in df.columns:
        df["ratio_gold_silver"] = df["gold_Close"] / (df["silver_Close"] + eps)

    if "crude_oil_Close" in df.columns:
        df["ratio_gold_oil"] = df["gold_Close"] / (df["crude_oil_Close"] + eps)

    if "sp500_Close" in df.columns:
        df["ratio_gold_sp500"] = df["gold_Close"] / (df["sp500_Close"] + eps)

    if "dollar_index_Close" in df.columns:
        df["ratio_gold_dollar"] = df["gold_Close"] / (df["dollar_index_Close"] + eps)

    if "bitcoin_Close" in df.columns:
        df["ratio_gold_bitcoin"] = df["gold_Close"] / (df["bitcoin_Close"] + eps)

    return df


def _add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns for every asset's Close price."""
    close_cols = [c for c in df.columns if c.endswith("_Close")]
    for col in close_cols:
        asset = col.replace("_Close", "")
        df[f"{asset}_log_return"] = np.log(df[col] / df[col].shift(1).replace(0, np.nan))
    return df


def _add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility, mean, min, max of gold log returns."""
    return_col = "gold_log_return"
    if return_col not in df.columns:
        return df

    for w in config.ROLLING_WINDOWS:
        df[f"gold_return_vol_{w}"] = df[return_col].rolling(window=w, min_periods=1).std()
        df[f"gold_return_mean_{w}"] = df[return_col].rolling(window=w, min_periods=1).mean()
        df[f"gold_return_min_{w}"] = df[return_col].rolling(window=w, min_periods=1).min()
        df[f"gold_return_max_{w}"] = df[return_col].rolling(window=w, min_periods=1).max()

    # Dedicated volatility at VOLATILITY_WINDOW
    vw = config.VOLATILITY_WINDOW
    df[f"gold_volatility_{vw}"] = df[return_col].rolling(window=vw, min_periods=1).std()

    return df


def _add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged gold Close and gold log-return values."""
    for lag in config.LAG_PERIODS:
        df[f"gold_Close_lag_{lag}"] = df["gold_Close"].shift(lag)

        if "gold_log_return" in df.columns:
            df[f"gold_log_return_lag_{lag}"] = df["gold_log_return"].shift(lag)

    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived calendar features."""
    idx = pd.DatetimeIndex(df.index)

    df["day_of_week"] = idx.dayofweek.astype(np.float32)
    df["month"] = idx.month.astype(np.float32)
    df["quarter"] = idx.quarter.astype(np.float32)
    df["is_month_start"] = idx.is_month_start.astype(np.float32)
    df["is_month_end"] = idx.is_month_end.astype(np.float32)
    df["day_of_year"] = idx.dayofyear.astype(np.float32)
    df["week_of_year"] = idx.isocalendar().week.values.astype(np.float32)

    return df


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Apply all feature engineering steps to the raw DataFrame.

    Parameters
    ----------
    df:
        Raw merged OHLCV DataFrame (output of
        :func:`data_ingestion.download_all_data` or
        :func:`data_ingestion.load_raw_data`).
    output_path:
        Override for ``config.FEATURES_FILE``.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with 50+ features, saved to disk.
    """
    output_path = Path(output_path or config.FEATURES_FILE)

    logger.info("Starting feature engineering on DataFrame of shape %s", df.shape)
    df = df.copy()

    # 1. Technical indicators on gold
    df = _add_technical_indicators(df)
    logger.info("Added technical indicators — columns: %d", len(df.columns))

    # 2. Price ratios
    df = _add_price_ratios(df)
    logger.info("Added price ratios — columns: %d", len(df.columns))

    # 3. Log returns (must come before rolling stats & lags)
    df = _add_log_returns(df)
    logger.info("Added log returns — columns: %d", len(df.columns))

    # 4. Rolling statistics on gold returns
    df = _add_rolling_statistics(df)
    logger.info("Added rolling statistics — columns: %d", len(df.columns))

    # 5. Lagged features
    df = _add_lagged_features(df)
    logger.info("Added lagged features — columns: %d", len(df.columns))

    # 6. Calendar features
    df = _add_calendar_features(df)
    logger.info("Added calendar features — columns: %d", len(df.columns))

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info(
        "Feature-engineered data saved to %s (shape: %s, %d features)",
        output_path,
        df.shape,
        len(df.columns),
    )

    return df


def load_features(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the previously-saved feature-engineered Parquet file.

    Parameters
    ----------
    path:
        Override for ``config.FEATURES_FILE``.

    Returns
    -------
    pd.DataFrame
        The enriched DataFrame.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.  Run
        :func:`engineer_features` first.
    """
    path = Path(path or config.FEATURES_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found at {path}. "
            "Run engineer_features() first."
        )

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded features: shape=%s from %s", df.shape, path)
    return df
