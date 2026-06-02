"""
Gold Price Forecasting System — Data Ingestion Module

Downloads OHLCV data for every ticker defined in ``config.TICKERS`` via the
*yfinance* library, merges them into a single wide-format DataFrame with
prefixed column names (e.g. ``gold_Open``, ``sp500_Close``), forward-fills
missing values, and persists the result as a Parquet file at
``config.RAW_DATA_FILE``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from app import config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_SECONDS: float = 5.0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _download_ticker(
    ticker_symbol: str,
    start: str,
    end: str,
    retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with retry logic.

    Parameters
    ----------
    ticker_symbol:
        Yahoo Finance ticker string (e.g. ``"GC=F"``).
    start:
        Start date in ``YYYY-MM-DD`` format.
    end:
        End date in ``YYYY-MM-DD`` format.
    retries:
        Maximum number of retry attempts on failure.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Date`` with columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']``.

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d) ...",
                ticker_symbol,
                attempt,
                retries,
            )
            df: pd.DataFrame = yf.download(
                ticker_symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                raise ValueError(
                    f"yfinance returned an empty DataFrame for {ticker_symbol}"
                )

            # Flatten MultiIndex columns if yfinance returns them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only the standard OHLCV columns that are present
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            available = [c for c in ohlcv_cols if c in df.columns]
            df = df[available].copy()

            logger.info(
                "Downloaded %d rows for %s (%s → %s)",
                len(df),
                ticker_symbol,
                df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d"),
            )
            return df

        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            logger.warning(
                "Attempt %d/%d for %s failed: %s",
                attempt,
                retries,
                ticker_symbol,
                exc,
            )
            if attempt < retries:
                wait = _RETRY_BACKOFF_SECONDS * attempt
                logger.info("Retrying in %.1f seconds …", wait)
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {ticker_symbol} after {retries} attempts"
    ) from last_exception


def _merge_dataframes(
    ticker_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-ticker DataFrames into a single wide-format DataFrame.

    Each column is prefixed with its ticker key, e.g. ``gold_Open``,
    ``sp500_Close``.

    Parameters
    ----------
    ticker_frames:
        Mapping from ticker key (``"gold"``, ``"sp500"``, …) to OHLCV
        DataFrame.

    Returns
    -------
    pd.DataFrame
        A single DataFrame indexed by ``Date`` with prefixed columns.
    """
    renamed_frames: list[pd.DataFrame] = []

    for key, df in ticker_frames.items():
        renamed = df.rename(columns={col: f"{key}_{col}" for col in df.columns})
        renamed_frames.append(renamed)

    merged = pd.concat(renamed_frames, axis=1)
    merged.sort_index(inplace=True)
    return merged


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def download_all_data(
    tickers: Optional[Dict[str, str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download OHLCV data for **all** configured tickers and save as Parquet.

    Parameters
    ----------
    tickers:
        Override for ``config.TICKERS``.  Mapping from human-readable key
        to Yahoo Finance symbol.
    start_date:
        Override for ``config.DATA_START_DATE``.
    end_date:
        Override for ``config.DATA_END_DATE``.
    output_path:
        Override for ``config.RAW_DATA_FILE``.

    Returns
    -------
    pd.DataFrame
        The merged, forward-filled DataFrame that was written to disk.
    """
    tickers = tickers or config.TICKERS
    start_date = start_date or config.DATA_START_DATE
    end_date = end_date or config.DATA_END_DATE
    output_path = output_path or config.RAW_DATA_FILE

    logger.info(
        "Starting data ingestion for %d tickers (%s → %s)",
        len(tickers),
        start_date,
        end_date,
    )

    ticker_frames: Dict[str, pd.DataFrame] = {}
    for key, symbol in tickers.items():
        ticker_frames[key] = _download_ticker(symbol, start_date, end_date)

    # Merge into a single wide DataFrame
    merged = _merge_dataframes(ticker_frames)

    # Forward-fill then backward-fill residual leading NaNs
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)

    logger.info(
        "Merged DataFrame shape: %s  |  Date range: %s → %s",
        merged.shape,
        merged.index.min().strftime("%Y-%m-%d"),
        merged.index.max().strftime("%Y-%m-%d"),
    )

    # Persist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, engine="pyarrow")
    logger.info("Raw data saved to %s", output_path)

    return merged


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the previously-saved raw data Parquet file.

    Parameters
    ----------
    path:
        Override for ``config.RAW_DATA_FILE``.

    Returns
    -------
    pd.DataFrame
        The raw merged DataFrame.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist.  Run
        :func:`download_all_data` first.
    """
    path = Path(path or config.RAW_DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Run download_all_data() first."
        )

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded raw data: shape=%s from %s", df.shape, path)
    return df
