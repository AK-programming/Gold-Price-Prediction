"""
Gold Price Forecasting System — Preprocessor Module

Handles every step between *engineered features* and *model-ready tensors*:

* Drop residual NaN rows (after forward-fill).
* Identify feature columns (everything except target and date index).
* Fit a ``MinMaxScaler`` on the training split; persist via *joblib*.
* Chronological train / val / test split per ``config`` ratios.
* Sliding-window sequence creation for deep-learning models.
* Flattened-window creation for tree-based (tabular) models.
* A ``torch.utils.data.Dataset`` wrapper and DataLoader factory.
* Inverse-scaling utility for predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from app import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all columns **except** the target column.

    The DataFrame index is assumed to be a DatetimeIndex and is not
    counted as a column.
    """
    return [c for c in df.columns if c != config.TARGET_COLUMN]


def _chronological_split(
    data: np.ndarray,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split *data* chronologically into train / val / test arrays.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_timesteps, n_features)``.
    train_ratio:
        Fraction of data for training.
    val_ratio:
        Fraction of data for validation.

    Returns
    -------
    tuple of np.ndarray
        ``(train, val, test)`` arrays.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    logger.info(
        "Chronological split — train: %d, val: %d, test: %d (total: %d)",
        len(train),
        len(val),
        len(test),
        n,
    )
    return train, val, test


# ------------------------------------------------------------------
# Public: preprocessing
# ------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    scaler_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    List[str], int, MinMaxScaler,
]:
    """Clean, scale, and split a feature-engineered DataFrame.

    Parameters
    ----------
    df:
        Output of :func:`feature_engineering.engineer_features` or
        :func:`feature_engineering.load_features`.
    scaler_path:
        Where to save the fitted ``MinMaxScaler`` via *joblib*.
        Defaults to ``config.SCALER_FILE``.
    output_path:
        Where to save the processed DataFrame as Parquet.
        Defaults to ``config.PROCESSED_FILE``.

    Returns
    -------
    train : np.ndarray
        Training split (scaled).
    val : np.ndarray
        Validation split (scaled).
    test : np.ndarray
        Test split (scaled).
    feature_cols : list[str]
        Ordered list of feature column names (including target).
    target_idx : int
        Index of the target column in *feature_cols* (and in the arrays).
    scaler : MinMaxScaler
        The fitted scaler (also saved to disk).
    """
    scaler_path = Path(scaler_path or config.SCALER_FILE)
    output_path = Path(output_path or config.PROCESSED_FILE)

    df = df.copy()

    # 1. Forward-fill then drop any remaining NaN rows
    df.ffill(inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    if rows_before != rows_after:
        logger.info(
            "Dropped %d NaN rows (%d → %d)",
            rows_before - rows_after,
            rows_before,
            rows_after,
        )

    # 2. Identify feature columns (all columns — target is kept in the array
    #    so sequences can include it as a feature + label).
    all_cols: List[str] = list(df.columns)
    if config.TARGET_COLUMN not in all_cols:
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' not found in DataFrame. "
            f"Available columns: {all_cols[:20]} …"
        )

    # Move target to last position for easy slicing later
    all_cols.remove(config.TARGET_COLUMN)
    all_cols.append(config.TARGET_COLUMN)
    df = df[all_cols]

    feature_cols = all_cols  # includes target as last element
    target_idx = len(feature_cols) - 1
    logger.info(
        "Feature columns: %d (target '%s' at index %d)",
        len(feature_cols),
        config.TARGET_COLUMN,
        target_idx,
    )

    # 3. Chronological split (on raw values, so scaler is fit only on train)
    values = df.values.astype(np.float64)
    train_raw, val_raw, test_raw = _chronological_split(values)

    # 4. Fit MinMaxScaler on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # 5. Persist scaler
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    # 6. Persist processed DataFrame (unscaled, for reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Processed DataFrame saved to %s", output_path)

    return train_scaled, val_scaled, test_scaled, feature_cols, target_idx, scaler


# ------------------------------------------------------------------
# Public: sequence creation
# ------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    input_len: int = config.INPUT_SEQUENCE_LENGTH,
    output_len: int = config.OUTPUT_SEQUENCE_LENGTH,
    stride: int = config.STRIDE,
    target_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) pairs for deep-learning models.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_timesteps, n_features)``, **already scaled**.
    input_len:
        Number of past time-steps in each input window.
    output_len:
        Number of future time-steps to predict.
    stride:
        Step size between consecutive windows.
    target_idx:
        Column index of the target variable.  Defaults to the last column
        (``-1``).

    Returns
    -------
    X : np.ndarray
        Shape ``(n_samples, input_len, n_features)``.
    y : np.ndarray
        Shape ``(n_samples, output_len)``.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    total_window = input_len + output_len
    n = len(data)

    for start in range(0, n - total_window + 1, stride):
        x_window = data[start : start + input_len]
        y_window = data[start + input_len : start + total_window, target_idx]
        X_list.append(x_window)
        y_list.append(y_window)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(
        "Created sequences — X: %s, y: %s (input=%d, output=%d, stride=%d)",
        X.shape,
        y.shape,
        input_len,
        output_len,
        stride,
    )
    return X, y


def create_tabular_dataset(
    data: np.ndarray,
    input_len: int = config.INPUT_SEQUENCE_LENGTH,
    target_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a flattened tabular dataset for tree-based models.

    Each sample is constructed by flattening an ``input_len``-step window
    into a single feature row.  The target ``y`` is the value at the very
    next time-step (single-step forecast).

    Parameters
    ----------
    data:
        2-D array ``(n_timesteps, n_features)``, **already scaled**.
    input_len:
        Number of past time-steps to flatten into one row.
    target_idx:
        Column index of the target variable.

    Returns
    -------
    X_flat : np.ndarray
        Shape ``(n_samples, input_len * n_features)``.
    y : np.ndarray
        Shape ``(n_samples,)``.
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    n = len(data)

    for start in range(0, n - input_len, 1):
        x_window = data[start : start + input_len].flatten()
        y_val = data[start + input_len, target_idx]
        X_list.append(x_window)
        y_list.append(y_val)

    X_flat = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(
        "Created tabular dataset — X_flat: %s, y: %s (input_len=%d)",
        X_flat.shape,
        y.shape,
        input_len,
    )
    return X_flat, y


# ------------------------------------------------------------------
# Public: PyTorch Dataset & DataLoaders
# ------------------------------------------------------------------

class GoldDataset(Dataset):
    """PyTorch :class:`Dataset` wrapping (X, y) numpy arrays.

    Parameters
    ----------
    X:
        Input tensor of shape ``(n_samples, input_len, n_features)``.
    y:
        Target tensor of shape ``(n_samples, output_len)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    batch_size: int = config.TRANSFORMER_CONFIG["batch_size"],
    input_len: int = config.INPUT_SEQUENCE_LENGTH,
    output_len: int = config.OUTPUT_SEQUENCE_LENGTH,
    stride: int = config.STRIDE,
    target_idx: int = -1,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build PyTorch DataLoaders for train / val / test splits.

    Parameters
    ----------
    train_data, val_data, test_data:
        Scaled numpy arrays from :func:`preprocess_data`.
    batch_size:
        Mini-batch size.
    input_len:
        Input window length.
    output_len:
        Output (forecast) window length.
    stride:
        Sliding-window stride.
    target_idx:
        Column index of the target.
    num_workers:
        ``DataLoader`` worker processes.

    Returns
    -------
    dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, ``"test"``.
    """
    loaders: Dict[str, DataLoader] = {}

    for name, data, shuffle in [
        ("train", train_data, True),
        ("val", val_data, False),
        ("test", test_data, False),
    ]:
        X, y = create_sequences(
            data,
            input_len=input_len,
            output_len=output_len,
            stride=stride,
            target_idx=target_idx,
        )
        ds = GoldDataset(X, y)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        logger.info(
            "DataLoader '%s': %d batches (batch_size=%d, samples=%d)",
            name,
            len(loaders[name]),
            batch_size,
            len(ds),
        )

    return loaders


# ------------------------------------------------------------------
# Public: inverse transform
# ------------------------------------------------------------------

def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: MinMaxScaler,
    target_idx: int = -1,
    n_features: Optional[int] = None,
) -> np.ndarray:
    """Reverse the MinMax scaling on model predictions.

    Because the scaler was fit on **all** features jointly, we need to
    embed the predictions into a full-width dummy array, inverse-transform,
    then extract the target column.

    Parameters
    ----------
    predictions:
        1-D or 2-D array of scaled predictions.  If 2-D, shape should be
        ``(n_samples, output_len)``.
    scaler:
        The fitted ``MinMaxScaler`` from :func:`preprocess_data`.
    target_idx:
        Index of the target column in the scaler's feature set.
    n_features:
        Total number of features the scaler was fit on.  If ``None``,
        inferred from ``scaler.n_features_in_``.

    Returns
    -------
    np.ndarray
        Predictions in original (unscaled) price space, same shape as
        *predictions*.
    """
    original_shape = predictions.shape
    flat = predictions.flatten()

    n_feat = n_features or scaler.n_features_in_

    # Build a dummy array of zeros with the correct width
    dummy = np.zeros((len(flat), n_feat), dtype=np.float64)
    dummy[:, target_idx] = flat

    inversed = scaler.inverse_transform(dummy)[:, target_idx]
    return inversed.reshape(original_shape).astype(np.float32)
