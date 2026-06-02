"""
Gold Price Forecasting System — Evaluation Metrics

Provides functions for computing regression metrics, directional accuracy,
per-horizon breakdowns, and pretty-printed comparison tables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values, shape ``(n,)`` or ``(n, horizon)``.
    y_pred : np.ndarray
        Predicted values, same shape as *y_true*.

    Returns
    -------
    float
        RMSE value.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Entries where ``y_true == 0`` are excluded to avoid division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAPE as a percentage (e.g. 2.5 means 2.5 %).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        logger.warning("All y_true values are zero — MAPE undefined; returning NaN.")
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of times the predicted *direction* of price change is correct.

    Direction is computed as the sign of consecutive differences along the first
    axis (``diff`` along axis 0 for 2-D inputs, plain ``diff`` for 1-D inputs).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values, shape ``(n,)`` or ``(n, horizon)``.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Directional accuracy as a percentage (0–100).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if y_true.ndim == 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
    else:
        # Multi-step: direction between consecutive forecast days
        true_dir = np.sign(np.diff(y_true, axis=1))
        pred_dir = np.sign(np.diff(y_pred, axis=1))

    if true_dir.size == 0:
        logger.warning("Not enough data points for directional accuracy.")
        return float("nan")

    return float(np.mean(true_dir == pred_dir) * 100.0)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R² value (can be negative for very poor models).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        logger.warning("y_true has zero variance — R² undefined; returning NaN.")
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------- #
# Aggregate helpers
# --------------------------------------------------------------------------- #


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute every scalar metric in one call.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        ``{"rmse": …, "mae": …, "mape": …, "directional_accuracy": …, "r2": …}``
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


def compute_per_horizon_metrics(
    y_true_2d: np.ndarray, y_pred_2d: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute metrics individually for each forecast day (1 to *horizon*).

    Parameters
    ----------
    y_true_2d : np.ndarray
        Shape ``(n_samples, horizon)`` — ground-truth values.
    y_pred_2d : np.ndarray
        Shape ``(n_samples, horizon)`` — predicted values.

    Returns
    -------
    dict
        Mapping ``{"day_1": {metric_dict}, …, "day_<horizon>": {metric_dict}}``.
    """
    y_true_2d = np.asarray(y_true_2d, dtype=np.float64)
    y_pred_2d = np.asarray(y_pred_2d, dtype=np.float64)
    if y_true_2d.ndim != 2 or y_pred_2d.ndim != 2:
        raise ValueError("Inputs must be 2-D arrays with shape (n_samples, horizon).")
    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_2d.shape} vs y_pred {y_pred_2d.shape}."
        )

    horizon = y_true_2d.shape[1]
    results: Dict[str, Dict[str, float]] = {}
    for day_idx in range(horizon):
        day_label = f"day_{day_idx + 1}"
        results[day_label] = compute_all_metrics(
            y_true_2d[:, day_idx], y_pred_2d[:, day_idx]
        )
        logger.debug("Horizon %s: %s", day_label, results[day_label])

    return results


# --------------------------------------------------------------------------- #
# Pretty-print
# --------------------------------------------------------------------------- #


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """Render a comparison table of model metrics as a formatted string.

    Parameters
    ----------
    metrics_dict : dict
        Mapping ``{model_name: {metric_name: value, …}, …}``.

    Returns
    -------
    str
        Human-readable table string suitable for logging or console output.

    Example
    -------
    >>> table = format_metrics_table({
    ...     "Transformer": {"rmse": 12.3, "mae": 9.1, "mape": 1.5, "r2": 0.97},
    ...     "LSTM":        {"rmse": 14.0, "mae": 10.2, "mape": 1.8, "r2": 0.95},
    ... })
    >>> print(table)
    """
    if not metrics_dict:
        return "(no metrics to display)"

    # Collect all unique metric names across models
    all_metric_names: List[str] = []
    for m in metrics_dict.values():
        for k in m:
            if k not in all_metric_names:
                all_metric_names.append(k)

    # Column widths
    model_col_width = max(len(str(k)) for k in metrics_dict) + 2
    metric_col_width = 14

    # Header
    header = f"{'Model':<{model_col_width}}"
    for mname in all_metric_names:
        header += f"{mname:>{metric_col_width}}"
    separator = "─" * len(header)

    rows = [separator, header, separator]
    for model_name, mvals in metrics_dict.items():
        row = f"{model_name:<{model_col_width}}"
        for mname in all_metric_names:
            val = mvals.get(mname, float("nan"))
            row += f"{val:>{metric_col_width}.4f}"
        rows.append(row)
    rows.append(separator)

    table_str = "\n".join(rows)
    logger.info("Metrics table:\n%s", table_str)
    return table_str


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _validate_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cast to float64 and verify shapes match."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch after flattening: y_true {y_true.shape} "
            f"vs y_pred {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Empty arrays — cannot compute metrics.")
    return y_true, y_pred
