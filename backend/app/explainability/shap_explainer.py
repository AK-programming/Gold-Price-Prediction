"""
Gold Price Forecasting System — SHAP Explainability

Wraps ``shap.TreeExplainer`` and ``shap.DeepExplainer`` to produce
feature-importance explanations for tree-based and deep-learning models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app import config

logger = logging.getLogger(__name__)


class ShapExplainer:
    """Unified SHAP explainer for tree and deep-learning models.

    Attributes
    ----------
    shap_values : Optional[np.ndarray]
        Raw SHAP values from the most recent ``explain_*`` call.
    feature_names : Optional[List[str]]
        Feature column names aligned with the SHAP value columns.
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = feature_names
        self._mean_abs_shap: Optional[np.ndarray] = None
        logger.info("ShapExplainer initialised (features=%s).",
                     len(feature_names) if feature_names else "unknown")

    # ------------------------------------------------------------------ #
    # Tree models (XGBoost, LightGBM, CatBoost)
    # ------------------------------------------------------------------ #

    def explain_tree_model(
        self, model: Any, X_sample: np.ndarray
    ) -> Dict[str, float]:
        """Compute SHAP values for a tree-based model.

        Parameters
        ----------
        model : Any
            A fitted tree model (XGBRegressor, LGBMRegressor, CatBoostRegressor).
        X_sample : np.ndarray
            Input data, shape ``(n_samples, n_features)``.

        Returns
        -------
        dict
            ``{feature_name: mean_abs_shap_value, …}`` sorted descending.
        """
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)

            # shap_values may be a list (multi-output) — take first output
            if isinstance(sv, list):
                sv = sv[0]
            self.shap_values = np.asarray(sv, dtype=np.float64)
            self._mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

            importance = self._build_importance_dict()
            logger.info("Tree SHAP computed — top feature: %s",
                        next(iter(importance), "N/A"))
            return importance

        except Exception as exc:
            logger.error("SHAP tree explanation failed: %s", exc, exc_info=True)
            return {}

    # ------------------------------------------------------------------ #
    # Deep models (Transformer / LSTM via PyTorch)
    # ------------------------------------------------------------------ #

    def explain_deep_model(
        self,
        model: Any,
        X_sample: np.ndarray,
        background: np.ndarray,
    ) -> Dict[str, float]:
        """Compute SHAP values for a deep-learning model.

        Parameters
        ----------
        model : Any
            A PyTorch ``nn.Module`` in eval mode.
        X_sample : np.ndarray
            Input data, shape ``(n_samples, seq_len, n_features)``.
        background : np.ndarray
            Background reference data, shape ``(n_background, seq_len, n_features)``.

        Returns
        -------
        dict
            ``{feature_name: mean_abs_shap_value, …}`` sorted descending.
        """
        try:
            import shap
            import torch

            model.eval()

            bg_tensor = torch.tensor(background, dtype=torch.float32)
            sample_tensor = torch.tensor(X_sample, dtype=torch.float32)

            explainer = shap.DeepExplainer(model, bg_tensor)
            sv = explainer.shap_values(sample_tensor)

            if isinstance(sv, list):
                sv = sv[0]
            sv_np = np.asarray(sv, dtype=np.float64)

            # Average over the time-step axis → (n_samples, n_features)
            if sv_np.ndim == 3:
                sv_np = np.mean(sv_np, axis=1)

            self.shap_values = sv_np
            self._mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

            importance = self._build_importance_dict()
            logger.info("Deep SHAP computed — top feature: %s",
                        next(iter(importance), "N/A"))
            return importance

        except Exception as exc:
            logger.error("SHAP deep explanation failed: %s", exc, exc_info=True)
            return {}

    # ------------------------------------------------------------------ #
    # Feature ranking
    # ------------------------------------------------------------------ #

    def get_top_features(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the top *n* features ranked by mean |SHAP| value.

        Parameters
        ----------
        n : int
            Number of features to return.

        Returns
        -------
        list[dict]
            Each dict: ``{"rank": int, "feature": str, "importance": float}``.
        """
        if self._mean_abs_shap is None:
            logger.warning("No SHAP values computed yet — returning empty list.")
            return []

        importance = self._build_importance_dict()
        top = list(importance.items())[:n]
        return [
            {"rank": rank + 1, "feature": feat, "importance": float(val)}
            for rank, (feat, val) in enumerate(top)
        ]

    # ------------------------------------------------------------------ #
    # JSON serialisation
    # ------------------------------------------------------------------ #

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the latest SHAP analysis.

        Returns
        -------
        dict
            Keys: ``feature_importance`` (list), ``num_samples``, ``num_features``.
        """
        top_features = self.get_top_features(n=20)
        return {
            "feature_importance": top_features,
            "num_samples": int(self.shap_values.shape[0]) if self.shap_values is not None else 0,
            "num_features": int(self._mean_abs_shap.shape[0]) if self._mean_abs_shap is not None else 0,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_importance_dict(self) -> Dict[str, float]:
        """Build ``{feature: mean_abs_shap}`` sorted descending."""
        if self._mean_abs_shap is None:
            return {}

        n_features = self._mean_abs_shap.shape[0]
        names = (
            self.feature_names
            if self.feature_names and len(self.feature_names) == n_features
            else [f"feature_{i}" for i in range(n_features)]
        )

        pairs = sorted(
            zip(names, self._mean_abs_shap.tolist()),
            key=lambda p: p[1],
            reverse=True,
        )
        return {feat: val for feat, val in pairs}
