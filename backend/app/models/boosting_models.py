"""
Tree-based boosting model wrappers for gold price forecasting.

Provides a unified interface (``BaseBoostingModel``) for XGBoost, LightGBM,
and CatBoost regressors, including:

* Single-step ``fit`` / ``predict``.
* Recursive multi-step prediction (``predict_multistep``).
* Feature importance retrieval.
* Joblib-based serialisation.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from app import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Abstract Base
# ------------------------------------------------------------------
class BaseBoostingModel(ABC):
    """Abstract base class for tree-based boosting models.

    Defines a consistent interface for fitting, predicting, persisting,
    and inspecting feature importance across different gradient boosting
    libraries.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.model: object = None
        self.feature_names: Optional[List[str]] = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List] = None,
    ) -> "BaseBoostingModel":
        """Fit the model on training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.
            eval_set: Optional list of ``(X_val, y_val)`` tuples for early
                stopping evaluation.

        Returns:
            ``self`` for method chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate single-step predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)``.
        """

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances as a name→score mapping.

        Returns:
            Dictionary mapping feature names (or indices) to importance scores.
        """

    def predict_multistep(
        self,
        X: np.ndarray,
        steps: int = config.OUTPUT_SEQUENCE_LENGTH,
    ) -> np.ndarray:
        """Recursive multi-step prediction.

        Starting from the initial feature vector(s) ``X``, predict one step
        at a time and append each prediction as the last feature for the
        subsequent step.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            steps: Number of future steps to predict.

        Returns:
            Array of shape ``(n_samples, steps)`` with predictions for each
            horizon.
        """
        predictions: List[np.ndarray] = []
        current_X = X.copy()

        for step in range(steps):
            step_pred = self.predict(current_X)  # (n_samples,)
            predictions.append(step_pred)

            # Roll features: drop the oldest lag, append new prediction
            current_X = np.roll(current_X, shift=-1, axis=1)
            current_X[:, -1] = step_pred

        return np.column_stack(predictions)  # (n_samples, steps)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the model to disk using joblib.

        Args:
            path: File path to save to (typically ``.joblib``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names, "name": self.name},
            path,
        )
        logger.info("Saved %s model to %s", self.name, path)

    def load(self, path: Union[str, Path]) -> "BaseBoostingModel":
        """Load a persisted model from disk.

        Args:
            path: File path to load from.

        Returns:
            ``self`` for method chaining.
        """
        path = Path(path)
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names")
        self.name = data.get("name", self.name)
        logger.info("Loaded %s model from %s", self.name, path)
        return self


# ------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------
class XGBoostModel(BaseBoostingModel):
    """XGBoost regressor wrapper.

    Uses hyperparameters from ``config.XGBOOST_CONFIG`` by default.

    Args:
        params: Optional dict to override default XGBoost parameters.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(name="xgboost")
        merged = {**config.XGBOOST_CONFIG, **(params or {})}
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            **merged,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostModel":
        """Fit the XGBoost model.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training targets of shape ``(n_samples,)``.
            eval_set: Optional validation sets for early stopping.
            feature_names: Optional list of feature names for interpretability.

        Returns:
            ``self``.
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X.shape[1])
        ]
        fit_params: dict = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
        logger.info("XGBoost fitted — %d samples, %d features.", X.shape[0], X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances.tolist()))


# ------------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------------
class LightGBMModel(BaseBoostingModel):
    """LightGBM regressor wrapper.

    Uses hyperparameters from ``config.LIGHTGBM_CONFIG`` by default.

    Args:
        params: Optional dict to override default LightGBM parameters.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(name="lightgbm")
        merged = {**config.LIGHTGBM_CONFIG, **(params or {})}
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_jobs=-1,
            random_state=42,
            **merged,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LightGBMModel":
        """Fit the LightGBM model.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training targets of shape ``(n_samples,)``.
            eval_set: Optional validation sets for early stopping.
            feature_names: Optional list of feature names.

        Returns:
            ``self``.
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X.shape[1])
        ]
        fit_params: dict = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X, y, **fit_params)
        logger.info("LightGBM fitted — %d samples, %d features.", X.shape[0], X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances.tolist()))


# ------------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------------
class CatBoostModel(BaseBoostingModel):
    """CatBoost regressor wrapper.

    Uses hyperparameters from ``config.CATBOOST_CONFIG`` by default.

    Args:
        params: Optional dict to override default CatBoost parameters.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(name="catboost")
        merged = {**config.CATBOOST_CONFIG, **(params or {})}
        self.model = cb.CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            **merged,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "CatBoostModel":
        """Fit the CatBoost model.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Training targets of shape ``(n_samples,)``.
            eval_set: Optional validation sets for early stopping.
            feature_names: Optional list of feature names.

        Returns:
            ``self``.
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X.shape[1])
        ]
        fit_params: dict = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X, y, **fit_params)
        logger.info("CatBoost fitted — %d samples, %d features.", X.shape[0], X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        importances = self.model.get_feature_importance()
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances.tolist()))
