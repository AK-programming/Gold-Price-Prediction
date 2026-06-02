"""
Stacking ensemble for gold price forecasting.

Combines predictions from five base models (Transformer, LSTM, XGBoost,
LightGBM, CatBoost) via a Ridge regression meta-learner trained on
out-of-fold base predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from app import config
from app.models.transformer_model import GoldTransformer
from app.models.rnn_model import GoldLSTM
from app.models.boosting_models import (
    BaseBoostingModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
)

logger = logging.getLogger(__name__)

# Alias for readability
_BoostModel = BaseBoostingModel


class StackingEnsemble:
    """Stacking ensemble combining deep-learning and tree-based base models.

    The ensemble:

    1. Collects single-step (or multi-step) predictions from each base model.
    2. Stacks them column-wise into a meta-feature matrix.
    3. Trains a Ridge regression meta-learner to produce final predictions.

    Args:
        transformer: Trained ``GoldTransformer`` instance.
        lstm: Trained ``GoldLSTM`` instance.
        xgb_model: Trained ``XGBoostModel`` instance.
        lgb_model: Trained ``LightGBMModel`` instance.
        cat_model: Trained ``CatBoostModel`` instance.
        cv_folds: Number of cross-validation folds used when fitting
            the meta-learner (default from ``config.ENSEMBLE_CONFIG``).
        device: PyTorch device for neural network inference.
    """

    BASE_MODEL_NAMES = ["transformer", "lstm", "xgboost", "lightgbm", "catboost"]

    def __init__(
        self,
        transformer: GoldTransformer,
        lstm: GoldLSTM,
        xgb_model: XGBoostModel,
        lgb_model: LightGBMModel,
        cat_model: CatBoostModel,
        cv_folds: int = config.ENSEMBLE_CONFIG["cv_folds"],
        device: Optional[torch.device] = None,
    ) -> None:
        self.transformer = transformer
        self.lstm = lstm
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.cv_folds = cv_folds
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.meta_learner: Optional[RidgeCV] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Collect base predictions
    # ------------------------------------------------------------------
    def collect_base_predictions(
        self,
        X_seq: np.ndarray,
        X_tab: np.ndarray,
    ) -> np.ndarray:
        """Collect predictions from all five base models.

        Args:
            X_seq: Sequential input for Transformer and LSTM of shape
                ``(n_samples, seq_len, num_features)``.
            X_tab: Tabular input for boosting models of shape
                ``(n_samples, n_features)``.

        Returns:
            Meta-feature matrix of shape ``(n_samples, 5 * output_seq_len)``,
            i.e. each base model contributes ``output_seq_len`` columns.
        """
        preds_list: List[np.ndarray] = []

        # --- Neural models (Transformer & LSTM) ---
        self.transformer.to(self.device).eval()
        self.lstm.to(self.device).eval()

        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)

            # Transformer → use median quantile (q50)
            t_out = self.transformer(x_tensor)
            preds_list.append(t_out["q50"].cpu().numpy())  # (N, T)

            # LSTM → point prediction
            l_out = self.lstm(x_tensor)
            preds_list.append(l_out.cpu().numpy())  # (N, T)

        # --- Boosting models (multi-step recursive) ---
        output_len = config.OUTPUT_SEQUENCE_LENGTH
        for bm in [self.xgb_model, self.lgb_model, self.cat_model]:
            preds_list.append(bm.predict_multistep(X_tab, steps=output_len))  # (N, T)

        # Stack horizontally: each model → output_seq_len columns
        meta_features = np.concatenate(preds_list, axis=1)  # (N, 5*T)
        return meta_features

    # ------------------------------------------------------------------
    # Fit meta-learner
    # ------------------------------------------------------------------
    def fit_meta_learner(
        self,
        X_seq: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
    ) -> "StackingEnsemble":
        """Train the Ridge meta-learner on base model predictions.

        Collects out-of-fold base predictions and fits a ``RidgeCV``
        meta-learner with built-in cross-validated alpha selection.

        Args:
            X_seq: Sequential input of shape ``(n_samples, seq_len, num_features)``.
            X_tab: Tabular input of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples, output_seq_len)``.

        Returns:
            ``self``.
        """
        n_samples = X_seq.shape[0]
        output_len = y.shape[1] if y.ndim > 1 else 1

        # Collect base predictions for the full training set
        meta_features = self.collect_base_predictions(X_seq, X_tab)

        # Flatten targets for multi-output Ridge
        y_flat = y.reshape(n_samples, -1) if y.ndim > 1 else y

        # Fit RidgeCV meta-learner (alpha selected via internal LOO-CV)
        self.meta_learner = RidgeCV(
            alphas=np.logspace(-4, 4, 20),
            fit_intercept=True,
        )
        self.meta_learner.fit(meta_features, y_flat)
        self._is_fitted = True

        logger.info(
            "Meta-learner fitted — alpha=%.4e, meta-features shape=%s",
            self.meta_learner.alpha_,
            meta_features.shape,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        X_seq: np.ndarray,
        X_tab: np.ndarray,
    ) -> np.ndarray:
        """Generate final ensemble predictions.

        Args:
            X_seq: Sequential input of shape ``(n_samples, seq_len, num_features)``.
            X_tab: Tabular input of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples, output_seq_len)``.

        Raises:
            RuntimeError: If the meta-learner has not been fitted.
        """
        if not self._is_fitted or self.meta_learner is None:
            raise RuntimeError(
                "Meta-learner is not fitted. Call fit_meta_learner() first."
            )

        meta_features = self.collect_base_predictions(X_seq, X_tab)
        raw_preds = self.meta_learner.predict(meta_features)

        # Reshape to (n_samples, output_seq_len) if needed
        output_len = config.OUTPUT_SEQUENCE_LENGTH
        if raw_preds.ndim == 1:
            raw_preds = raw_preds.reshape(-1, output_len)

        return raw_preds

    # ------------------------------------------------------------------
    # Model weights / contribution
    # ------------------------------------------------------------------
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Return the contribution (Ridge coefficients) of each base model.

        The meta-learner coefficients are split into contiguous blocks
        corresponding to each base model's output columns.

        Returns:
            Dictionary mapping model name to its coefficient block
            (array of shape ``(output_seq_len,)`` or the summed absolute
            weight as a scalar).

        Raises:
            RuntimeError: If the meta-learner has not been fitted.
        """
        if not self._is_fitted or self.meta_learner is None:
            raise RuntimeError(
                "Meta-learner is not fitted. Call fit_meta_learner() first."
            )

        coefs = self.meta_learner.coef_  # (output_dim, 5*T) or (5*T,)
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)

        output_len = config.OUTPUT_SEQUENCE_LENGTH
        weights: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(self.BASE_MODEL_NAMES):
            start = idx * output_len
            end = start + output_len
            block = coefs[:, start:end]  # (output_dim, T)
            # Summarise as mean absolute weight across all outputs & horizons
            weights[name] = float(np.mean(np.abs(block)))

        # Normalise to sum to 1 for interpretability
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save the meta-learner and model references to disk.

        Base models must be saved independently. This method only persists
        the meta-learner and ensemble metadata.

        Args:
            path: File path (typically ``.joblib``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "meta_learner": self.meta_learner,
                "is_fitted": self._is_fitted,
                "cv_folds": self.cv_folds,
            },
            path,
        )
        logger.info("Ensemble meta-learner saved to %s", path)

    def load(self, path: Union[str, Path]) -> "StackingEnsemble":
        """Load a previously saved meta-learner.

        The base models must already be loaded into this instance before
        calling ``predict()``.

        Args:
            path: File path to load from.

        Returns:
            ``self``.
        """
        path = Path(path)
        data = joblib.load(path)
        self.meta_learner = data["meta_learner"]
        self._is_fitted = data.get("is_fitted", True)
        self.cv_folds = data.get("cv_folds", self.cv_folds)
        logger.info("Ensemble meta-learner loaded from %s", path)
        return self
