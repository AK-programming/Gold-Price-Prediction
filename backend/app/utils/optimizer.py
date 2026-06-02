"""
Gold Price Forecasting System — Hyperparameter Optimisation

Uses Optuna to search for optimal hyperparameters for each model family
(XGBoost, LightGBM, CatBoost, Transformer).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from app import config
from app.utils.metrics import compute_mape, compute_rmse

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optuna-based hyperparameter search for all model types.

    Parameters
    ----------
    n_trials : int, optional
        Number of trials per study (overrides ``config.OPTUNA_CONFIG``).
    timeout : int, optional
        Maximum seconds per study (overrides ``config.OPTUNA_CONFIG``).
    """

    def __init__(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.n_trials: int = n_trials or config.OPTUNA_CONFIG["n_trials"]
        self.timeout: int = timeout or config.OPTUNA_CONFIG["timeout"]
        self.direction: str = config.OPTUNA_CONFIG["direction"]
        logger.info(
            "HyperparameterOptimizer init — n_trials=%d, timeout=%ds",
            self.n_trials,
            self.timeout,
        )

    # ------------------------------------------------------------------ #
    # XGBoost
    # ------------------------------------------------------------------ #

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Run an Optuna study for XGBoost.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and targets.
        X_val, y_val : np.ndarray
            Validation features and targets.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        import optuna
        from xgboost import XGBRegressor

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
            model = XGBRegressor(**params, verbosity=0, n_jobs=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = model.predict(X_val)
            return compute_mape(y_val, preds)

        best = self._run_study("xgboost", objective)
        logger.info("Best XGBoost params: %s", best)
        return best

    # ------------------------------------------------------------------ #
    # LightGBM
    # ------------------------------------------------------------------ #

    def optimize_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Run an Optuna study for LightGBM.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and targets.
        X_val, y_val : np.ndarray
            Validation features and targets.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        import optuna
        from lightgbm import LGBMRegressor

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "verbose": -1,
            }
            model = LGBMRegressor(**params, n_jobs=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
            preds = model.predict(X_val)
            return compute_mape(y_val, preds)

        best = self._run_study("lightgbm", objective)
        logger.info("Best LightGBM params: %s", best)
        return best

    # ------------------------------------------------------------------ #
    # CatBoost
    # ------------------------------------------------------------------ #

    def optimize_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Run an Optuna study for CatBoost.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and targets.
        X_val, y_val : np.ndarray
            Validation features and targets.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        import optuna
        from catboost import CatBoostRegressor

        def objective(trial: optuna.Trial) -> float:
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "iterations": trial.suggest_int("iterations", 100, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "verbose": 0,
            }
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=0,
            )
            preds = model.predict(X_val)
            return compute_mape(y_val, preds)

        best = self._run_study("catboost", objective)
        logger.info("Best CatBoost params: %s", best)
        return best

    # ------------------------------------------------------------------ #
    # Transformer
    # ------------------------------------------------------------------ #

    def optimize_transformer(
        self,
        train_loader: Any,
        val_loader: Any,
        num_features: int,
    ) -> Dict[str, Any]:
        """Run an Optuna study for the Transformer forecaster.

        Parameters
        ----------
        train_loader : DataLoader
            PyTorch training data loader.
        val_loader : DataLoader
            PyTorch validation data loader.
        num_features : int
            Number of input features per time step.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        import optuna
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def objective(trial: optuna.Trial) -> float:
            from app.models.transformer import TransformerForecaster

            trial_config = {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
                "num_attention_heads": trial.suggest_categorical("num_attention_heads", [4, 8]),
                "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 6),
                "num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 4),
                "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.05, 0.4),
                "quantiles": config.TRANSFORMER_CONFIG["quantiles"],
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "max_grad_norm": config.TRANSFORMER_CONFIG["max_grad_norm"],
                "batch_size": config.TRANSFORMER_CONFIG["batch_size"],
                "max_epochs": 30,  # shorter for HPO
                "early_stopping_patience": 5,
            }

            model = TransformerForecaster(
                num_features=num_features,
                config=trial_config,
            )

            val_loss = model.fit(train_loader, val_loader, device=device)
            return val_loss

        best = self._run_study("transformer", objective)
        logger.info("Best Transformer params: %s", best)
        return best

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_study(
        self,
        study_name: str,
        objective: Callable,
    ) -> Dict[str, Any]:
        """Create and run an Optuna study.

        Parameters
        ----------
        study_name : str
            Human-readable study identifier.
        objective : callable
            Optuna objective function.

        Returns
        -------
        dict
            Best trial parameters.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
        )

        logger.info(
            "Starting Optuna study '%s' — %d trials, %ds timeout",
            study_name,
            self.n_trials,
            self.timeout,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        logger.info(
            "Study '%s' finished — best value=%.6f, best params=%s",
            study_name,
            study.best_value,
            study.best_params,
        )

        return dict(study.best_params)
