"""
Gold Price Forecasting System — Training Orchestrator

Coordinates the full training pipeline: data download → feature engineering →
preprocessing → model training → ensemble → evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app import config
from app.utils.metrics import (
    compute_all_metrics,
    compute_per_horizon_metrics,
    format_metrics_table,
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """End-to-end training orchestrator for all gold-price forecasting models.

    Parameters
    ----------
    device : str, optional
        PyTorch device string. Auto-detected when *None*.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or self._detect_device()
        self.results: Dict[str, Dict[str, float]] = {}
        self._trained_models: Dict[str, Any] = {}
        logger.info("TrainingOrchestrator init — device=%s", self.device)

    # ------------------------------------------------------------------ #
    # Device detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_device() -> str:
        """Return ``'cuda'`` if a CUDA GPU is available, otherwise ``'cpu'``."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("CUDA GPU detected: %s", gpu_name)
                return "cuda"
        except ImportError:
            logger.warning("PyTorch not installed — falling back to CPU.")
        return "cpu"

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def run_full_pipeline(self) -> Dict[str, Dict[str, float]]:
        """Execute the complete training pipeline.

        Steps:
        1. Download / update market data.
        2. Engineer features.
        3. Preprocess (scale, create sequences).
        4. Train deep-learning models (Transformer, LSTM).
        5. Train tree-based models (XGBoost, LightGBM, CatBoost).
        6. Train stacking ensemble.
        7. Evaluate all models.
        8. Persist results.

        Returns
        -------
        dict
            ``{model_name: {metric_name: value, …}, …}`` for all models.
        """
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("FULL TRAINING PIPELINE — START")
        logger.info("=" * 60)

        # Step 1 — Data download
        logger.info("[1/8] Downloading market data …")
        try:
            from app.pipeline.data_downloader import download_all_data

            download_all_data()
        except ImportError:
            logger.warning("data_downloader not available — skipping download step.")

        # Step 2 — Feature engineering
        logger.info("[2/8] Engineering features …")
        try:
            from app.pipeline.feature_engineer import engineer_features

            engineer_features()
        except ImportError:
            logger.warning("feature_engineer not available — skipping feature step.")

        # Step 3 — Preprocessing
        logger.info("[3/8] Preprocessing data …")
        try:
            from app.pipeline.preprocessor import preprocess_data

            preprocess_data()
        except ImportError:
            logger.warning("preprocessor not available — skipping preprocess step.")

        # Step 4 — Deep models
        logger.info("[4/8] Training deep-learning models …")
        self.train_deep_models()

        # Step 5 — Tree models
        logger.info("[5/8] Training tree-based models …")
        self.train_tree_models()

        # Step 6 — Ensemble
        logger.info("[6/8] Training ensemble …")
        self.train_ensemble()

        # Step 7 — Evaluate
        logger.info("[7/8] Evaluating all models …")
        self.results = self.evaluate_all()

        # Step 8 — Save
        logger.info("[8/8] Saving results …")
        self.save_results(self.results)

        elapsed = time.perf_counter() - t0
        logger.info("FULL PIPELINE COMPLETE — %.1f s", elapsed)
        return self.results

    # ------------------------------------------------------------------ #
    # Deep-learning models
    # ------------------------------------------------------------------ #

    def train_deep_models(self) -> None:
        """Train Transformer and LSTM models."""
        # --- Transformer ---
        try:
            from app.models.transformer import TransformerForecaster
            from app.pipeline.preprocessor import load_processed_data

            logger.info("Loading processed data for Transformer …")
            data = load_processed_data()
            train_loader = data["train_loader"]
            val_loader = data["val_loader"]
            num_features = data["num_features"]

            model = TransformerForecaster(
                num_features=num_features,
                config=config.TRANSFORMER_CONFIG,
            )
            model.fit(
                train_loader,
                val_loader,
                device=self.device,
            )
            self._trained_models["transformer"] = model
            logger.info("Transformer training complete.")

        except ImportError:
            logger.warning("Transformer module not available — skipping.")
        except Exception as exc:
            logger.error("Transformer training failed: %s", exc, exc_info=True)

        # --- LSTM ---
        try:
            from app.models.lstm import LSTMForecaster
            from app.pipeline.preprocessor import load_processed_data

            data = load_processed_data()
            train_loader = data["train_loader"]
            val_loader = data["val_loader"]
            num_features = data["num_features"]

            model = LSTMForecaster(
                num_features=num_features,
                config=config.RNN_CONFIG,
            )
            model.fit(
                train_loader,
                val_loader,
                device=self.device,
            )
            self._trained_models["lstm"] = model
            logger.info("LSTM training complete.")

        except ImportError:
            logger.warning("LSTM module not available — skipping.")
        except Exception as exc:
            logger.error("LSTM training failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Tree-based models
    # ------------------------------------------------------------------ #

    def train_tree_models(self) -> None:
        """Train XGBoost, LightGBM, and CatBoost models."""
        try:
            from app.pipeline.preprocessor import load_processed_data

            data = load_processed_data()
            X_train = data["X_train_flat"]
            y_train = data["y_train_flat"]
            X_val = data["X_val_flat"]
            y_val = data["y_val_flat"]
        except (ImportError, KeyError) as exc:
            logger.error("Cannot load flat training data: %s", exc)
            return

        # --- XGBoost ---
        try:
            from app.models.xgboost_model import XGBoostForecaster

            model = XGBoostForecaster(config=config.XGBOOST_CONFIG)
            model.fit(X_train, y_train, X_val, y_val)
            self._trained_models["xgboost"] = model
            logger.info("XGBoost training complete.")
        except ImportError:
            logger.warning("XGBoost module not available — skipping.")
        except Exception as exc:
            logger.error("XGBoost training failed: %s", exc, exc_info=True)

        # --- LightGBM ---
        try:
            from app.models.lightgbm_model import LightGBMForecaster

            model = LightGBMForecaster(config=config.LIGHTGBM_CONFIG)
            model.fit(X_train, y_train, X_val, y_val)
            self._trained_models["lightgbm"] = model
            logger.info("LightGBM training complete.")
        except ImportError:
            logger.warning("LightGBM module not available — skipping.")
        except Exception as exc:
            logger.error("LightGBM training failed: %s", exc, exc_info=True)

        # --- CatBoost ---
        try:
            from app.models.catboost_model import CatBoostForecaster

            model = CatBoostForecaster(config=config.CATBOOST_CONFIG)
            model.fit(X_train, y_train, X_val, y_val)
            self._trained_models["catboost"] = model
            logger.info("CatBoost training complete.")
        except ImportError:
            logger.warning("CatBoost module not available — skipping.")
        except Exception as exc:
            logger.error("CatBoost training failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Ensemble
    # ------------------------------------------------------------------ #

    def train_ensemble(self) -> None:
        """Train a stacking ensemble over all individual model predictions."""
        if not self._trained_models:
            logger.warning("No trained models available — cannot build ensemble.")
            return

        try:
            from app.models.ensemble import EnsembleForecaster
            from app.pipeline.preprocessor import load_processed_data

            data = load_processed_data()
            X_val = data.get("X_val_flat", data.get("X_val"))
            y_val = data.get("y_val_flat", data.get("y_val"))

            if X_val is None or y_val is None:
                logger.error("Validation data unavailable — cannot train ensemble.")
                return

            ensemble = EnsembleForecaster(
                models=self._trained_models,
                config=config.ENSEMBLE_CONFIG,
            )
            ensemble.fit(X_val, y_val)
            self._trained_models["ensemble"] = ensemble
            logger.info("Ensemble training complete (%d base models).",
                        len(self._trained_models) - 1)

        except ImportError:
            logger.warning("Ensemble module not available — skipping.")
        except Exception as exc:
            logger.error("Ensemble training failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate every trained model on the test set.

        Returns
        -------
        dict
            ``{model_name: {metric_name: value}}``.
        """
        if not self._trained_models:
            logger.warning("No trained models to evaluate.")
            return {}

        try:
            from app.pipeline.preprocessor import load_processed_data

            data = load_processed_data()
        except (ImportError, Exception) as exc:
            logger.error("Cannot load test data for evaluation: %s", exc)
            return {}

        # Support both sequential and flat test data
        X_test_seq = data.get("X_test")
        y_test_seq = data.get("y_test")
        X_test_flat = data.get("X_test_flat")
        y_test_flat = data.get("y_test_flat")

        results: Dict[str, Dict[str, float]] = {}

        for name, model in self._trained_models.items():
            logger.info("Evaluating model: %s", name)
            try:
                # Deep models expect sequential input
                if name in ("transformer", "lstm"):
                    if X_test_seq is None or y_test_seq is None:
                        logger.warning("Sequential test data missing for %s.", name)
                        continue
                    preds = model.predict(X_test_seq)
                    y_true = np.asarray(y_test_seq)
                else:
                    if X_test_flat is None or y_test_flat is None:
                        logger.warning("Flat test data missing for %s.", name)
                        continue
                    preds = model.predict(X_test_flat)
                    y_true = np.asarray(y_test_flat)

                preds = np.asarray(preds)
                metrics = compute_all_metrics(y_true, preds)
                results[name] = metrics
                logger.info("  %s: RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  R²=%.4f",
                            name, metrics["rmse"], metrics["mae"],
                            metrics["mape"], metrics["r2"])

            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", name, exc, exc_info=True)

        if results:
            table = format_metrics_table(results)
            logger.info("\n%s", table)

        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_results(self, results: Dict[str, Dict[str, float]]) -> Path:
        """Save evaluation results to a timestamped JSON in ``config.LOG_DIR``.

        Parameters
        ----------
        results : dict
            ``{model_name: {metric_name: value}}``.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.LOG_DIR / f"metrics_{timestamp}.json"

        payload = {
            "timestamp": timestamp,
            "device": self.device,
            "models": results,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info("Metrics saved to %s", out_path)
        return out_path
