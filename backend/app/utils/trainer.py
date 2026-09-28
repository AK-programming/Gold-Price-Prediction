"""
Gold Price Forecasting System — Training Orchestrator

Coordinates data download → features → preprocessing → model training → evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app import config
from app.utils.metrics import compute_all_metrics, format_metrics_table

logger = logging.getLogger(__name__)

_SERVEABLE = ("transformer", "lstm", "xgboost", "lightgbm", "catboost", "ensemble")


class TrainingOrchestrator:
    """End-to-end training for all gold-price forecasting models."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or self._detect_device()
        self.results: Dict[str, Dict[str, float]] = {}
        self._torch_device = torch.device(self.device)
        logger.info("TrainingOrchestrator init — device=%s", self.device)

    @staticmethod
    def _detect_device() -> str:
        try:
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def run_full_pipeline(self, *, refresh_data: bool = False) -> Dict[str, Dict[str, float]]:
        """Execute download → features → preprocess → train → evaluate → save metrics."""
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("FULL TRAINING PIPELINE — START")
        logger.info("=" * 60)

        features_df = self._prepare_data(refresh_data=refresh_data)
        (
            train_scaled,
            val_scaled,
            test_scaled,
            feature_cols,
            target_idx,
        ) = self._preprocess(features_df)
        datasets = self._build_datasets(
            train_scaled, val_scaled, test_scaled, target_idx
        )

        flat_feature_names = [
            f"{col}_t{step}"
            for step in range(config.INPUT_SEQUENCE_LENGTH)
            for col in feature_cols
        ]

        self._train_tree_models(datasets, flat_feature_names)
        dl_models = self._train_deep_models(datasets, feature_cols)
        self._train_ensemble(datasets, dl_models)
        self.results = self._evaluate_all(datasets, dl_models)
        self.save_results(self.results)

        logger.info("FULL PIPELINE COMPLETE — %.1f s", time.perf_counter() - t0)
        return self.results

    def _prepare_data(self, *, refresh_data: bool) -> Any:
        import pandas as pd

        from app.pipeline.data_ingestion import download_all_data, load_raw_data
        from app.pipeline.feature_engineering import engineer_features, load_features

        if refresh_data or not config.RAW_DATA_FILE.exists():
            logger.info("[1/6] Downloading market data …")
            raw_df = download_all_data()
        else:
            logger.info("[1/6] Loading cached raw data …")
            raw_df = load_raw_data()

        if refresh_data or not config.FEATURES_FILE.exists():
            logger.info("[2/6] Engineering features …")
            return engineer_features(raw_df)

        logger.info("[2/6] Loading cached features …")
        return load_features()

    def _preprocess(self, features_df: Any) -> Tuple[np.ndarray, ...]:
        from app.pipeline.preprocessor import preprocess_data

        logger.info("[3/6] Preprocessing …")
        train_scaled, val_scaled, test_scaled, feature_cols, target_idx, _ = (
            preprocess_data(features_df)
        )
        return train_scaled, val_scaled, test_scaled, feature_cols, target_idx

    def _build_datasets(
        self,
        train_scaled: np.ndarray,
        val_scaled: np.ndarray,
        test_scaled: np.ndarray,
        target_idx: int,
    ) -> Dict[str, Any]:
        from app.pipeline.preprocessor import (
            create_sequences,
            create_tabular_dataset,
            get_dataloaders,
        )

        loaders = get_dataloaders(
            train_scaled, val_scaled, test_scaled, target_idx=target_idx
        )
        X_train, y_train = create_tabular_dataset(train_scaled, target_idx=target_idx)
        X_val, y_val = create_tabular_dataset(val_scaled, target_idx=target_idx)
        X_test, y_test = create_tabular_dataset(test_scaled, target_idx=target_idx)

        X_val_seq, y_val_seq = create_sequences(val_scaled, target_idx=target_idx)
        X_test_seq, y_test_seq = create_sequences(test_scaled, target_idx=target_idx)

        return {
            "loaders": loaders,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "X_val_seq": X_val_seq,
            "y_val_seq": y_val_seq,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "num_features": train_scaled.shape[1],
        }

    def _train_tree_models(
        self, datasets: Dict[str, Any], flat_feature_names: List[str]
    ) -> Dict[str, Any]:
        from app.models import CatBoostModel, LightGBMModel, XGBoostModel

        logger.info("[4/6] Training tree models …")
        models: Dict[str, Any] = {}
        eval_set = [(datasets["X_val"], datasets["y_val"])]

        for name, cls, filename in [
            ("xgboost", XGBoostModel, "xgboost.joblib"),
            ("lightgbm", LightGBMModel, "lightgbm.joblib"),
            ("catboost", CatBoostModel, "catboost.joblib"),
        ]:
            logger.info("  Training %s …", name)
            model = cls()
            model.fit(
                datasets["X_train"],
                datasets["y_train"],
                eval_set=eval_set,
                feature_names=flat_feature_names,
            )
            model.save(config.CHECKPOINT_DIR / filename)
            models[name] = model

        return models

    def _train_deep_models(
        self, datasets: Dict[str, Any], feature_cols: List[str]
    ) -> Dict[str, Any]:
        from app.models import GoldLSTM, GoldTransformer, train_rnn, train_transformer

        logger.info("[5/6] Training deep models …")
        loaders = datasets["loaders"]
        num_features = datasets["num_features"]
        models: Dict[str, Any] = {}

        lstm = GoldLSTM(num_features=num_features)
        train_rnn(
            lstm,
            loaders["train"],
            loaders["val"],
            device=self._torch_device,
        )
        torch.save(
            {
                "model_state_dict": lstm.state_dict(),
                "num_features": num_features,
                "feature_cols": feature_cols,
            },
            config.CHECKPOINT_DIR / "lstm.pt",
        )
        models["lstm"] = lstm

        transformer = GoldTransformer(num_features=num_features)
        train_transformer(
            transformer,
            loaders["train"],
            loaders["val"],
            device=self._torch_device,
        )
        torch.save(
            {
                "model_state_dict": transformer.state_dict(),
                "num_features": num_features,
                "feature_cols": feature_cols,
            },
            config.CHECKPOINT_DIR / "transformer.pt",
        )
        models["transformer"] = transformer

        return models

    @staticmethod
    def _ensemble_tabular_from_sequences(X_seq: np.ndarray) -> np.ndarray:
        """Flatten sequence windows to match boosting model input layout."""
        return X_seq.reshape(X_seq.shape[0], -1).astype(np.float32)

    def _train_ensemble(
        self, datasets: Dict[str, Any], dl_models: Dict[str, Any]
    ) -> None:
        from app.models import StackingEnsemble

        logger.info("[6/6] Training stacking ensemble …")
        try:
            from app.models import CatBoostModel, LightGBMModel, XGBoostModel

            ensemble = StackingEnsemble(
                transformer=dl_models["transformer"],
                lstm=dl_models["lstm"],
                xgb_model=XGBoostModel().load(config.CHECKPOINT_DIR / "xgboost.joblib"),
                lgb_model=LightGBMModel().load(config.CHECKPOINT_DIR / "lightgbm.joblib"),
                cat_model=CatBoostModel().load(config.CHECKPOINT_DIR / "catboost.joblib"),
                device=self._torch_device,
            )
            x_val_tab = self._ensemble_tabular_from_sequences(datasets["X_val_seq"])
            ensemble.fit_meta_learner(
                datasets["X_val_seq"],
                x_val_tab,
                datasets["y_val_seq"],
            )
            ensemble.save(config.CHECKPOINT_DIR / "ensemble.joblib")
            logger.info("Ensemble meta-learner saved.")
        except Exception as exc:
            logger.error("Ensemble training failed: %s", exc, exc_info=True)

    @staticmethod
    def _predict_deep(model: Any, X: np.ndarray, device: torch.device) -> np.ndarray:
        model.eval().to(device)
        batch_size = config.TRANSFORMER_CONFIG["batch_size"]
        chunks: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.tensor(
                    X[start : start + batch_size], dtype=torch.float32, device=device
                )
                output = model(batch)
                if isinstance(output, dict):
                    output = output["q50"]
                chunks.append(output.cpu().numpy())

        return np.concatenate(chunks, axis=0)

    def _evaluate_all(
        self, datasets: Dict[str, Any], dl_models: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        from app.models import CatBoostModel, LightGBMModel, XGBoostModel

        logger.info("Evaluating models on validation and test splits …")
        results: Dict[str, Dict[str, float]] = {}

        tree_models = {
            "xgboost": XGBoostModel().load(config.CHECKPOINT_DIR / "xgboost.joblib"),
            "lightgbm": LightGBMModel().load(config.CHECKPOINT_DIR / "lightgbm.joblib"),
            "catboost": CatBoostModel().load(config.CHECKPOINT_DIR / "catboost.joblib"),
        }

        for name, model in tree_models.items():
            val_pred = model.predict(datasets["X_val"])
            test_pred = model.predict(datasets["X_test"])
            results[name] = self._metric_bundle(
                datasets["y_val"], val_pred, datasets["y_test"], test_pred
            )

        for name in ("lstm", "transformer"):
            model = dl_models[name]
            val_pred = self._predict_deep(model, datasets["X_val_seq"], self._torch_device)
            test_pred = self._predict_deep(model, datasets["X_test_seq"], self._torch_device)
            results[name] = self._metric_bundle(
                datasets["y_val_seq"], val_pred, datasets["y_test_seq"], test_pred
            )

        ensemble_path = config.CHECKPOINT_DIR / "ensemble.joblib"
        if ensemble_path.exists():
            try:
                from app.models.ensemble import load_stacking_ensemble

                ensemble = load_stacking_ensemble(self._torch_device)
                x_val_tab = self._ensemble_tabular_from_sequences(datasets["X_val_seq"])
                x_test_tab = self._ensemble_tabular_from_sequences(datasets["X_test_seq"])
                val_pred = ensemble.predict(datasets["X_val_seq"], x_val_tab)
                test_pred = ensemble.predict(datasets["X_test_seq"], x_test_tab)
                results["ensemble"] = self._metric_bundle(
                    datasets["y_val_seq"], val_pred, datasets["y_test_seq"], test_pred
                )
            except Exception as exc:
                logger.error("Ensemble evaluation failed: %s", exc, exc_info=True)

        if results:
            logger.info("\n%s", format_metrics_table(
                {k: {"rmse": v.get("val_rmse", v.get("rmse", 0))} for k, v in results.items()}
            ))

        return results

    def build_ensemble_from_checkpoints(self) -> Path:
        """Fit the stacking meta-learner from existing base model checkpoints."""
        import pandas as pd

        from app.pipeline.preprocessor import preprocess_data

        if not config.FEATURES_FILE.exists():
            raise FileNotFoundError(f"Feature file not found: {config.FEATURES_FILE}")

        for name in ("transformer", "lstm", "xgboost", "lightgbm", "catboost"):
            if not (config.CHECKPOINT_DIR / (
                f"{name}.pt" if name in ("transformer", "lstm") else f"{name}.joblib"
            )).exists():
                raise FileNotFoundError(
                    f"Missing base model '{name}'. Train all base models first."
                )

        features_df = pd.read_parquet(config.FEATURES_FILE)
        train_scaled, val_scaled, test_scaled, _, target_idx, _ = preprocess_data(
            features_df
        )
        datasets = self._build_datasets(
            train_scaled, val_scaled, test_scaled, target_idx
        )

        logger.info("Fitting ensemble meta-learner on validation split …")
        from app.models import (
            CatBoostModel,
            GoldLSTM,
            GoldTransformer,
            LightGBMModel,
            XGBoostModel,
            StackingEnsemble,
        )

        transformer_ckpt = torch.load(
            config.CHECKPOINT_DIR / "transformer.pt", map_location="cpu"
        )
        lstm_ckpt = torch.load(config.CHECKPOINT_DIR / "lstm.pt", map_location="cpu")
        num_features = int(transformer_ckpt["num_features"])

        transformer = GoldTransformer(num_features=num_features)
        transformer.load_state_dict(transformer_ckpt["model_state_dict"])
        lstm = GoldLSTM(num_features=num_features)
        lstm.load_state_dict(lstm_ckpt["model_state_dict"])

        ensemble = StackingEnsemble(
            transformer=transformer,
            lstm=lstm,
            xgb_model=XGBoostModel().load(config.CHECKPOINT_DIR / "xgboost.joblib"),
            lgb_model=LightGBMModel().load(config.CHECKPOINT_DIR / "lightgbm.joblib"),
            cat_model=CatBoostModel().load(config.CHECKPOINT_DIR / "catboost.joblib"),
            device=self._torch_device,
        )
        x_val_tab = self._ensemble_tabular_from_sequences(datasets["X_val_seq"])
        ensemble.fit_meta_learner(
            datasets["X_val_seq"],
            x_val_tab,
            datasets["y_val_seq"],
        )
        out_path = config.CHECKPOINT_DIR / "ensemble.joblib"
        ensemble.save(out_path)
        logger.info("Ensemble saved to %s", out_path)

        self.results = self.evaluate_existing_checkpoints()
        return out_path

    def save_results(self, results: Dict[str, Dict[str, float]]) -> Path:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = config.LOG_DIR / f"metrics_{timestamp}.json"

        best_model = None
        best_val = float("inf")
        for name in _SERVEABLE:
            val_rmse = results.get(name, {}).get("val_rmse")
            if isinstance(val_rmse, (int, float)) and val_rmse < best_val:
                best_val = float(val_rmse)
                best_model = name

        payload = {
            "timestamp": timestamp,
            "device": self.device,
            "best_model": best_model,
            "models": results,
        }

        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        logger.info("Metrics saved to %s (best_model=%s)", out_path, best_model)
        return out_path

    def evaluate_existing_checkpoints(self) -> Dict[str, Dict[str, float]]:
        """Evaluate saved checkpoints without retraining (writes metrics JSON)."""
        import pandas as pd

        from app.pipeline.preprocessor import (
            create_sequences,
            create_tabular_dataset,
            preprocess_data,
        )

        if not config.FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Feature file not found: {config.FEATURES_FILE}. Run training first."
            )

        features_df = pd.read_parquet(config.FEATURES_FILE)
        train_scaled, val_scaled, test_scaled, feature_cols, target_idx, _ = (
            preprocess_data(features_df)
        )
        datasets = self._build_datasets(
            train_scaled, val_scaled, test_scaled, target_idx
        )

        from app.models import (
            CatBoostModel,
            GoldLSTM,
            GoldTransformer,
            LightGBMModel,
            XGBoostModel,
        )

        results: Dict[str, Dict[str, float]] = {}

        for name, cls, filename in [
            ("xgboost", XGBoostModel, "xgboost.joblib"),
            ("lightgbm", LightGBMModel, "lightgbm.joblib"),
            ("catboost", CatBoostModel, "catboost.joblib"),
        ]:
            path = config.CHECKPOINT_DIR / filename
            if not path.exists():
                continue
            model = cls().load(path)
            val_pred = model.predict(datasets["X_val"])
            test_pred = model.predict(datasets["X_test"])
            results[name] = self._metric_bundle(
                datasets["y_val"], val_pred, datasets["y_test"], test_pred
            )

        for name, cls, filename in [
            ("lstm", GoldLSTM, "lstm.pt"),
            ("transformer", GoldTransformer, "transformer.pt"),
        ]:
            path = config.CHECKPOINT_DIR / filename
            if not path.exists():
                continue
            ckpt = torch.load(path, map_location="cpu")
            model = cls(num_features=int(ckpt["num_features"]))
            model.load_state_dict(ckpt["model_state_dict"])
            val_pred = self._predict_deep(
                model, datasets["X_val_seq"], self._torch_device
            )
            test_pred = self._predict_deep(
                model, datasets["X_test_seq"], self._torch_device
            )
            results[name] = self._metric_bundle(
                datasets["y_val_seq"], val_pred, datasets["y_test_seq"], test_pred
            )

        ensemble_path = config.CHECKPOINT_DIR / "ensemble.joblib"
        if ensemble_path.exists():
            try:
                from app.models.ensemble import load_stacking_ensemble

                ensemble = load_stacking_ensemble(self._torch_device)
                x_val_tab = self._ensemble_tabular_from_sequences(datasets["X_val_seq"])
                x_test_tab = self._ensemble_tabular_from_sequences(datasets["X_test_seq"])
                val_pred = ensemble.predict(datasets["X_val_seq"], x_val_tab)
                test_pred = ensemble.predict(datasets["X_test_seq"], x_test_tab)
                results["ensemble"] = self._metric_bundle(
                    datasets["y_val_seq"], val_pred, datasets["y_test_seq"], test_pred
                )
            except Exception as exc:
                logger.error("Ensemble evaluation failed: %s", exc, exc_info=True)

        if not results:
            raise FileNotFoundError("No checkpoints found to evaluate.")

        self.results = results
        return self.save_results(results)

    @staticmethod
    def _metric_bundle(
        y_val: np.ndarray,
        val_pred: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
    ) -> Dict[str, float]:
        val_metrics = compute_all_metrics(y_val, val_pred)
        test_metrics = compute_all_metrics(y_test, test_pred)
        return {
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "val_rmse": val_metrics["rmse"],
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "mape": test_metrics["mape"],
            "r2": test_metrics["r2"],
            "directional_accuracy": test_metrics["directional_accuracy"],
        }
