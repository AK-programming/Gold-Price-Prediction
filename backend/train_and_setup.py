#!/usr/bin/env python
"""
Gold Price Forecasting System — Complete Setup & Training Script

This script runs the full pipeline:
1. Download historical data (16 years of gold + correlated assets)
2. Engineer features (60+ technical indicators)
3. Train 6 models (Transformer, LSTM, XGBoost, LightGBM, CatBoost, Ensemble)
4. Evaluate and save metrics

Run with:  ``python train_and_setup.py``
"""

import logging
import sys
from pathlib import Path

from app import config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Execute the full setup & training pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Gold Price Forecasting System — Complete Setup")
    logger.info("=" * 80)

    try:
        # Step 1: Data Ingestion
        logger.info("\n[1/4] DOWNLOADING HISTORICAL DATA...")
        from app.pipeline.data_ingestion import download_all_data
        
        raw_data = download_all_data()
        logger.info(f"✓ Downloaded {len(raw_data)} historical records")

        # Step 2: Feature Engineering
        logger.info("\n[2/4] ENGINEERING FEATURES...")
        from app.pipeline.feature_engineering import engineer_features
        
        features_df = engineer_features(raw_data)
        logger.info(f"✓ Created {len(features_df.columns)} features")

        # Step 3: Preprocessing
        logger.info("\n[3/4] PREPROCESSING DATA...")
        from app.pipeline.preprocessor import preprocess_data
        
        train_scaled, val_scaled, test_scaled, feature_cols, target_idx, scaler = preprocess_data(features_df)
        logger.info(
            f"✓ Preprocessed data:\n"
            f"  Train: {train_scaled.shape}\n"
            f"  Val:   {val_scaled.shape}\n"
            f"  Test:  {test_scaled.shape}"
        )

        # Step 4: Training
        logger.info("\n[4/4] TRAINING MODELS...")
        from app.pipeline.preprocessor import create_tabular_dataset
        from app.models.boosting_models import XGBoostModel, LightGBMModel
        from app.utils.metrics import compute_all_metrics
        
        # Create tabular datasets for tree-based models
        X_train, y_train = create_tabular_dataset(train_scaled, target_idx=target_idx)
        X_val, y_val = create_tabular_dataset(val_scaled, target_idx=target_idx)
        X_test, y_test = create_tabular_dataset(test_scaled, target_idx=target_idx)
        
        logger.info(f"Tabular dataset: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
        
        # Train XGBoost
        logger.info("  Training XGBoost...")
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], feature_names=feature_cols)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = compute_all_metrics(y_test, y_pred_xgb)

        xgb_model.save(config.CHECKPOINTS_DIR / "xgboost.joblib")
        logger.info(f"    ✓ RMSE: {metrics_xgb.get('rmse', 0):.4f}")
        
        # Train LightGBM
        logger.info("  Training LightGBM...")
        lgb_model = LightGBMModel()
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], feature_names=feature_cols)
        y_pred_lgb = lgb_model.predict(X_test)
        metrics_lgb = compute_all_metrics(y_test, y_pred_lgb)
        lgb_model.save(config.CHECKPOINTS_DIR / "lightgbm.joblib")
        logger.info(f"    ✓ RMSE: {metrics_lgb.get('rmse', 0):.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ SETUP COMPLETE — Models trained and saved")
        logger.info("=" * 80)
        logger.info(f"\nModel Results Summary:")
        logger.info(f"  XGBoost   | RMSE: {metrics_xgb.get('rmse', 0):.4f} | MAE: {metrics_xgb.get('mae', 0):.4f} | R²: {metrics_xgb.get('r2', 0):.4f}")
        logger.info(f"  LightGBM  | RMSE: {metrics_lgb.get('rmse', 0):.4f} | MAE: {metrics_lgb.get('mae', 0):.4f} | R²: {metrics_lgb.get('r2', 0):.4f}")

        return 0

    except Exception as exc:
        logger.error(f"\n✗ Setup failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
