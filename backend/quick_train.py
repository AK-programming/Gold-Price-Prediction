#!/usr/bin/env python
"""
Gold Price Forecasting — Quick Start Training Script

This fast version trains only the XGBoost model (fastest to train) to:
1. Get something in the checkpoints directory 
2. Populate metrics for the API
3. Get the dashboard showing real data immediately

For full training with all 6 models, see train_and_setup.py

Run with:  ``python quick_train.py``
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Execute quick training with XGBoost only."""
    logger.info("=" * 80)
    logger.info("Quick Start — Training XGBoost Model (fastest)")
    logger.info("=" * 80)

    try:
        # Step 1: Load features
        from app import config
        import pandas as pd
        
        if not config.FEATURES_FILE.exists():
            logger.error("Features file not found. Run train_and_setup.py first.")
            return 1

        logger.info("\n[1/3] LOADING FEATURES...")
        features_df = pd.read_parquet(config.FEATURES_FILE)
        logger.info(f"✓ Loaded {features_df.shape[0]} rows × {features_df.shape[1]} columns")

        # Step 2: Preprocess data
        logger.info("\n[2/3] PREPROCESSING DATA...")
        from app.pipeline.preprocessor import preprocess_data, create_tabular_dataset
        from app.utils.metrics import compute_all_metrics
        
        train_scaled, val_scaled, test_scaled, feature_cols, target_idx, scaler = preprocess_data(features_df)
        logger.info(f"✓ Train: {train_scaled.shape}, Val: {val_scaled.shape}, Test: {test_scaled.shape}")
        
        # Create tabular (non-sequential) dataset for tree-based models
        X_train, y_train = create_tabular_dataset(train_scaled, target_idx=target_idx)
        X_val, y_val = create_tabular_dataset(val_scaled, target_idx=target_idx)
        X_test, y_test = create_tabular_dataset(test_scaled, target_idx=target_idx)
        logger.info(f"✓ Tabular dataset: X_train {X_train.shape}, y_train {y_train.shape}")

        # Step 3: Train XGBoost
        logger.info("\n[3/3] TRAINING XGBOOST...")
        from app.models.boosting_models import XGBoostModel
        
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], feature_names=feature_cols)
        
        # Predict on all sets
        y_pred_train = xgb_model.predict(X_train)
        y_pred_val = xgb_model.predict(X_val)
        y_pred_test = xgb_model.predict(X_test)
        
        # Compute metrics
        metrics = compute_all_metrics(
            y_test, y_pred_test, 
            y_val, y_pred_val, 
            y_train, y_pred_train
        )
        
        # Save model
        xgb_model.save(config.CHECKPOINTS_DIR / "xgboost.joblib")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ QUICK START COMPLETE")
        logger.info("=" * 80)
        logger.info(f"XGBoost Test Metrics:")
        logger.info(f"  RMSE: {metrics.get('test_rmse', 0):.4f}")
        logger.info(f"  MAE:  {metrics.get('test_mae', 0):.4f}")
        logger.info(f"  R²:   {metrics.get('test_r2', 0):.4f}")
        logger.info("\n✓ Restart the frontend dev server to see real data on the dashboard")

        return 0

    except Exception as exc:
        logger.error(f"\n✗ Quick start failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
