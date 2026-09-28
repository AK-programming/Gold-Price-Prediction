#!/usr/bin/env python
"""
Gold Price Forecasting System — Setup & Training

Full pipeline (download → features → train all models → metrics JSON):
    python train_and_setup.py

Skip re-download when raw data already exists:
    python train_and_setup.py

Force fresh Yahoo Finance download:
    python train_and_setup.py --refresh-data

Evaluate existing checkpoints only (no retraining):
    python eval_models.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.utils.trainer import TrainingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all gold forecasting models.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download market data even if cached raw parquet exists.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Gold Price Forecasting — Training Pipeline")
    logger.info("=" * 80)

    try:
        orchestrator = TrainingOrchestrator()
        orchestrator.run_full_pipeline(refresh_data=args.refresh_data)
        logger.info("Training complete. Metrics written to backend/logs/metrics_*.json")
        return 0
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
