#!/usr/bin/env python
"""
Build the stacking ensemble from existing base model checkpoints.

Requires: transformer.pt, lstm.pt, xgboost.joblib, lightgbm.joblib, catboost.joblib

    python build_ensemble.py
"""

from __future__ import annotations

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
    try:
        path = TrainingOrchestrator().build_ensemble_from_checkpoints()
        logger.info("Done. Ensemble checkpoint: %s", path)
        return 0
    except Exception as exc:
        logger.error("Ensemble build failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
