#!/usr/bin/env python
"""Evaluate saved checkpoints and write metrics JSON (no retraining)."""

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
        path = TrainingOrchestrator().evaluate_existing_checkpoints()
        logger.info("Metrics saved to %s", path)
        return 0
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
