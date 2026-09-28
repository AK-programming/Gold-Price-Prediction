"""
Gold Price Forecasting System — Server Entry Point

Starts the uvicorn ASGI server pointing at the main FastAPI application.
Run with:  ``python run_server.py``
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from app import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(
        "Starting Gold Price Forecasting API on %s:%s",
        config.API_HOST,
        config.API_PORT,
    )
    # Auto-reload is handy for local dev but fatal on a hosted platform: any
    # file the platform itself touches after boot (pip metadata, a health
    # probe writing a log, etc.) trips the watcher and reload kills the
    # server a few seconds after startup. Hugging Face Spaces (and most
    # other hosts) set SPACE_ID / PORT-style env vars we can key off of, so
    # only enable reload when neither is present, i.e. running locally.
    is_hosted = bool(os.getenv("SPACE_ID") or os.getenv("RENDER") or os.getenv("DYNO"))
    uvicorn.run(
        "app.routes.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=not is_hosted,
    )
