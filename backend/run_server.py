"""
Gold Price Forecasting System — Server Entry Point

Starts the uvicorn ASGI server pointing at the main FastAPI application.
Run with:  ``python run_server.py``
"""

from __future__ import annotations

import logging
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
    uvicorn.run(
        "app.routes.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
