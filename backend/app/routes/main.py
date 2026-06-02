"""
Gold Price Forecasting System — Main FastAPI Application

Creates and configures the FastAPI app instance, registers middleware,
includes API routers, and optionally serves the frontend as static files.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.routes.predict import router as predict_router
from app.routes.model_info import router as model_info_router

# ------------------------------------------------------------------ #
# Logging configuration
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# FastAPI application
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Gold Price Forecasting API",
    description=(
        "REST API for the Gold Price Forecasting System. "
        "Provides endpoints for multi-horizon gold price predictions, "
        "model inspection, SHAP explainability, attention visualisation, "
        "and historical price retrieval."
    ),
    version="1.0.0",
)

# ------------------------------------------------------------------ #
# CORS middleware
# ------------------------------------------------------------------ #

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Router registration
# ------------------------------------------------------------------ #

app.include_router(predict_router)
app.include_router(model_info_router)

logger.info("Registered predict and model_info routers.")

# ------------------------------------------------------------------ #
# Root & health endpoints
# ------------------------------------------------------------------ #


@app.get("/", tags=["status"])
async def root() -> dict[str, str]:
    """Return basic API information.

    Returns
    -------
    dict
        JSON payload with API name, version, and documentation URL.
    """
    return {
        "api": "Gold Price Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health", tags=["status"])
async def health_check() -> dict[str, str]:
    """Lightweight health-check endpoint for load balancers and probes.

    Returns
    -------
    dict
        ``{"status": "healthy"}``
    """
    return {"status": "healthy"}


# ------------------------------------------------------------------ #
# Static file serving (frontend)
# ------------------------------------------------------------------ #

if config.FRONTEND_DIR.exists() and config.FRONTEND_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(config.FRONTEND_DIR)),
        name="static",
    )
    logger.info("Serving frontend static files from %s", config.FRONTEND_DIR)
else:
    logger.info(
        "Frontend directory '%s' not found — static file serving disabled.",
        config.FRONTEND_DIR,
    )
