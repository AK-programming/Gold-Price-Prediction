"""
Gold Price Forecasting System — Main FastAPI Application

Creates and configures the FastAPI app instance, registers middleware,
and includes API routers.
"""

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.data import router as data_router
from app.routes.predict import router as predict_router
from app.routes.model_info import router as model_info_router
from app.utils.model_selection import readiness_check

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
# CORS middleware — allow frontend on any port to connect
# ------------------------------------------------------------------ #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Router registration
# ------------------------------------------------------------------ #

app.include_router(predict_router)
app.include_router(model_info_router)
app.include_router(data_router)

logger.info("Registered predict and model_info routers.")

# ------------------------------------------------------------------ #
# Root & health endpoints
# ------------------------------------------------------------------ #

# Hugging Face Spaces sets SPACE_ID for every running Space. Its Gradio-SDK
# supervisor expects a live Gradio interface at "/" to consider the app
# healthy - serving plain JSON there instead makes it decide the app never
# became ready and kill it a few seconds after startup. So on a Space we
# mount a trivial status page at "/" instead of the JSON root below; every
# other endpoint (/health, /docs, /api/v1/...) is unaffected either way.
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))

if not IS_HF_SPACE:

    @app.get("/", tags=["status"])
    async def root() -> dict[str, str]:
        """Return basic API information."""
        return {
            "api": "Gold Price Forecasting API",
            "version": "1.0.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }


@app.get("/health", tags=["status"])
async def health_check() -> dict:
    """Health check with data, scaler, and model readiness."""
    return readiness_check()


if IS_HF_SPACE:
    import gradio as gr

    with gr.Blocks(title="GoldSight AI Backend") as _status_ui:
        gr.Markdown(
            "## GoldSight AI — Backend is running\n\n"
            "This Space serves the FastAPI backend for GoldSight AI.\n\n"
            "- API docs: [/docs](/docs)\n"
            "- Health check: [/health](/health)"
        )
    app = gr.mount_gradio_app(app, _status_ui, path="/")
