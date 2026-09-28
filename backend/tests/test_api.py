"""Smoke tests for the Gold Price Forecasting API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config
from app.routes.main import app
from app.utils.model_selection import checkpoint_exists

client = TestClient(app)

_HAS_MODEL = any(
    checkpoint_exists(name)
    for name in ("transformer", "lstm", "xgboost", "lightgbm", "catboost")
)
_HAS_DATA = config.FEATURES_FILE.exists() or config.RAW_DATA_FILE.exists()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "Gold Price Forecasting API"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "data" in body
    assert "model_checkpoint" in body


@pytest.mark.skipif(not _HAS_DATA, reason="No feature or raw data on disk")
def test_historical():
    response = client.get("/api/v1/historical")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert len(body["data"]) == body["count"]
    point = body["data"][0]
    assert "date" in point
    assert "actual" in point


@pytest.mark.skipif(
    not (_HAS_MODEL and _HAS_DATA and config.SCALER_FILE.exists()),
    reason="Forecast requires checkpoints, features, and scaler",
)
def test_forecast_shape():
    response = client.get("/api/v1/forecast")
    assert response.status_code == 200
    body = response.json()
    horizon = len(body["predictions"])
    assert horizon > 0
    assert len(body["dates"]) == horizon
    assert len(body["confidence_low"]) == horizon
    assert len(body["confidence_high"]) == horizon
    assert body["model_used"] in {
        "transformer",
        "lstm",
        "xgboost",
        "lightgbm",
        "catboost",
    }


def test_data_status():
    response = client.get("/api/v1/data/status")
    assert response.status_code == 200
    body = response.json()
    assert "last_data_date" in body
    assert "last_close_usd" in body
    assert body["last_close_usd"] > 0


def test_models_list():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert len(body["models"]) >= 1
    names = {m["name"] for m in body["models"]}
    assert "transformer" in names
