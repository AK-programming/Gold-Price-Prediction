"""
Gold Price Forecasting System — API Routes Package

Re-exports the ``predict`` and ``model_info`` routers so consumers can
do ``from app.routes import predict_router, model_info_router``.
"""

from app.routes.model_info import router as model_info_router
from app.routes.predict import router as predict_router

__all__: list[str] = ["predict_router", "model_info_router"]
