"""
Gold Price Forecasting System — Data Pipeline Package

Re-exports the public API from each sub-module for convenient access::

    from app.pipeline import download_all_data, engineer_features, preprocess_data
"""

from app.pipeline.data_ingestion import download_all_data, load_raw_data
from app.pipeline.feature_engineering import engineer_features, load_features
from app.pipeline.preprocessor import (
    GoldDataset,
    create_sequences,
    create_tabular_dataset,
    get_dataloaders,
    inverse_transform_predictions,
    preprocess_data,
)

__all__ = [
    # data_ingestion
    "download_all_data",
    "load_raw_data",
    # feature_engineering
    "engineer_features",
    "load_features",
    # preprocessor
    "preprocess_data",
    "create_sequences",
    "create_tabular_dataset",
    "GoldDataset",
    "get_dataloaders",
    "inverse_transform_predictions",
]
