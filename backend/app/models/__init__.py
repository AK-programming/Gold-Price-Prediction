# Model architecture modules
"""
Public API for the ``app.models`` package.

Exports all model classes, training functions, and loss utilities so that
downstream code can do::

    from app.models import GoldTransformer, GoldLSTM, StackingEnsemble
"""

from app.models.transformer_model import (
    PositionalEncoding,
    GoldTransformer,
    quantile_loss,
    train_transformer,
)
from app.models.rnn_model import (
    Attention,
    GoldLSTM,
    train_rnn,
)
from app.models.boosting_models import (
    BaseBoostingModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
)
from app.models.ensemble import StackingEnsemble

__all__ = [
    # Transformer
    "PositionalEncoding",
    "GoldTransformer",
    "quantile_loss",
    "train_transformer",
    # LSTM
    "Attention",
    "GoldLSTM",
    "train_rnn",
    # Boosting
    "BaseBoostingModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    # Ensemble
    "StackingEnsemble",
]
