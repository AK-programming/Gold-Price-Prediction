"""
Gold Price Forecasting System — Central Configuration
All hyperparameters, paths, feature definitions, and API settings.
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Ensure directories exist
for d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA INGESTION
# ============================================================
TICKERS = {
    "gold": "GC=F",
    "sp500": "^GSPC",
    "treasury_10y": "^TNX",
    "dollar_index": "DX-Y.NYB",
    "bitcoin": "BTC-USD",
    "crude_oil": "CL=F",
    "silver": "SI=F",
}

DATA_START_DATE = "2010-01-01"
DATA_END_DATE = "2026-05-20"

RAW_DATA_FILE = DATA_DIR / "raw_gold_data.parquet"
FEATURES_FILE = DATA_DIR / "features.parquet"
PROCESSED_FILE = DATA_DIR / "processed.parquet"
SCALER_FILE = DATA_DIR / "scalers.joblib"

# ============================================================
# FEATURE ENGINEERING
# ============================================================
SMA_WINDOWS = [7, 14, 30, 50, 200]
EMA_WINDOWS = [7, 14, 30]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
VOLATILITY_WINDOW = 20
LAG_PERIODS = [1, 3, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20]

# ============================================================
# SEQUENCE / WINDOWING
# ============================================================
INPUT_SEQUENCE_LENGTH = 60   # Past 60 days
OUTPUT_SEQUENCE_LENGTH = 10  # Predict next 10 days
STRIDE = 1

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# TRANSFORMER MODEL
# ============================================================
TRANSFORMER_CONFIG = {
    "hidden_size": 128,
    "num_attention_heads": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 2,
    "num_lstm_layers": 2,
    "dropout": 0.1,
    "quantiles": [0.1, 0.5, 0.9],
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "batch_size": 64,
    "max_epochs": 200,
    "early_stopping_patience": 15,
}

# ============================================================
# LSTM / GRU MODEL
# ============================================================
RNN_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 150,
    "early_stopping_patience": 15,
}

# ============================================================
# BOOSTING MODELS
# ============================================================
XGBOOST_CONFIG = {
    "max_depth": 8,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

LIGHTGBM_CONFIG = {
    "num_leaves": 63,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

CATBOOST_CONFIG = {
    "depth": 8,
    "iterations": 500,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "verbose": 0,
}

# ============================================================
# ENSEMBLE
# ============================================================
ENSEMBLE_CONFIG = {
    "meta_learner": "ridge",  # ridge | linear | mlp
    "cv_folds": 5,
}

# ============================================================
# OPTIMIZATION (OPTUNA)
# ============================================================
OPTUNA_CONFIG = {
    "n_trials": 50,
    "timeout": 3600,  # 1 hour max
    "direction": "minimize",
    "metric": "val_mape",
}

# ============================================================
# VALIDATION
# ============================================================
WALK_FORWARD_CONFIG = {
    "min_train_size": 252,  # ~1 year of trading days
    "step_size": 21,        # ~1 month
    "gap": 10,              # Prevent leakage
}

# ============================================================
# API
# ============================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Most hosting platforms (Render, Heroku, and it turns out HF Spaces too)
# inject a $PORT env var and expect the app to bind there, while their own
# proxy holds the conventional port (e.g. 7860) for incoming traffic. If we
# ignore $PORT and hardcode 7860, our process fights the platform's own
# listener for that port - which is exactly the "address already in use"
# crash seen on Spaces. Prefer $PORT when the platform sets it.
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "7860")))
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["*"]

# ============================================================
# TARGETS
# ============================================================
TARGET_COLUMN = "gold_Close"
PRICE_COLUMNS = ["gold_Open", "gold_High", "gold_Low", "gold_Close"]
