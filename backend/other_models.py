"""
Legacy training script — prefer train_and_setup.py or eval_models.py.

    python train_and_setup.py      # full pipeline
    python eval_models.py          # metrics only
"""

import logging
import pandas as pd
import torch

from app import config
from app.pipeline.preprocessor import preprocess_data, create_tabular_dataset, get_dataloaders
from app.models import CatBoostModel, GoldLSTM, train_rnn, GoldTransformer, train_transformer

logging.basicConfig(level=logging.INFO)

features_df = pd.read_parquet(config.FEATURES_FILE)

train_scaled, val_scaled, test_scaled, feature_cols, target_idx, scaler = preprocess_data(features_df)

X_train, y_train = create_tabular_dataset(train_scaled, target_idx=target_idx)
X_val, y_val = create_tabular_dataset(val_scaled, target_idx=target_idx)

flat_feature_cols = [
    f"{col}_t{step}"
    for step in range(config.INPUT_SEQUENCE_LENGTH)
    for col in feature_cols
]

print("Training CatBoost...")
cat_model = CatBoostModel()
cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], feature_names=flat_feature_cols)
cat_model.save(config.CHECKPOINT_DIR / "catboost.joblib")

print("Training LSTM...")
loaders = get_dataloaders(train_scaled, val_scaled, test_scaled, target_idx=target_idx)
num_features = len(feature_cols)

lstm = GoldLSTM(num_features=num_features)
train_rnn(lstm, loaders["train"], loaders["val"])
torch.save(
    {"model_state_dict": lstm.state_dict(), "num_features": num_features, "feature_cols": feature_cols},
    config.CHECKPOINT_DIR / "lstm.pt",
)

print("Training Transformer...")
transformer = GoldTransformer(num_features=num_features)
train_transformer(transformer, loaders["train"], loaders["val"])
torch.save(
    {"model_state_dict": transformer.state_dict(), "num_features": num_features, "feature_cols": feature_cols},
    config.CHECKPOINT_DIR / "transformer.pt",
)

print("Remaining model training complete.")