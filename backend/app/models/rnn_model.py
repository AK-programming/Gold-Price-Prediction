"""
Bidirectional LSTM with Bahdanau attention for gold price forecasting.

Architecture:
    Input → Bidirectional LSTM → Attention → FC Output

Produces point predictions for multi-step horizons.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from app import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Bahdanau-style Attention
# ------------------------------------------------------------------
class Attention(nn.Module):
    """Bahdanau (additive) attention mechanism.

    Computes a context vector as a weighted sum of encoder hidden states,
    where the weights are learned via a small feed-forward network.

    Args:
        hidden_size: Dimensionality of the hidden states the attention
            operates over (after any bidirectional concatenation).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            query: Decoder query of shape ``(batch, 1, hidden_size)`` or the
                final hidden state reshaped to ``(batch, 1, hidden_size)``.
            keys: Encoder outputs of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Tuple of:
            - context: weighted sum ``(batch, 1, hidden_size)``
            - weights: attention distribution ``(batch, seq_len)``
        """
        # Additive scoring
        scores = self.V(
            torch.tanh(self.W_query(query) + self.W_key(keys))
        )  # (B, S, 1)
        scores = scores.squeeze(-1)  # (B, S)
        weights = F.softmax(scores, dim=-1)  # (B, S)
        self._attention_weights = weights.detach().cpu()

        # Context vector
        context = torch.bmm(weights.unsqueeze(1), keys)  # (B, 1, H)
        return context, weights


# ------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------
class GoldLSTM(nn.Module):
    """Bidirectional LSTM with Bahdanau attention for gold price forecasting.

    Args:
        num_features: Number of input features per time step.
        hidden_size: LSTM hidden dimension per direction.
        num_layers: Number of stacked LSTM layers.
        output_seq_len: Number of future time-steps to predict.
        dropout: Dropout probability between LSTM layers.
        bidirectional: Whether the LSTM is bidirectional.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = config.RNN_CONFIG["hidden_size"],
        num_layers: int = config.RNN_CONFIG["num_layers"],
        output_seq_len: int = config.OUTPUT_SEQUENCE_LENGTH,
        dropout: float = config.RNN_CONFIG["dropout"],
        bidirectional: bool = config.RNN_CONFIG["bidirectional"],
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.effective_hidden = hidden_size * self.num_directions

        # --- Layers ---
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.attention = Attention(self.effective_hidden)

        self.layer_norm = nn.LayerNorm(self.effective_hidden)
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Sequential(
            nn.Linear(self.effective_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, num_features)``.

        Returns:
            Tensor of shape ``(batch, output_seq_len)`` with point predictions.
        """
        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (B, S, H*D)

        # Construct query from the final hidden state
        if self.bidirectional:
            # h_n shape: (num_layers * num_directions, B, H)
            # Take the last layer's forward and backward hidden states
            h_forward = h_n[-2]  # (B, H)
            h_backward = h_n[-1]  # (B, H)
            query = torch.cat([h_forward, h_backward], dim=-1)  # (B, H*2)
        else:
            query = h_n[-1]  # (B, H)

        query = query.unsqueeze(1)  # (B, 1, H*D)

        # Attention
        context, _ = self.attention(query, lstm_out)  # (B, 1, H*D)
        context = context.squeeze(1)  # (B, H*D)
        context = self.layer_norm(context)
        context = self.dropout(context)

        # Output projection
        output = self.fc_out(context)  # (B, output_seq_len)
        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights.

        Returns:
            Tensor of shape ``(batch, seq_len)`` or ``None``.
        """
        return self.attention._attention_weights


# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train_rnn(
    model: GoldLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train the GoldLSTM with early stopping and checkpointing.

    Uses:
    * Adam optimiser with ``learning_rate`` and ``weight_decay`` from config.
    * ReduceLROnPlateau scheduler monitoring validation loss.
    * MSE loss.
    * Early stopping with ``early_stopping_patience``.

    Args:
        model: ``GoldLSTM`` instance.
        train_loader: DataLoader yielding ``(X, y)`` batches where
            ``X`` has shape ``(batch, seq_len, num_features)`` and
            ``y`` has shape ``(batch, output_seq_len)``.
        val_loader: Validation DataLoader with same format.
        cfg: Optional override dict; defaults to ``config.RNN_CONFIG``.
        device: PyTorch device; defaults to CUDA if available.

    Returns:
        Dictionary with keys ``'train_loss'``, ``'val_loss'``, ``'lr'`` each
        containing a list of per-epoch values.
    """
    cfg = cfg or config.RNN_CONFIG
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    patience = cfg.get("early_stopping_patience", 15)
    max_epochs = cfg.get("max_epochs", 150)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state: Optional[dict] = None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        train_losses: List[float] = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # --- Validation ---
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch, max_epochs, avg_train_loss, avg_val_loss, current_lr,
        )

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch, patience,
                )
                break

        scheduler.step(avg_val_loss)

    # --- Restore best model & save checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = config.CHECKPOINT_DIR / "lstm_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        checkpoint_path,
    )
    logger.info("Best LSTM checkpoint saved to %s", checkpoint_path)

    return history
