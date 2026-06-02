"""
Transformer-based model for gold price time-series forecasting.

Architecture:
    Input Projection → Positional Encoding → TransformerEncoder
    → LSTM bridge → TransformerDecoder → Quantile Output Heads

Outputs three quantile forecasts (q10, q50, q90) for probabilistic prediction.
"""

import copy
import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs.

    Injects information about the relative or absolute position of tokens
    in the sequence using sine and cosine functions of different frequencies.

    Args:
        d_model: Embedding / hidden dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ------------------------------------------------------------------
# Transformer Model
# ------------------------------------------------------------------
class GoldTransformer(nn.Module):
    """Hybrid Transformer–LSTM model for multi-horizon gold price forecasting.

    The architecture combines:
    * A linear input projection from raw feature dimension to ``hidden_size``.
    * Sinusoidal positional encoding.
    * A multi-layer TransformerEncoder to learn temporal patterns.
    * A bidirectional LSTM bridge for sequential smoothing.
    * A TransformerDecoder that attends over the encoded memory.
    * Three independent quantile heads producing quantile forecasts
      at the 10th, 50th, and 90th percentiles.

    Args:
        num_features: Number of input features per time step.
        hidden_size: Model / embedding dimension throughout.
        num_attention_heads: Number of attention heads in encoder/decoder.
        num_encoder_layers: Number of TransformerEncoder layers.
        num_decoder_layers: Number of TransformerDecoder layers.
        num_lstm_layers: Number of LSTM layers in the bridge.
        output_seq_len: Number of future time-steps to predict.
        dropout: Dropout probability.
        quantiles: List of quantile levels for the output heads.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = config.TRANSFORMER_CONFIG["hidden_size"],
        num_attention_heads: int = config.TRANSFORMER_CONFIG["num_attention_heads"],
        num_encoder_layers: int = config.TRANSFORMER_CONFIG["num_encoder_layers"],
        num_decoder_layers: int = config.TRANSFORMER_CONFIG["num_decoder_layers"],
        num_lstm_layers: int = config.TRANSFORMER_CONFIG["num_lstm_layers"],
        output_seq_len: int = config.OUTPUT_SEQUENCE_LENGTH,
        dropout: float = config.TRANSFORMER_CONFIG["dropout"],
        quantiles: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.quantiles = quantiles or config.TRANSFORMER_CONFIG["quantiles"]
        self._last_encoder_attn_weights: Optional[torch.Tensor] = None

        # --- Input projection ---
        self.input_projection = nn.Linear(num_features, hidden_size)

        # --- Positional encoding ---
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # --- LSTM bridge ---
        self.lstm_bridge = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_projection = nn.Linear(hidden_size * 2, hidden_size)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # --- Decoder input embedding (learnable start tokens) ---
        self.decoder_embedding = nn.Parameter(
            torch.randn(1, output_seq_len, hidden_size) * 0.02
        )
        self.pos_decoder = PositionalEncoding(hidden_size, dropout=dropout)

        # --- Quantile output heads ---
        self.quantile_heads = nn.ModuleDict(
            {
                f"q{int(q * 100)}": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )
                for q in self.quantiles
            }
        )

        self._init_weights()

    # ----- weight initialisation -----
    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to linear layers."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # ----- causal mask helper -----
    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Generate an upper-triangular causal mask for the decoder."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    # ----- forward -----
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            src: Input tensor of shape ``(batch, input_seq_len, num_features)``.
            src_key_padding_mask: Optional mask of shape ``(batch, input_seq_len)``;
                ``True`` positions are ignored.

        Returns:
            Dict with keys ``'q10'``, ``'q50'``, ``'q90'``, each of shape
            ``(batch, output_seq_len)``.
        """
        batch_size = src.size(0)

        # 1. Project input features → hidden_size
        x = self.input_projection(src)  # (B, S, H)
        x = self.pos_encoder(x)

        # 2. Transformer Encoder
        memory = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, S, H)

        # Capture attention weights from the last encoder layer for interpretability
        self._capture_attention_weights(x, src_key_padding_mask)

        # 3. LSTM bridge
        lstm_out, _ = self.lstm_bridge(memory)  # (B, S, 2H)
        memory = self.lstm_projection(lstm_out)  # (B, S, H)

        # 4. Decoder
        tgt = self.decoder_embedding.expand(batch_size, -1, -1)  # (B, T, H)
        tgt = self.pos_decoder(tgt)
        tgt_mask = self._generate_causal_mask(self.output_seq_len, device=src.device)
        decoded = self.transformer_decoder(
            tgt, memory, tgt_mask=tgt_mask
        )  # (B, T, H)

        # 5. Quantile outputs
        outputs: Dict[str, torch.Tensor] = {}
        for key, head in self.quantile_heads.items():
            out = head(decoded).squeeze(-1)  # (B, T)
            outputs[key] = out

        return outputs

    # ----- attention weight capture -----
    @torch.no_grad()
    def _capture_attention_weights(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> None:
        """Run the last encoder layer's self-attention to capture weights.

        This is called during ``forward()`` and stored for later retrieval
        via ``get_attention_weights()``.
        """
        last_layer = self.transformer_encoder.layers[-1]
        self_attn = last_layer.self_attn
        attn_output, attn_weights = self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self._last_encoder_attn_weights = attn_weights.detach().cpu()

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last captured encoder attention weights.

        Returns:
            Tensor of shape ``(batch, num_heads, seq_len, seq_len)`` or
            ``None`` if no forward pass has been executed yet.
        """
        return self._last_encoder_attn_weights


# ------------------------------------------------------------------
# Quantile (Pinball) Loss
# ------------------------------------------------------------------
def quantile_loss(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    quantiles: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute combined pinball (quantile) loss over multiple quantile heads.

    Args:
        predictions: Dict mapping quantile keys (e.g. ``'q10'``, ``'q50'``,
            ``'q90'``) to tensors of shape ``(batch, output_seq_len)``.
        targets: Ground-truth tensor of shape ``(batch, output_seq_len)``.
        quantiles: List of quantile levels corresponding to the prediction keys.

    Returns:
        Scalar loss tensor (mean over all quantiles, batch, and time-steps).
    """
    quantiles = quantiles or config.TRANSFORMER_CONFIG["quantiles"]
    total_loss = torch.tensor(0.0, device=targets.device)

    for q in quantiles:
        key = f"q{int(q * 100)}"
        pred = predictions[key]
        errors = targets - pred
        loss = torch.max(q * errors, (q - 1.0) * errors)
        total_loss = total_loss + loss.mean()

    return total_loss / len(quantiles)


# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train_transformer(
    model: GoldTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train the GoldTransformer with early stopping and checkpointing.

    Uses:
    * AdamW optimiser with ``learning_rate`` and ``weight_decay`` from config.
    * CosineAnnealingLR scheduler.
    * Gradient clipping at ``max_grad_norm``.
    * Early stopping with ``early_stopping_patience``.

    Args:
        model: ``GoldTransformer`` instance (already on the target device).
        train_loader: DataLoader yielding ``(X, y)`` batches where
            ``X`` has shape ``(batch, seq_len, num_features)`` and
            ``y`` has shape ``(batch, output_seq_len)``.
        val_loader: Validation DataLoader with same format.
        cfg: Optional override dict; defaults to ``config.TRANSFORMER_CONFIG``.
        device: PyTorch device; defaults to CUDA if available.

    Returns:
        Dictionary with keys ``'train_loss'``, ``'val_loss'``, ``'lr'`` each
        containing a list of per-epoch values.
    """
    cfg = cfg or config.TRANSFORMER_CONFIG
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"], eta_min=1e-6)

    quantiles = cfg.get("quantiles", config.TRANSFORMER_CONFIG["quantiles"])
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    patience = cfg.get("early_stopping_patience", 15)
    max_epochs = cfg.get("max_epochs", 200)

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
            loss = quantile_loss(preds, y_batch, quantiles)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
                loss = quantile_loss(preds, y_val, quantiles)
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

        scheduler.step()

    # --- Restore best model & save checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = config.CHECKPOINT_DIR / "transformer_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        checkpoint_path,
    )
    logger.info("Best transformer checkpoint saved to %s", checkpoint_path)

    return history
