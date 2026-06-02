"""
Gold Price Forecasting System — Attention Visualisation

Extracts and post-processes attention weights from Transformer and
LSTM-with-attention models for temporal importance analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app import config

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Extract, aggregate, and serialise attention weights.

    Attributes
    ----------
    raw_attention : Optional[np.ndarray]
        Raw attention weights from the latest ``extract_attention`` call.
    temporal_importance : Optional[np.ndarray]
        Per-time-step importance derived from attention weights.
    """

    def __init__(self) -> None:
        self.raw_attention: Optional[np.ndarray] = None
        self.temporal_importance: Optional[np.ndarray] = None
        logger.info("AttentionVisualizer initialised.")

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    def extract_attention(
        self, model: Any, X_sample: np.ndarray
    ) -> np.ndarray:
        """Run a forward pass and collect attention weights.

        The model is expected to expose one of the following:
        * A method ``get_attention_weights(x)`` that returns a NumPy array
          of shape ``(n_samples, n_heads, seq_len, seq_len)`` or similar.
        * A PyTorch hook-based approach when the above is unavailable.

        Parameters
        ----------
        model : Any
            A PyTorch ``nn.Module`` with an attention mechanism.
        X_sample : np.ndarray
            Input data, shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        np.ndarray
            Attention weights array (varies by architecture).
        """
        try:
            import torch

            model.eval()
            x_tensor = torch.tensor(X_sample, dtype=torch.float32)

            # Strategy 1: model explicitly provides attention weights
            if hasattr(model, "get_attention_weights"):
                with torch.no_grad():
                    attn = model.get_attention_weights(x_tensor)
                self.raw_attention = (
                    attn.cpu().numpy() if isinstance(attn, torch.Tensor) else np.asarray(attn)
                )
                logger.info(
                    "Extracted attention via get_attention_weights — shape %s.",
                    self.raw_attention.shape,
                )
                return self.raw_attention

            # Strategy 2: register forward hooks on MultiheadAttention layers
            attention_outputs: List[np.ndarray] = []

            def _hook_fn(
                module: torch.nn.Module,
                inp: Any,
                out: Any,
            ) -> None:
                # MultiheadAttention returns (attn_output, attn_weights)
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    attention_outputs.append(out[1].detach().cpu().numpy())

            hooks = []
            for submodule in model.modules():
                if isinstance(submodule, torch.nn.MultiheadAttention):
                    hooks.append(submodule.register_forward_hook(_hook_fn))

            if not hooks:
                logger.warning(
                    "Model has no MultiheadAttention layers and no "
                    "get_attention_weights method — generating uniform attention."
                )
                seq_len = X_sample.shape[1]
                self.raw_attention = np.full(
                    (X_sample.shape[0], 1, seq_len, seq_len),
                    1.0 / seq_len,
                )
                return self.raw_attention

            with torch.no_grad():
                model(x_tensor)

            for h in hooks:
                h.remove()

            if attention_outputs:
                # Stack across layers → (n_layers, n_samples, n_heads, seq, seq)
                self.raw_attention = np.stack(attention_outputs, axis=0)
                logger.info(
                    "Extracted attention via hooks — shape %s.", self.raw_attention.shape
                )
            else:
                seq_len = X_sample.shape[1]
                self.raw_attention = np.full(
                    (X_sample.shape[0], 1, seq_len, seq_len),
                    1.0 / seq_len,
                )
                logger.warning("Hooks registered but no attention captured; using uniform.")

            return self.raw_attention

        except Exception as exc:
            logger.error("Attention extraction failed: %s", exc, exc_info=True)
            seq_len = X_sample.shape[1] if X_sample.ndim >= 2 else config.INPUT_SEQUENCE_LENGTH
            self.raw_attention = np.full(
                (X_sample.shape[0], 1, seq_len, seq_len),
                1.0 / seq_len,
            )
            return self.raw_attention

    # ------------------------------------------------------------------ #
    # Temporal importance
    # ------------------------------------------------------------------ #

    def get_temporal_importance(
        self, attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Average attention across heads (and optionally layers) to get per-step importance.

        Parameters
        ----------
        attention_weights : np.ndarray, optional
            Raw attention array. If *None*, uses ``self.raw_attention``.

        Returns
        -------
        np.ndarray
            Shape ``(seq_len,)`` — importance score per input time step.
        """
        attn = attention_weights if attention_weights is not None else self.raw_attention
        if attn is None:
            logger.warning("No attention weights available; returning empty array.")
            return np.array([])

        attn = np.asarray(attn, dtype=np.float64)

        # Collapse until we have (seq_len, seq_len) or (seq_len,)
        while attn.ndim > 2:
            attn = attn.mean(axis=0)

        # If (seq_len, seq_len) — average over the key dimension (axis=1)
        if attn.ndim == 2:
            importance = attn.mean(axis=0)
        else:
            importance = attn

        # Normalise to sum to 1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        self.temporal_importance = importance
        logger.info("Temporal importance computed — %d steps.", importance.shape[0])
        return importance

    # ------------------------------------------------------------------ #
    # JSON serialisation
    # ------------------------------------------------------------------ #

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the attention analysis.

        Returns
        -------
        dict
            Keys:
            * ``temporal_importance`` — list of per-step importance floats.
            * ``sequence_length`` — number of input time steps.
            * ``attention_shape`` — shape of the raw attention array.
        """
        if self.temporal_importance is None and self.raw_attention is not None:
            self.get_temporal_importance()

        temporal = (
            self.temporal_importance.tolist()
            if self.temporal_importance is not None
            else []
        )
        attn_shape = (
            list(self.raw_attention.shape)
            if self.raw_attention is not None
            else []
        )

        return {
            "temporal_importance": temporal,
            "sequence_length": len(temporal),
            "attention_shape": attn_shape,
            "input_window": config.INPUT_SEQUENCE_LENGTH,
        }
