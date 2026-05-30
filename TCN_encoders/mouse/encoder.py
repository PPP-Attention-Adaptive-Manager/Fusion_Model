"""
MouseBufferedEncoder — passthrough encoder.

Mouse pre_embedder (MouseEncoderP2) already runs a frozen TCN internally
and outputs m_t (1, 128) every second. There is nothing left to encode.

This class exists purely for interface consistency — fusion_model.py
calls encoder.step(raw) on all 4 modalities uniformly. Mouse just
happens to have a trivial implementation.

No buffer. No TCN. No staleness (mouse is always fresh at 1Hz).
Freshness is always 1.0.

Expected input to step():
    m_t : torch.Tensor of shape (1, 128)  — from MouseEncoderP2
"""

import torch
from typing import Optional, Tuple
import numpy as np


class MouseBufferedEncoder:
    """
    Pure passthrough. Does not subclass BufferedEncoder because
    it bypasses the buffer and TCN entirely.

    Args:
        d_out : expected embedding dimension (default 128, must match MouseEncoderP2)
    """

    def __init__(self, d_out: int = 64) -> None:
        self.d_out = d_out

    def step(self, raw_input: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            raw_input : (1, 128) tensor from MouseEncoderP2
        Returns:
            embedding  : (1, 128) — unchanged
            freshness  : 1.0      — always fresh
        """
        assert raw_input.shape[-1] == self.d_out, \
            f"MouseBufferedEncoder expected d_out={self.d_out}, got {raw_input.shape[-1]}"
        return raw_input, 1.0

    def reset(self) -> None:
        """No state to reset. Exists for interface consistency."""
        pass

    def _parse_input(self, raw) -> Optional[np.ndarray]:
        """Not used — step() is overridden entirely."""
        return None