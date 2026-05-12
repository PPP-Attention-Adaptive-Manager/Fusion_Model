"""
FixedTCNEncoder — frozen TCN feature extractor.

Takes a raw per-timestep sequence (B, T, d_in), passes it through a stack
of TemporalBlocks built from a TCNEncoderConfig, pools over T, returns
a fixed-size embedding (B, n_channels).

"Fixed" means: weights are initialised once and permanently frozen.
No gradients ever flow through this module. It is a deterministic
feature extractor, not a trained model. The trained parts are the
predictive_models downstream.

Usage:
    from TCN_encoders.configs import multiscale
    enc = FixedTCNEncoder(d_in=7, cfg=multiscale)
    emb = enc(x)   # x: (B, T, 7)  →  emb: (B, 64)
"""

import torch
import torch.nn as nn
from .tcn_block import TemporalBlock
from .configs   import TCNEncoderConfig


class FixedTCNEncoder(nn.Module):
    """
    Args:
        d_in : number of input features per timestep (modality-specific)
        cfg  : TCNEncoderConfig — controls depth, width, kernel, dilation, pool
    """

    def __init__(self, d_in: int, cfg: TCNEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input projection: d_in → n_channels (handles arbitrary input width)
        self.input_proj = nn.Conv1d(d_in, cfg.n_channels, kernel_size=1, bias=False)

        # Stack of TemporalBlocks with exponentially increasing dilation
        blocks = []
        for i in range(cfg.n_layers):
            dilation = cfg.dilation_base ** i
            blocks.append(
                TemporalBlock(
                    in_channels  = cfg.n_channels,
                    out_channels = cfg.n_channels,
                    kernel_size  = cfg.kernel_size,
                    dilation     = dilation,
                    dropout      = cfg.dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Freeze everything immediately after init
        self._freeze()

    def _freeze(self) -> None:
        """Permanently disable gradient computation for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_in)   — raw per-timestep sequence
        Returns:
            (B, n_channels)   — fixed embedding, T collapsed
        """
        # Conv1d expects (B, C, T) — transpose in, transpose out
        out = self.input_proj(x.transpose(1, 2))   # (B, n_channels, T)
        out = self.blocks(out)                      # (B, n_channels, T)

        # Pool over T → (B, n_channels)
        if self.cfg.pool == "last":
            return out[:, :, -1]
        elif self.cfg.pool == "mean":
            return out.mean(dim=-1)
        elif self.cfg.pool == "max":
            return out.max(dim=-1).values
        else:
            raise ValueError(f"Unknown pool mode: '{self.cfg.pool}'")

    def receptive_field(self) -> int:
        """Returns how many past timesteps this encoder can see."""
        return self.cfg.receptive_field()