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

        if cfg.filter_init != "random":
            self._apply_filter_init()

        # Freeze everything immediately after init
        self._freeze()

    def _freeze(self) -> None:
        """Permanently disable gradient computation for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def _apply_filter_init(self) -> None:
        """
        Overwrite the first TemporalBlock's conv1 kernel weights with a
        principled temporal filter bank, before _freeze() locks them.

        Target: self.blocks[0].conv1.conv  — CausalConv1d inside TemporalBlock
        Weight shape: (C_out, C_in, kernel_size) where C_out == C_in == n_channels

        Diagonal structure: W[ch, ch, :] = filter_weights, W[i, j!=i, :] = 0
        Each output channel operates on exactly one input channel.
        Block assignment used (not cyclic) — see ADR note in configs.py.

        Filter kernel orientation (causal, kernel_size=3):
            position 0 = t-2 (oldest)
            position 1 = t-1
            position 2 = t   (most recent)
        """
        C   = self.cfg.n_channels
        k   = self.cfg.kernel_size     # always 3 for narrow and shallow presets
        W   = torch.zeros(C, C, k)

        mode = self.cfg.filter_init

        if mode == "temporal_bank_full":
            # keyboard: thirds
            # 0 .. C//3-1          → EMA  [0.08, 0.28, 0.64]
            # C//3 .. 2*C//3-1     → difference [-1, 1, 0]  (t-1 minus t-2, causal)
            # 2*C//3 .. C-1        → identity [0, 0, 1]
            t1 = C // 3        # 21 for C=64
            t2 = 2 * C // 3    # 42 for C=64
            ema_w  = torch.tensor([0.08, 0.28, 0.64])
            diff_w = torch.tensor([-1.0,  1.0,  0.0])
            id_w   = torch.tensor([ 0.0,  0.0,  1.0])
            for ch in range(t1):
                W[ch, ch, :] = ema_w
            for ch in range(t1, t2):
                W[ch, ch, :] = diff_w
            for ch in range(t2, C):
                W[ch, ch, :] = id_w

        elif mode == "temporal_bank_half":
            # notif: halves — no difference (fires on zero-pad boundary)
            # 0 .. C//2-1     → EMA  [0.08, 0.28, 0.64]
            # C//2 .. C-1     → identity [0, 0, 1]
            half  = C // 2     # 16 for C=32
            ema_w = torch.tensor([0.08, 0.28, 0.64])
            id_w  = torch.tensor([ 0.0,  0.0,  1.0])
            for ch in range(half):
                W[ch, ch, :] = ema_w
            for ch in range(half, C):
                W[ch, ch, :] = id_w

        elif mode == "identity":
            # switching: all channels pure passthrough
            # buffer is sparse (zeros between 120s updates)
            # staleness/freshness handles temporal weighting externally
            id_w = torch.tensor([0.0, 0.0, 1.0])
            for ch in range(C):
                W[ch, ch, :] = id_w

        with torch.no_grad():
            self.blocks[0].conv1.conv.weight.copy_(W)

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