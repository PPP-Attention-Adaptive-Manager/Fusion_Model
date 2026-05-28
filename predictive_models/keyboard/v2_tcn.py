# predictive_models/keyboard/v2_tcn.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin


class TemporalBlock(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.net(x)


class KeyboardTCN(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 16,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.tcn = nn.Sequential(
            TemporalBlock(d_proj, dilation=1),
            TemporalBlock(d_proj, dilation=2),
            TemporalBlock(d_proj, dilation=4),
            TemporalBlock(d_proj, dilation=8),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(d_proj, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(128, 5)
        self.state_head = nn.Linear(128, 5)

    def forward(self, x):

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        seq = seq.transpose(1, 2)

        feat = self.tcn(seq)
        feat = self.pool(feat).squeeze(-1)
        feat = self.head(feat)

        factors = self.factor_head(feat)
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        self.clear_history()