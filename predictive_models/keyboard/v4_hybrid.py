# predictive_models/keyboard/v4_hybrid.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin
from .common import AttentionPooling


class HybridTCNBlock(nn.Module):

    def __init__(self, dim, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.block(x)


class KeyboardHybrid(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 24,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.tcn = nn.Sequential(
            HybridTCNBlock(d_proj, 1),
            HybridTCNBlock(d_proj, 2),
            HybridTCNBlock(d_proj, 4),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_proj,
            nhead=8,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.pool = AttentionPooling(d_proj)

        self.shared = nn.Sequential(
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

        tcn_feat = self.tcn(seq.transpose(1, 2))
        tcn_feat = tcn_feat.transpose(1, 2)

        tr_feat = self.transformer(tcn_feat)

        feat = self.pool(tr_feat)
        feat = self.shared(feat)

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