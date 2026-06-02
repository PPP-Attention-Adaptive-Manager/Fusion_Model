# predictive_models/keyboard/v3_transformer.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin
from .common import AttentionPooling


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class KeyboardTransformer(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 24,
        nhead: int = 8,
        num_layers: int = 3,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.positional = PositionalEncoding(d_proj)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_proj,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
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

        seq = self.positional(seq)

        feat = self.transformer(seq)
        feat = self.pool(feat)
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