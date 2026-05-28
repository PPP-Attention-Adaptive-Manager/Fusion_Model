# predictive_models/keyboard/v1_gru.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin


class KeyboardGRU(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        hidden_dim: int = 128,
        seq_len: int = 16,
        num_layers: int = 2,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(hidden_dim, 5)
        self.state_head = nn.Linear(hidden_dim, 5)

        self.microstate = {}

    def forward(self, x: torch.Tensor):
        """
        x: (B, input_flat_dim)
        """

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        h = self.microstate.get("h", None)

        out, h_new = self.gru(seq, h)

        self.microstate["h"] = h_new.detach()

        feat = out[:, -1]
        feat = self.norm(feat)
        feat = feat + self.mlp(feat)

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
        self.microstate = {}
        self.clear_history()