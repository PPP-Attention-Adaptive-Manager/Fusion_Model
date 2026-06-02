# predictive_models/keyboard/dummy.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty


class DummyKeyboardModel(BaseModalityModel):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.backbone = nn.Sequential(
            nn.Linear(d_proj, 128),
            nn.ReLU(),
        )

        self.factor_head = nn.Linear(128, 5)
        self.state_head = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor):

        # mandatory projector
        x = self.projector(x)

        feat = self.backbone(x)

        factors = self.factor_head(feat)

        # RAW logits
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        pass