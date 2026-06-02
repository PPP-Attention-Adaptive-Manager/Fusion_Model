"""Baseline predictive models for smoke testing fusion wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from ema.ema import compute_uncertainty
from predictive_models.base import BaseModalityModel


class BaselineMLP(BaseModalityModel):
    """Small contract-compliant MLP for untrained modality slots.

    This is a wiring baseline, not a trained cognitive-state model.
    Replace per-modality `ActiveModel` imports with trained models when ready.
    """

    def __init__(self, input_flat_dim: int, d_proj: int = 256, hidden_dim: int = 128):
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj)
        self.net = nn.Sequential(
            nn.LayerNorm(d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, hidden_dim),
            nn.ReLU(),
        )
        self.factor_head = nn.Linear(hidden_dim, self.N_FACTORS)
        self.state_head = nn.Linear(hidden_dim, self.N_STATES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.projector(x)
        hidden = self.net(feat)

        factors = self.factor_head(hidden)
        logits = self.state_head(hidden)
        H_norm, M = compute_uncertainty(logits)

        return torch.cat(
            [
                factors,
                logits,
                H_norm.unsqueeze(-1),
                M.unsqueeze(-1),
            ],
            dim=-1,
        )

