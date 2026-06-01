"""Simple MLP diagnostic baseline for the switching Tucker slice."""

from __future__ import annotations

import torch
import torch.nn as nn

from predictive_models.base import BaseModalityModel
from predictive_models.switching.v1_gru_switching import compute_state_uncertainty


class SwitchingMLP(BaseModalityModel):
    """MLP baseline used only for diagnostics and baseline comparison."""

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        hidden_dim: int = 256,
        num_states: int = 5,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.net = nn.Sequential(
            nn.LayerNorm(d_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, hidden_dim),
            nn.ReLU(),
        )
        self.factor_head = nn.Linear(hidden_dim, self.N_FACTORS)
        self.state_head = nn.Linear(hidden_dim, num_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.projector(x)
        hidden = self.net(feat)
        factors = self.factor_head(hidden)
        logits = self.state_head(hidden)
        H_norm, M = compute_state_uncertainty(logits)
        return torch.cat(
            [
                factors,
                logits,
                H_norm.unsqueeze(-1),
                M.unsqueeze(-1),
            ],
            dim=-1,
        )
