"""GRU predictive model for the switching Tucker slice.

Input:
    x: Tensor shape (B, 512)

Output:
    Tensor shape (B, 12)
    dims 0-4: factor scores
    dims 5-9: raw state logits
    dim 10: H_norm
    dim 11: margin M
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ema.ema import compute_uncertainty
from predictive_models.base import BaseModalityModel


class SwitchingGRU(BaseModalityModel):
    """GRU-based predictive model for switching/behavior graph interactions."""

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.factor_head = nn.Linear(hidden_dim, self.N_FACTORS)
        self.state_head = nn.Linear(hidden_dim, self.N_STATES)

    def _hidden_for_batch(self, batch_size: int):
        h = self.microstate.get("h")
        if h is None or h.shape[1] != batch_size:
            return None
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.projector(x)
        h = self._hidden_for_batch(feat.shape[0])
        out, h_new = self.gru(feat.unsqueeze(1), h)
        self.microstate["h"] = h_new.detach()

        hidden = out.squeeze(1)
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

    def reset_microstate(self) -> None:
        self.microstate = {}
