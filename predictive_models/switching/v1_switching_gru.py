"""Temporal switching predictive model.

The switching slot receives its modality-specific Tucker slice, not the raw
GraphSAGE embedding. With the default fusion rank this input is (B, 512).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ema.ema import compute_uncertainty
from predictive_models.base import BaseModalityModel


class SwitchingGRU(BaseModalityModel):
    """GRU predictor for the switching/behavior graph modality.

    Input contract:
        x: (B, input_flat_dim), where input_flat_dim is TFN.flat_size(3).

    Output contract:
        (B, 12): 5 factor scores, 5 raw state logits, H_norm, M.
    """

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=d_proj,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.factor_head = nn.Linear(d_proj, self.N_FACTORS)
        self.state_head = nn.Linear(d_proj, self.N_STATES)

    def _get_hidden(self, batch_size: int):
        h = self.microstate.get("h")
        if h is None or h.shape[1] != batch_size:
            return None
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.projector(x)

        h = self._get_hidden(batch_size=feat.shape[0])
        out, h_new = self.gru(feat.unsqueeze(1), h)
        self.microstate["h"] = h_new.detach()

        out = out.squeeze(1)
        factors = self.factor_head(out)
        logits = self.state_head(out)
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

    def reset_microstate(self):
        self.microstate = {}
