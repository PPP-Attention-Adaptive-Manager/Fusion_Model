"""
Mouse GRU predictive model.
Subclasses BaseModalityModel — plugs into fusion model.
Input:  (B, 512) Tucker slice
Output: (B, 12)  — see BaseModalityModel output contract
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ema.ema import compute_uncertainty
from predictive_models.base import BaseModalityModel


class MouseGRU(BaseModalityModel):
    """
    GRU-based mouse cognitive load predictor.

    Architecture:
        projector  : Linear(512 → d_proj)     [from base]
        layernorm  : LayerNorm(d_proj)
        gru        : GRU(d_proj → d_proj)
        factor_head: Linear(d_proj → 5)       [MD, TD, EF, FR, AR]
        state_head : Linear(d_proj → 5)       [raw logits, no softmax]
    """

    def __init__(
        self,
        input_flat_dim: int = 512,
        d_proj:         int = 256,
        num_layers:     int = 1,
        dropout:        float = 0.1,
        **kwargs,
    ):
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        self.norm = nn.LayerNorm(d_proj)
        self.gru  = nn.GRU(
            input_size  = d_proj,
            hidden_size = d_proj,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.factor_head = nn.Linear(d_proj, self.N_FACTORS)
        self.state_head  = nn.Linear(d_proj, self.N_STATES)

    def _get_hidden(self, batch_size: int):
        h = self.microstate.get("h")
        if h is None or h.shape[1] != batch_size:
            return None
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, 512)
        Returns: (B, 12)
        """
        feat = self.projector(x)          # (B, d_proj) — mandatory first line
        feat = self.norm(feat)

        h         = self._get_hidden(feat.shape[0])
        out, h_new = self.gru(feat.unsqueeze(1), h)   # (B, 1, d_proj)
        self.microstate["h"] = h_new.detach()

        out     = out.squeeze(1)                       # (B, d_proj)
        factors = self.factor_head(out)                # (B, 5)
        logits  = self.state_head(out)                 # (B, 5) — raw, no softmax

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)                                     # (B, 12)

    def reset_microstate(self):
        self.microstate = {}
