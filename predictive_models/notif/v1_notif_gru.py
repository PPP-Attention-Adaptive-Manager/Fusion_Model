import torch
import torch.nn as nn
from ema.ema import compute_uncertainty
from predictive_models.base import BaseModalityModel


class NotifGRU(BaseModalityModel):

    def __init__(self, input_flat_dim: int, d_proj: int = 256, **kwargs):
        super().__init__(input_flat_dim=input_flat_dim, d_proj=d_proj, **kwargs)
        # self.projector is already registered by BaseModalityModel
        # it maps (B, input_flat_dim) -> (B, d_proj) where d_proj=256

        self.gru = nn.GRU(
            input_size  = d_proj,
            hidden_size = d_proj,
            num_layers  = 1,
            batch_first = True,
        )
        self.factor_head = nn.Linear(d_proj, 5)
        self.state_head  = nn.Linear(d_proj, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_flat_dim)

        # Mandatory first line -- always call projector first
        feat = self.projector(x)              # (B, d_proj)

        # Retrieve hidden state if it exists
        h = self.microstate.get("h", None)

        # GRU expects (B, seq_len, input_size)
        out, h_new = self.gru(feat.unsqueeze(1), h)   # out: (B,1,d_proj)
        self.microstate["h"] = h_new.detach()

        out = out.squeeze(1)                  # (B, d_proj)

        factors = self.factor_head(out)       # (B, 5)
        logits  = self.state_head(out)        # (B, 5) RAW -- no softmax

        H_norm, M = compute_uncertainty(logits)   # both (B,)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)                            # (B, 12)

    def reset_microstate(self):
        self.microstate = {}
