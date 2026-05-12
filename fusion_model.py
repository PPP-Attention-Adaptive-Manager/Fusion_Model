"""
AAM Inférer — Fusion Model (orchestrator)
==========================================
Wires together: TFN → 4 model slots → PoE → EMA → final output.

Active model per modality is controlled by its __init__.py:
    fusion/models/mouse/__init__.py      ← change import here to swap mouse model
    fusion/models/keyboard/__init__.py
    fusion/models/notif/__init__.py
    fusion/models/gnn/__init__.py

Output contract
---------------
{
  "global"   : (B, 11)
      [mental_demand, temporal_demand, effort, frustration, arousal,
       p_Flow, p_Neutral, p_Bored, p_Distracted, p_Overloaded,
       H_norm_ensemble]

  "per_model": [(B, 12)] × 4
      dims 0-4  : factor scores
      dims 5-9  : state logits
      dim  10   : H_norm
      dim  11   : M
}
"""

from typing import Dict, List
import torch
import torch.nn as nn

from TFN.tfn import TFNLayer
from poe.poe import PoEFusion
from ema.ema import EMAsmoother, compute_uncertainty
from predictive_models import MODALITY_MODELS


class InferrerFusion(nn.Module):

    MODALITY_NAMES = ["mouse", "keyboard", "notif", "gnn"]

    def __init__(
        self,
        d_dims    : List[int],
        d_proj    : int   = 256,
        poe_mode  : str   = "vanilla",
        ema_alpha : float = 0.7,
    ):
        super().__init__()
        assert len(d_dims) == 4

        self.d_dims = d_dims
        self.tfn    = TFNLayer(d_dims)

        self.models = nn.ModuleList([
            MODALITY_MODELS[i](
                input_flat_dim = self.tfn.flat_size(i),
                d_proj         = d_proj,
            )
            for i in range(4)
        ])

        self.poe = PoEFusion(mode=poe_mode)
        self.ema = EMAsmoother(alpha=ema_alpha)

    def reset_subject(self):
        """Call at every LOSO subject boundary."""
        self.ema.reset()
        for model in self.models:
            model.reset_microstate()

    def set_poe_mode(self, mode: str):
        """Hot-swap PoE mode for ablation runs."""
        self.poe.mode = mode

    def replace_model(self, modality_idx: int, new_model: nn.Module):
        """Plug in a trained model for one modality slot at runtime."""
        assert hasattr(new_model, "reset_microstate"), \
            "Model must implement reset_microstate()"
        self.models[modality_idx] = new_model
        print(f"[InferrerFusion] '{self.MODALITY_NAMES[modality_idx]}' → {type(new_model).__name__}")

    def forward(self, embeddings: List[torch.Tensor]) -> Dict[str, object]:
        """
        Args:
            embeddings: [h_mouse, h_kb, h_notif, h_gnn], each (B, d_m)
        Returns:
            {"global": (B,11), "per_model": [(B,12)] x 4}
        """
        tensor = self.tfn(embeddings)

        per_model_outputs = [
            self.models[i](self.tfn.get_slice(tensor, i))
            for i in range(4)
        ]

        p_poe   = self.poe(per_model_outputs)
        p_final = self.ema.step(p_poe)

        factors = torch.stack(
            [out[:, :5] for out in per_model_outputs], dim=0
        ).mean(dim=0)

        H_ens, _ = compute_uncertainty(torch.log(p_final + 1e-8))

        global_out = torch.cat([
            factors,
            p_final,
            H_ens.unsqueeze(-1),
        ], dim=-1)

        return {"global": global_out, "per_model": per_model_outputs}