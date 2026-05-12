"""
BaseModalityModel — interface contract every modality model must satisfy.

Teammates subclass this. The fusion model calls:
    - forward(x)          at every window
    - reset_microstate()  at every LOSO subject boundary

Output contract (B, 12):
    dims  0–4  : [mental_demand, temporal_demand, effort, frustration, arousal]
    dims  5–9  : [logit_Flow, logit_Neutral, logit_Bored, logit_Distracted, logit_Overloaded]
    dim  10    : H_norm   (from compute_uncertainty)
    dim  11    : M        (from compute_uncertainty)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModalityModel(nn.Module, ABC):
    OUTPUT_DIM = 12
    N_STATES   = 5
    N_FACTORS  = 5

    def __init__(self, input_flat_dim: int, d_proj: int = 256):
        """
        Args:
            input_flat_dim : flat size of the TFN slice for this modality
            d_proj         : size of the compressed projection (tune per model)
        """
        super().__init__()
        self.input_flat_dim = input_flat_dim
        self.d_proj         = d_proj

        # Mandatory first compression step — always call this in forward()
        # before any model-specific logic.
        self.projector = nn.Linear(input_flat_dim, d_proj)

        # Microstate dict — populate for sequential models (GRU hidden state etc.)
        self.microstate: dict = {}

    def reset_microstate(self):
        """Override if your model carries hidden state across windows."""
        self.microstate = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_flat_dim)
        Returns:
            (B, 12) — must satisfy output contract above
        """
        ...