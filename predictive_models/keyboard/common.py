# predictive_models/keyboard/common.py

from collections import deque

import torch
import torch.nn as nn


class RollingSequenceMixin:
    """
    Maintains rolling temporal context across 1Hz ticks.
    """

    def init_sequence_buffer(self, seq_len: int):
        self.seq_len = seq_len
        self.history = deque(maxlen=seq_len)

    def append_step(self, x: torch.Tensor):
        """
        x: (B, D)
        """
        self.history.append(x.detach())

    def get_sequence(self, current_x: torch.Tensor):
        """
        Returns:
            (B, T, D)
        """

        if len(self.history) == 0:
            for _ in range(self.seq_len):
                self.history.append(current_x.detach())

        while len(self.history) < self.seq_len:
            self.history.append(self.history[-1])

        seq = torch.stack(list(self.history), dim=1)
        return seq

    def clear_history(self):
        self.history.clear()


class AttentionPooling(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: (B, T, D)
        """

        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled