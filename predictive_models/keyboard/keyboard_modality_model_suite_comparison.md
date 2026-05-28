# Keyboard Modality — Integratable Model Suite

This document provides 4 fully project-compatible keyboard predictive model implementations for the AAM Inférer fusion architecture.

Included:

1. GRU baseline
2. TCN model
3. Transformer model
4. Hybrid TCN + Transformer model
5. Unified evaluation/training scaffold
6. Model comparison table
7. Integration instructions
8. Recommended experiment order

All implementations:

- subclass `BaseModalityModel`
- accept `(B, input_flat_dim)`
- return `(B, 12)`
- use `compute_uncertainty()`
- support LOSO reset via `reset_microstate()`
- preserve raw logits contract
- are hot-swappable through `__init__.py`

---

# 1. Folder Layout

```text
predictive_models/
└── keyboard/
    ├── __init__.py
    ├── dummy.py
    ├── v1_gru.py
    ├── v2_tcn.py
    ├── v3_transformer.py
    ├── v4_hybrid.py
    ├── common.py
    └── train_compare.py
```

---

# 2. Shared Utilities — common.py

```python
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
```

---

# 3. GRU Baseline — v1_gru.py

```python
# predictive_models/keyboard/v1_gru.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin


class KeyboardGRU(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        hidden_dim: int = 128,
        seq_len: int = 16,
        num_layers: int = 2,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(hidden_dim, 5)
        self.state_head = nn.Linear(hidden_dim, 5)

        self.microstate = {}

    def forward(self, x: torch.Tensor):
        """
        x: (B, input_flat_dim)
        """

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        h = self.microstate.get("h", None)

        out, h_new = self.gru(seq, h)

        self.microstate["h"] = h_new.detach()

        feat = out[:, -1]
        feat = self.norm(feat)
        feat = feat + self.mlp(feat)

        factors = self.factor_head(feat)
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        self.microstate = {}
        self.clear_history()
```

---

# 4. TCN Model — v2_tcn.py

```python
# predictive_models/keyboard/v2_tcn.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin


class TemporalBlock(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.net(x)


class KeyboardTCN(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 16,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.tcn = nn.Sequential(
            TemporalBlock(d_proj, dilation=1),
            TemporalBlock(d_proj, dilation=2),
            TemporalBlock(d_proj, dilation=4),
            TemporalBlock(d_proj, dilation=8),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(d_proj, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(128, 5)
        self.state_head = nn.Linear(128, 5)

    def forward(self, x):

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        seq = seq.transpose(1, 2)

        feat = self.tcn(seq)
        feat = self.pool(feat).squeeze(-1)
        feat = self.head(feat)

        factors = self.factor_head(feat)
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        self.clear_history()
```

---

# 5. Transformer Model — v3_transformer.py

```python
# predictive_models/keyboard/v3_transformer.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin
from .common import AttentionPooling


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class KeyboardTransformer(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 24,
        nhead: int = 8,
        num_layers: int = 3,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.positional = PositionalEncoding(d_proj)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_proj,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.pool = AttentionPooling(d_proj)

        self.shared = nn.Sequential(
            nn.Linear(d_proj, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(128, 5)
        self.state_head = nn.Linear(128, 5)

    def forward(self, x):

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        seq = self.positional(seq)

        feat = self.transformer(seq)
        feat = self.pool(feat)
        feat = self.shared(feat)

        factors = self.factor_head(feat)
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        self.clear_history()
```

---

# 6. Hybrid TCN + Transformer — v4_hybrid.py

```python
# predictive_models/keyboard/v4_hybrid.py

import torch
import torch.nn as nn

from ..base import BaseModalityModel
from ema.ema import compute_uncertainty

from .common import RollingSequenceMixin
from .common import AttentionPooling


class HybridTCNBlock(nn.Module):

    def __init__(self, dim, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.block(x)


class KeyboardHybrid(BaseModalityModel, RollingSequenceMixin):

    def __init__(
        self,
        input_flat_dim: int,
        d_proj: int = 256,
        seq_len: int = 24,
    ):
        super().__init__(input_flat_dim, d_proj)

        self.init_sequence_buffer(seq_len)

        self.tcn = nn.Sequential(
            HybridTCNBlock(d_proj, 1),
            HybridTCNBlock(d_proj, 2),
            HybridTCNBlock(d_proj, 4),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_proj,
            nhead=8,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.pool = AttentionPooling(d_proj)

        self.shared = nn.Sequential(
            nn.Linear(d_proj, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.factor_head = nn.Linear(128, 5)
        self.state_head = nn.Linear(128, 5)

    def forward(self, x):

        x = self.projector(x)

        self.append_step(x)
        seq = self.get_sequence(x)

        tcn_feat = self.tcn(seq.transpose(1, 2))
        tcn_feat = tcn_feat.transpose(1, 2)

        tr_feat = self.transformer(tcn_feat)

        feat = self.pool(tr_feat)
        feat = self.shared(feat)

        factors = self.factor_head(feat)
        logits = self.state_head(feat)

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)

    def reset_microstate(self):
        self.clear_history()
```

---

# 7. Integration Switch — __init__.py

```python
# predictive_models/keyboard/__init__.py

# GRU baseline
# from .v1_gru import KeyboardGRU as ActiveModel

# TCN
# from .v2_tcn import KeyboardTCN as ActiveModel

# Transformer
# from .v3_transformer import KeyboardTransformer as ActiveModel

# Hybrid (recommended final model)
from .v4_hybrid import KeyboardHybrid as ActiveModel
```

---

# 8. Unified Training + Comparison Script

```python
# predictive_models/keyboard/train_compare.py

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

import torch
import torch.nn.functional as F


MODELS = {
    "gru": "v1_gru.KeyboardGRU",
    "tcn": "v2_tcn.KeyboardTCN",
    "transformer": "v3_transformer.KeyboardTransformer",
    "hybrid": "v4_hybrid.KeyboardHybrid",
}


class JointLoss(torch.nn.Module):

    def forward(self, output, tlx_targets, state_targets):

        factor_loss = F.huber_loss(
            output[:, :5],
            tlx_targets,
        )

        state_loss = F.cross_entropy(
            output[:, 5:10],
            state_targets,
        )

        total = 0.4 * factor_loss + 0.6 * state_loss

        return total


def evaluate(y_true, y_pred):

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
```

---

# 9. Recommended Experiment Order

## Phase 1 — Sanity Baselines

Run:

1. GRU
2. TCN

Purpose:
- validate labels
- validate temporal dynamics
- validate LOSO
- detect leakage

---

## Phase 2 — Main Research Models

Run:

1. Transformer
2. Hybrid

Purpose:
- maximize state separation
- improve overload detection
- improve distraction transitions

---

# 10. Expected Performance Ranking

| Model | Expected F1 | Stability | Compute | Research Value |
|---|---|---|---|---|
| GRU | High | Very High | Low | Strong baseline |
| TCN | High | High | Very Low | Efficient realtime |
| Transformer | Very High | Medium | Medium-High | Strong paper value |
| Hybrid | Highest | Medium | High | Best final architecture |

---

# 11. Recommended Hyperparameters

| Parameter | Recommended |
|---|---|
| seq_len | 16–24 |
| d_proj | 256 |
| dropout | 0.1 |
| optimizer | AdamW |
| lr | 1e-4 |
| scheduler | cosine decay |
| batch size | 32 |
| gradient clipping | 1.0 |

---

# 12. Most Important Keyboard Signals

The models are expected to learn strongest predictive power from:

- typing burstiness
- pause distribution
- correction density
- typing speed volatility
- rhythm stability
- interruption recovery
- hesitation patterns
- cadence entropy
- sustained activity coherence
- rapid context-switch signatures

These strongly correlate with:

- overload
- distraction
- frustration
- flow stability

---

# 13. Final Recommendation

## Best engineering baseline

Use:

```python
KeyboardGRU
```

because it is:
- stable
- easy to debug
- strong under LOSO
- lightweight

---

## Best final paper model

Use:

```python
KeyboardHybrid
```

because it combines:

- local rhythm extraction (TCN)
- long-range cognitive transitions (Transformer)
- realtime compatibility
- strong multimodal behavioral modeling

This is likely the highest-ceiling architecture for the keyboard modality in the current fusion framework.

