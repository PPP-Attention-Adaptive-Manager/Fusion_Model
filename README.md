# AAM Inférer — Fusion Model: Teammate Integration Guide

> **Who this is for:** every teammate working on a modality embedder or predictive model  
> **What it covers:** how the fusion model works, how to plug your work in, how to experiment safely  
> **Rule #1:** never modify `fusion_model.py`, `TFN/`, `poe/`, or `ema/` — these are locked infrastructure

---

## 1. Pipeline Overview

```
Your embedder output
      ↓
TCN_encoders/{modality}/encoder.py   ← BufferedEncoder (handles timing + buffering)
      ↓
TFN/low_rank_tucker.py               ← cross-modal interaction (Tucker R=8)
      ↓
predictive_models/{modality}/        ← YOUR trained model lives here
      ↓
poe/poe.py                           ← combines 4 model predictions
      ↓
ema/ema.py                           ← temporal smoothing
      ↓
output {"global": (B,11), "per_model": [(B,12)] × 4}
```

The fusion model runs at **1Hz** — one forward pass per second. Every component must produce its output within that budget.

---

## 2. Locked Dimensions — Do Not Change These

```python
# Embedding dimensions entering TFN (output of each BufferedEncoder)
d_dims = [64, 64, 32, 32]
#        mouse  kb  notif  switching

# Tucker rank
R = 8

# Flat input to every predictive model slot
input_flat_dim = R ** 3   # = 512 for all 4 modalities
```

These numbers are set in `fusion_model.py`. If your embedder outputs a different dimension, the fix goes in **your `BufferedEncoder`** (add a projection layer there), not in the fusion model.

---

## 3. Full Folder Structure

```
fusion_model/
├── fusion_model.py              ← LOCKED — orchestrator, do not edit
│
├── TFN/
│   ├── __init__.py              ← LOCKED
│   ├── tfn.py                   ← LOCKED (full outer product, non-default)
│   └── low_rank_tucker.py       ← LOCKED (active default, Tucker R=8)
│
├── poe/
│   ├── __init__.py              ← LOCKED
│   └── poe.py                   ← LOCKED
│
├── ema/
│   ├── __init__.py              ← LOCKED
│   └── ema.py                   ← LOCKED
│
├── TCN_encoders/
│   ├── buffered_encoder.py      ← LOCKED (base class)
│   ├── fixed_tcn.py             ← LOCKED
│   ├── tcn_block.py             ← LOCKED
│   ├── configs.py               ← LOCKED
│   ├── mouse/
│   │   ├── __init__.py          ← LOCKED
│   │   └── encoder.py           ← LOCKED (passthrough)
│   ├── keyboard/
│   │   ├── __init__.py          ← LOCKED
│   │   └── encoder.py           ← LOCKED
│   ├── notif/
│   │   ├── __init__.py          ← LOCKED
│   │   └── encoder.py           ← LOCKED
│   └── switching/
│       ├── __init__.py          ← LOCKED
│       └── encoder.py           ← LOCKED
│
├── pre_embedders/               ← YOUR WORK GOES HERE (one folder per modality)
│   ├── mouse/
│   ├── keyboard/
│   ├── notif/
│   └── switching/
│
└── predictive_models/
    ├── base.py                  ← LOCKED (read carefully before subclassing)
    ├── __init__.py              ← LOCKED (registry)
    ├── mouse/
    │   ├── __init__.py          ← SWAP POINT — change one line to plug your model
    │   ├── dummy.py             ← LOCKED (structural placeholder)
    │   └── v1_your_model.py     ← YOUR MODEL GOES HERE
    ├── keyboard/
    ├── notif/
    └── switching/
```

---

## 4. How Your Embedder Connects to the Fusion Model

### 4.1 What the fusion model expects at each tick

At every 1Hz tick the orchestrator calls:

```python
embeddings = [
    mouse_enc.step(m_t),             # m_t: torch.Tensor (1, 64)
    keyboard_enc.step(kb_or_none),   # kb:  np.ndarray  (64,)  or None
    notif_enc.step(notif_or_none),   # dict or None
    switching_enc.step(sw_or_none),  # dict or None
]
output = inferrer_fusion([e for e, _ in embeddings])
```

Each `encoder.step()` call accepts your raw output and returns `(embedding, freshness)`.

### 4.2 Mouse embedder integration

**Your output:** `m_t` — `torch.Tensor` shape `(1, 64)` from `Phase2Pipeline`  
**How to push:** direct pass, no parsing needed

```python
# in your observer loop, every 1 second:
window_events = observer.get_window(window_sec=5.0)
m_t = pipeline.process_session(window_events, user_id=session.user_id)
if m_t:
    fusion_input.mouse_queue.put(m_t[-1])   # (1, 64) tensor
```

The `MouseBufferedEncoder` is a pure passthrough — it returns your tensor unchanged with freshness=1.0.

### 4.3 Keyboard embedder integration

**Your output:** embedding — `np.ndarray` shape `(64,)` from `KeystrokeEncoder`  
**Emission cadence:** one vector per S=10 completed keystrokes (~2–3s)  
**How to push:** call `step()` with the embedding when a window completes, `None` otherwise

```python
# every time KeystrokeEncoder completes a window:
embedding = keystroke_encoder.forward(window)   # (1, 64) or (64,)
fusion_input.keyboard_queue.put(embedding)

# in the 1Hz fusion loop:
kb_emb = fusion_input.keyboard_queue.get_nowait() if not queue.empty() else None
emb, fresh = keyboard_enc.step(kb_emb)
```

The `KeyboardBufferedEncoder` accepts both numpy `(64,)` and torch `(1,64)` — handles both.

### 4.4 Notif embedder integration

**Your output:** dict from `get_output(session)` every 5 seconds  
**Required fields (others are ignored):**

```python
{
    "embedding":        np.array shape (16,) dtype float32,
    "npi":              float,
    "burstiness":       float,
    "disruption_score": float,
    # "state" is IGNORED — do not pass to fusion
    # "metadata" is IGNORED
}
```

**How to push:**

```python
# every 5 seconds when notif module fires:
result = notif_module.get_output(session)
fusion_input.notif_queue.put(result)

# in the 1Hz fusion loop:
notif_dict = fusion_input.notif_queue.get_nowait() if not queue.empty() else None
emb, fresh = notif_enc.step(notif_dict)
```

The encoder parses the dict, extracts `[embedding(16), npi, burstiness, disruption_score]`, buffers the 19-dim vector, runs TCN. You do nothing extra.

### 4.5 Switching/GNN embedder integration

**Your output:** dict from behavioral graph module every 120 seconds  
**Required fields:**

```python
{
    "embedding":     np.array shape (64,) dtype float32,
    "graph_metrics": {
        "num_nodes":       int,
        "num_edges":       int,
        "density":         float,
        "switch_rate":     float,
        "fragmentation":   float,
        "focus_ratio":     float,
        "multitask_score": float,
    }
    # "state" is IGNORED
    # "metadata" is IGNORED
}
```

**How to push:** same pattern as notif, but fires every 120s. The encoder holds the last known embedding for up to 120s — staleness decays the contribution via `exp(-staleness/60)`.

---

## 5. The Output Contract Every Predictive Model Must Satisfy

This is the most important section if you are writing a predictive model.

Every model in `predictive_models/{modality}/` must:

1. Subclass `BaseModalityModel` from `predictive_models/base.py`
2. Accept input `(B, 512)` — the flat TFN slice (`input_flat_dim = R^3 = 512`)
3. Return output `(B, 12)` with this exact layout:

```
dims 0–4  : continuous factor scores
              [mental_demand, temporal_demand, effort, frustration, arousal]
dims 5–9  : state logits (RAW — do NOT apply softmax here)
              [logit_Flow, logit_Neutral, logit_Bored, logit_Distracted, logit_Overloaded]
dim  10   : H_norm — normalized predictive entropy ∈ [0,1]
dim  11   : M      — margin confidence (p_top1 − p_top2)
```

**H_norm and M must be computed using `compute_uncertainty` from `ema/ema.py`:**

```python
from ema.ema import compute_uncertainty

H_norm, M = compute_uncertainty(logits)   # logits: (B, 5)
```

### Why raw logits (no softmax before returning)

The PoE fusion layer sums logits from all 4 models before applying a single softmax. If you apply softmax inside your model first, you destroy confidence information — all distributions become equally normalized and the logit magnitudes (which carry certainty) are lost. **This is a hard rule. Do not apply softmax to dims 5–9 before returning.**

---

## 6. Training Loss

Use this joint loss for every modality model:

```python
loss = 0.4 * factor_loss + 0.6 * state_loss

# factor_loss: MSE or Huber on dims 0–4 vs NASA-TLX ground truth
factor_loss = F.huber_loss(output[:, :5], nasa_tlx_labels)

# state_loss: cross-entropy on dims 5–9 vs state labels
state_loss = F.cross_entropy(output[:, 5:10], state_labels)
```

State labels come from `derive_state_label()` using NASA-TLX subscore thresholds. Frustration is the primary discriminator between Flow and Overloaded.

**5 cognitive states — fixed, do not change:**

| Index | State | Description |
| --- | --- | --- |
| 0   | Flow | High performance, low frustration |
| 1   | Neutral | Baseline engaged |
| 2   | Bored | Low engagement |
| 3   | Distracted | Fragmented attention |
| 4   | Overloaded | High demand + high frustration |

Eureka and Surprise were intentionally excluded — they are observationally indistinguishable from mouse/keyboard signals. This is an architectural decision, not a gap.

---

## 7. How to Write and Plug In a Predictive Model

### Step 1 — Create your model file

```python
# predictive_models/mouse/v1_gru.py

import torch
import torch.nn as nn
from ..base import BaseModalityModel
from ema.ema import compute_uncertainty


class MouseGRU(BaseModalityModel):

    def __init__(self, input_flat_dim: int, d_proj: int = 256):
        super().__init__(input_flat_dim, d_proj)
        # self.projector = nn.Linear(512, 256) — already created by base class
        # build your architecture below, starting from d_proj
        self.gru    = nn.GRU(d_proj, 128, batch_first=True)
        self.factor_head = nn.Linear(128, 5)
        self.state_head  = nn.Linear(128, 5)
        self.microstate  = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 512)
        B = x.shape[0]

        # MANDATORY: projector is always called first
        x = self.projector(x)              # (B, 256)

        # retrieve GRU hidden state if exists
        h = self.microstate.get('h', None)

        out, h_new = self.gru(x.unsqueeze(1), h)   # (B, 1, 128)
        self.microstate['h'] = h_new.detach()

        feat = out.squeeze(1)              # (B, 128)

        factors = self.factor_head(feat)   # (B, 5)
        logits  = self.state_head(feat)    # (B, 5) — RAW logits, no softmax

        H_norm, M = compute_uncertainty(logits)

        return torch.cat([
            factors,
            logits,
            H_norm.unsqueeze(-1),
            M.unsqueeze(-1),
        ], dim=-1)                         # (B, 12)

    def reset_microstate(self):
        # called at every LOSO subject boundary
        self.microstate = {}
```

### Step 2 — Point the swap file at your model

```python
# predictive_models/mouse/__init__.py
# Change this one line only:

from .v1_gru import MouseGRU as ActiveModel   # ← your model
# from .dummy import DummyMouseModel as ActiveModel   # ← old placeholder
```

**That is the entire integration.** Nothing else changes anywhere in the codebase.

### Step 3 — Verify with the smoke test

```bash
cd fusion_model/
python test_fusion.py
```

All 6 tests must pass. If they do, your model is correctly integrated.

---

## 8. How to Experiment — Safely

### Trying a new architecture

Create `v2_transformer.py` in your modality folder. Change the `__init__.py` import to point at it. The previous version remains on disk — reverting is one line.

```
predictive_models/mouse/
├── dummy.py           ← always keep, never delete
├── v1_gru.py          ← previous version
├── v2_transformer.py  ← new experiment
└── __init__.py        ← one line: from .v2_transformer import ... as ActiveModel
```

### Trying a different TCN filter config for your modality

```python
# TCN_encoders/keyboard/__init__.py
# change this line only:
from ..configs import multiscale as ActiveConfig   # was narrow, trying multiscale
```

### Trying PoE weighted vs vanilla mode

```python
# in your training/eval script:
model.set_poe_mode("weighted")   # or "vanilla"
# hot-swappable, no reinitialization needed
```

### Trying different Tucker rank

```python
# fusion_model.py construction:
inferrer = InferrerFusion(d_dims=[64, 64, 32, 32], rank=4)   # was 8, try 4
```

Note: changing rank changes `flat_size` from 512 to 64. Your model's `input_flat_dim` updates automatically — the base class projector rebuilds itself at construction time.

---

## 9. Ablation Study — What to Report

These are the required ablation conditions for the technical report:

| Condition | What changes | Purpose |
| --- | --- | --- |
| Tucker R=4 | `rank=4` in fusion | lower bound |
| Tucker R=8 | default | primary result |
| Tucker R=16 | `rank=16` | upper bound |
| PoE vanilla | `set_poe_mode("vanilla")` | equal weight |
| PoE weighted | `set_poe_mode("weighted")` | uncertainty-weighted |
| Modality dropout | set one modality to zeros | contribution per signal |
| Concat baseline | replace TFN with concatenation MLP | justify cross-modal interactions |

Report per-state F1 + MCC + confusion matrix for each condition. LOSO only — no shuffled splits.

---

## 10. Evaluation Metrics

### Inside the model (already in output dims 10–11)

- `H_norm` — normalized predictive entropy `∈ [0,1]`. 0 = certain, 1 = maximally confused.
- `M` — margin between top-1 and top-2 probability. Small = two states nearly tied.

### Offline metrics (report all of these)

```python
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

# primary
macro_f1  = f1_score(y_true, y_pred, average='macro')
mcc       = matthews_corrcoef(y_true, y_pred)
kappa     = cohen_kappa_score(y_true, y_pred)

# debugging
cm        = confusion_matrix(y_true, y_pred)   # 5×5 matrix
```

### Calibration

```python
from sklearn.calibration import calibration_curve
# plot fraction_of_positives vs mean_predicted_value
# if curve deviates from diagonal → apply Platt scaling post-hoc
```

### Evaluation hierarchy (do in this order)

```
Level 1 — Offline F1 / MCC / confusion matrix
Level 2 — Temporal validity (no shuffled windows)
Level 3 — LOSO cross-validation (strict — no overlapping windows across splits)
Level 4 — Calibration check
Level 5 — Ablation table
```

---

## 11. LOSO — Strict Leakage Rules

Leave-One-Subject-Out cross-validation. For N subjects:

- Train on N-1 subjects, test on the held-out subject
- Repeat N times, average results

**Hard rules — if broken, results are invalid:**

- No window from subject X may appear in both train and test
- Sliding window overlap (50–90% is fine) — but only within one subject's data
- Per-user z-score normalization must be fit on training subjects only, never on the test subject
- `reset_subject()` must be called on the fusion model at every subject boundary

```python
for test_subject in all_subjects:
    train_subjects = [s for s in all_subjects if s != test_subject]
    model.reset_subject()   # ← mandatory
    # train on train_subjects, evaluate on test_subject
```

---

## 12. Biosignal Proxy — Optional Research Extension

> **Status:** post-deadline extension. Do not implement before the deadline. Read this section only after the core system is working.

### The idea

Train a secondary model (Path 2) that predicts biosignals (EDA, ECG, respiration — not EEG) from HCI data alone, using the D3 dataset where both HCI and biosignals are available. Use this model as a cross-validator and regularizer for the fusion model (Path 1).

```
Path 1:  HCI inputs → fusion model → cognitive state predictions
Path 2:  HCI inputs → biosignal proxy → predicted biosignals → cognitive state

Cross-validation: compare Path 1 and Path 2 state predictions
Regularization:   add KL divergence loss between Path 1 and Path 2 distributions
```

### Why it could help

- Provides an independent signal to validate fusion model against
- Reduces risk of overfitting to spurious HCI patterns
- Uses the rich D3 dataset's biosignal annotations directly

### Where to plug it in architecturally

Path 2 is a separate module. At training time it adds a regularization term to the loss. At inference time **only Path 1 runs** — no biosensor hardware required. The proxy model is training infrastructure only.

```python
# additional loss term during training only:
kl_loss = F.kl_div(
    F.log_softmax(path1_logits, dim=-1),
    F.softmax(path2_logits,     dim=-1),
    reduction='batchmean'
)
total_loss = primary_loss + lambda_kl * kl_loss
```

### Practical note

EDA and respiration prediction from HCI is tractable. EEG prediction is not — the signal is too high-dimensional and noisy. Focus the proxy on EDA + ECG only if implementing.

### Where to put it

```
predictive_models/{modality}/
└── biosignal_proxy/
    ├── proxy_model.py    ← predicts biosignals from TFN slice
    └── proxy_loss.py     ← KL divergence regularization term
```

It does not replace or modify the main predictive model. It is an additional training component only.

---

## 13. What You Are Not Allowed to Touch

| File / folder | Reason |
| --- | --- |
| `fusion_model.py` | orchestrator logic — any change breaks integration |
| `TFN/` | locked Tucker implementation — rank ablation only via constructor arg |
| `poe/poe.py` | PoE math is locked — mode switch via `set_poe_mode()` only |
| `ema/ema.py` | uncertainty math is locked |
| `TCN_encoders/buffered_encoder.py` | base class used by all modality encoders |
| `TCN_encoders/*/encoder.py` | buffer/TCN logic — only `_parse_input()` is yours |
| `predictive_models/base.py` | contract class — do not modify the output contract |
| `predictive_models/__init__.py` | registry — do not modify directly |
| Any `dummy.py` | structural placeholder — never delete |

---

## 14. Quick Checklist Before Submitting Your Model

```
□ Model is in predictive_models/{modality}/v{N}_yourmodel.py
□ Subclasses BaseModalityModel
□ Calls self.projector(x) as the first line of forward()
□ Returns (B, 12) with correct dim layout
□ Dims 5–9 are RAW logits — no softmax applied
□ Dims 10–11 computed via compute_uncertainty() from ema.ema
□ reset_microstate() clears all hidden state (GRU/LSTM h, c)
□ __init__.py in your modality folder updated to point at your model
□ test_fusion.py passes all 6 tests
□ Smoke tested with d_dims=[64, 64, 32, 32], rank=8
```

---

*Document version: AAM v3 — Fusion Model Integration Guide*  
*Contact: Haffouz (fusion / Inférer lead)*
