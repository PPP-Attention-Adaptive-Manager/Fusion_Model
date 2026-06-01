# Keyboard Modality — Model Suite

Predictive model implementations for the **AAM Inférer** fusion architecture. Four architectures are provided, all hot-swappable via `__init__.py`.

---

## Quick Start

Switch the active model by uncommenting the relevant line in `__init__.py`:

```python
# predictive_models/keyboard/__init__.py

# from .v1_gru        import KeyboardGRU         as ActiveModel  # stable baseline
# from .v2_tcn        import KeyboardTCN         as ActiveModel  # fast & efficient
# from .v3_transformer import KeyboardTransformer as ActiveModel  # strong paper model
from .v4_hybrid      import KeyboardHybrid      as ActiveModel  # recommended
```

---

## Project Layout

```
predictive_models/
└── keyboard/
    ├── __init__.py        ← model switch
    ├── dummy.py
    ├── v1_gru.py          ← GRU baseline
    ├── v2_tcn.py          ← TCN model
    ├── v3_transformer.py  ← Transformer model
    ├── v4_hybrid.py       ← Hybrid TCN + Transformer
    ├── common.py          ← shared utilities
    └── train_compare.py   ← training & evaluation script
```

---

## Shared Contract

All models:

- subclass `BaseModalityModel`
- accept input of shape `(B, input_flat_dim)`
- return output of shape `(B, 12)`
- use `compute_uncertainty()` for uncertainty outputs
- support LOSO reset via `reset_microstate()`
- preserve raw logits contract
- are hot-swappable through `__init__.py`

---

## Model Overview

| Model | File | Architecture | Best For |
|---|---|---|---|
| `KeyboardGRU` | `v1_gru.py` | GRU + MLP | Stable baseline, easy debugging |
| `KeyboardTCN` | `v2_tcn.py` | Dilated TCN | Fast inference, real-time use |
| `KeyboardTransformer` | `v3_transformer.py` | Transformer encoder | Strong research model |
| `KeyboardHybrid` | `v4_hybrid.py` | TCN → Transformer | Best overall architecture |

---

## Shared Utilities — `common.py`

### `RollingSequenceMixin`

Maintains a rolling temporal context window across 1 Hz ticks.

```python
self.init_sequence_buffer(seq_len)   # initialise buffer
self.append_step(x)                  # x: (B, D)
seq = self.get_sequence(x)           # returns (B, T, D)
self.clear_history()                 # call on LOSO reset
```

### `AttentionPooling`

Learnable attention-weighted pooling over the time dimension.

```python
# Input:  (B, T, D)
# Output: (B, D)
pool = AttentionPooling(dim=256)
out  = pool(seq)
```

---

## Models

### 1 · GRU Baseline — `v1_gru.py`

A straightforward recurrent model with a persistent hidden state across steps. The simplest architecture and the recommended starting point.

```python
model = KeyboardGRU(
    input_flat_dim = 512,
    d_proj         = 256,
    hidden_dim     = 128,
    seq_len        = 16,
    num_layers     = 2,
)
```

**Output layout** `(B, 12)`:

| Indices | Content |
|---|---|
| 0–4 | workload factor predictions |
| 5–9 | cognitive state logits |
| 10 | normalised entropy H |
| 11 | uncertainty margin M |

---

### 2 · TCN Model — `v2_tcn.py`

Four dilated temporal convolutional blocks with an exponentially growing receptive field. Very fast at inference and well-suited for real-time use.

```python
model = KeyboardTCN(
    input_flat_dim = 512,
    d_proj         = 256,
    seq_len        = 16,
)
```

Dilation schedule: `1 → 2 → 4 → 8`

---

### 3 · Transformer Model — `v3_transformer.py`

A multi-head self-attention encoder with sinusoidal positional encoding and attention pooling. Captures long-range dependencies across the sequence.

```python
model = KeyboardTransformer(
    input_flat_dim = 512,
    d_proj         = 256,
    seq_len        = 24,
    nhead          = 8,
    num_layers     = 3,
)
```

---

### 4 · Hybrid TCN + Transformer — `v4_hybrid.py`

Local rhythm patterns are first extracted by a TCN (dilations `1 → 2 → 4`), then refined by a 2-layer Transformer encoder with attention pooling. Combines the strengths of both architectures.

```python
model = KeyboardHybrid(
    input_flat_dim = 512,
    d_proj         = 256,
    seq_len        = 24,
)
```

---

## Training & Evaluation

### Input Data

```
data/
├── tucker_slices.npy     # shape (N, 4, 512)
├── nasa_tlx_labels.npy
└── metadata.json
```

The keyboard modality slice is extracted as:

```python
KEYBOARD_MODALITY_IDX = 1   # → model input shape (B, 512)
```

### Targets

**Workload factors** `(N, 5)` — continuous regression targets derived from NASA-TLX:

| Index | Factor |
|---|---|
| 0 | Mental Demand |
| 1 | Temporal Demand |
| 2 | Effort |
| 3 | Frustration |
| 4 | Arousal Proxy `(TD + Effort) / 2` |

**Cognitive states** `(N,)` — 5-class classification derived from NASA-TLX thresholds:

| Class | State |
|---|---|
| 0 | Flow |
| 1 | Neutral |
| 2 | Bored |
| 3 | Distracted |
| 4 | Overloaded |

### Loss Function

```python
total_loss = 0.4 * HuberLoss(pred_factors, true_factors) \
           + 0.6 * CrossEntropy(pred_logits, true_states)
```

### Evaluation Metrics

**Classification:**

| Metric | Description |
|---|---|
| Macro F1 | Primary metric |
| Accuracy | Overall accuracy |
| MCC | Matthews Correlation Coefficient |
| Cohen's Kappa | Agreement beyond chance |
| Confusion Matrix | Per-class analysis |

**Regression (workload factors):**

| Metric | Scale |
|---|---|
| R² | — |
| MAE | 0–1 and 0–100 (NASA-TLX) |
| RMSE | 0–1 and 0–100 (NASA-TLX) |

---

## Leave-One-Subject-Out (LOSO) Evaluation

```
Train = all subjects except one
Test  = held-out subject
```

Normalisation uses train-set statistics only to prevent subject leakage:

```python
x_norm = (x - train_mean) / train_std
```

---

## Running Experiments

### Benchmark all models (LOSO)

```bash
python predictive_models/keyboard/train_compare.py \
    --data_dir data \
    --model all \
    --out_csv results.csv \
    --plot_path comparison.png
```

### Benchmark a single model

```bash
python predictive_models/keyboard/train_compare.py \
    --data_dir data \
    --model hybrid
```

### Analyse learning dynamics

```bash
python predictive_models/keyboard/train_compare.py \
    --data_dir data \
    --model hybrid \
    --curve_plot_path hybrid_curves.png \
    --epoch_csv hybrid_epochs.csv
```

### Train final global model

```bash
python predictive_models/keyboard/train_compare.py \
    --data_dir data \
    --model hybrid \
    --train_full \
    --full_ckpt_path checkpoints/keyboard_hybrid_global.pt
```

---

## Checkpoint Format

**LOSO checkpoint** (one per subject):

```python
{
    "state_dict",
    "norm_mean",
    "norm_std",
    "metrics",
    "test_user",
}
```

**Global checkpoint:**

```python
{
    "state_dict",
    "norm_mean",
    "norm_std",
    "metrics",
    "model_key",
}
```

---

## Output Files

| Flag | File | Contents |
|---|---|---|
| `--out_csv` | `results.csv` | Per-model summary metrics |
| `--out_json` | `results.json` | Full comparison metrics |
| `--plot_path` | `comparison.png` | Macro F1 / MCC / Kappa bar chart |
| `--curve_plot_path` | `curves.png` | Train/val loss + val F1 per epoch |
| `--epoch_csv` | `epochs.csv` | Per-epoch metrics for all folds |

---

## Recommended Experiment Order

**Phase 1 — Sanity baselines**

Run `gru` then `tcn` to validate label quality, temporal dynamics, and LOSO setup before committing compute to larger models.

**Phase 2 — Main research models**

Run `transformer` then `hybrid` to maximise state separation and evaluate overload/distraction transitions.

**Phase 3 — Final deployment**

Train the best model with `--train_full` on all available data.

---

## Hyperparameters

| Parameter | Recommended |
|---|---|
| `seq_len` | 16–24 |
| `d_proj` | 256 |
| `dropout` | 0.1 |
| `optimizer` | AdamW |
| `lr` | 1e-4 |
| `scheduler` | cosine decay |
| `batch_size` | 32 |
| `gradient_clip` | 1.0 |

---

## Expected Performance

| Model | Expected F1 | Stability | Compute | Notes |
|---|---|---|---|---|
| GRU | High | Very High | Low | Strong baseline |
| TCN | High | High | Very Low | Best for real-time |
| Transformer | Very High | Medium | Medium | Strong paper value |
| Hybrid | Highest | Medium | High | Best overall |

---

## Key Predictive Signals

The models are expected to derive the most signal from:

- typing burstiness and pause distribution
- correction density and typing speed volatility
- rhythm stability and cadence entropy
- interruption recovery and hesitation patterns
- sustained activity coherence
- rapid context-switch signatures

These correlate most strongly with **overload**, **distraction**, **frustration**, and **flow stability**.

---

## Recommendations

**Best engineering baseline → `KeyboardGRU`**
Stable, lightweight, easy to debug, and strong under LOSO.

**Best final model → `KeyboardHybrid`**
Combines local rhythm extraction (TCN) with long-range cognitive transition modelling (Transformer). Highest representational capacity and the recommended architecture for integration into the AAM Inférer fusion framework.