# Fusion Training Pipeline Report

Date: 2026-06-01

This document explains the runtime dataset, split, smoke, training, overfit,
and evaluation pipeline added for the full fusion model.

## 1. Constraints Preserved

The switching/GNN module is not modified.

Preserved contract:

```text
GraphSAGE-GAE encoder
  -> 64D L2-normalized embedding
  -> SwitchingBufferedEncoder identity-only
  -> tensor [1,64]
```

Fusion dimensions remain:

```python
d_dims = [64, 64, 32, 64]
# mouse, keyboard, notif, switching
```

The GNN decoder is not used during fusion inference.

## 2. New Files

```text
fusion_data/dataset.py
fusion_data/splits.py
scripts/fusion/run_fusion_dataset_smoke.py
scripts/fusion/train_fusion.py
scripts/fusion/overfit_tiny_batch.py
scripts/fusion/evaluate_fusion.py
tests/test_fusion_shapes.py
docs/FUSION_TRAINING_REPORT.md
```

## 3. Data Assumptions

Expected session layout:

```text
data/
  session_x/
    raw/labels.csv
    labels/nasa_tlx.csv
    features/
      mouse/features_mouse_120s.csv
      keyboard/features_keyboard_120s.csv
      system/features_system_120s.csv
    data_graph/
      data_graph_120s/
        graph_001.json
        graph_002.json
```

The builder creates one sample per 120-second graph window.

Sample contract:

```python
{
    "mouse": Tensor(64),
    "keyboard": Tensor(64),
    "notif": Tensor(32),
    "switching": Tensor(64),
    "factors": Tensor(5),
    "state_label": Tensor scalar long,
    "metadata": {
        "session_id": str,
        "window_id": str,
        "window_start": float,
        "window_end": float
    }
}
```

## 4. Modality Inputs

### Switching

Switching uses the real exported GraphSAGE encoder:

```text
graph_*.json
  -> pre_embedders/switching/encoder.py
  -> exported output.py
  -> encoder.pt
  -> embedding [64] float32
  -> SwitchingBufferedEncoder(mode="identity")
  -> Tensor(64)
```

Validated properties:

```text
shape = [64]
dtype = float32
L2 norm ~= 1.0 unless cold_start=true
decoder_used_at_inference=false
```

### Notification

Notification uses:

```text
features/system/features_system_120s.csv
  -> notif ONNX embedding [16] if onnxruntime/model available
  -> NotifBufferedEncoder
  -> Tensor(32)
```

If ONNX is unavailable, the builder uses a zero 16D embedding fallback and
marks the metadata with:

```text
embedding_source = zero_fallback
```

### Mouse and Keyboard

The builder supports real 64D embedding columns if they exist in a CSV, using
common names such as:

```text
embedding
emb_0 ... emb_63
mouse_emb_0 ... mouse_emb_63
keyboard_emb_0 ... keyboard_emb_63
```

Current known limitation:

```text
features_mouse_120s.csv and features_keyboard_120s.csv contain engineered
features, not trained 64D embeddings. Therefore the builder falls back to
zero vectors for mouse/keyboard until their embedders are wired.
```

The fallback is explicit and counted in dataset summaries.

## 5. Labels

Factor labels:

```text
mental_demand
temporal_demand
effort
frustration
arousal
```

Values are normalized to `[0,1]`. If a value is larger than `1.0`, it is
treated as a 0-100 score and divided by 100.

State labels:

```text
0 = Flow
1 = Neutral
2 = Bored
3 = Distracted
4 = Overloaded
```

`derive_state_label(row)` uses configurable heuristics:

```text
Overloaded: mental_demand high and frustration high
Flow: performance good and frustration low and effort moderate
Bored: mental_demand low and arousal low
Distracted: temporal_demand high or fragmentation/interruption proxy high
Neutral: otherwise
```

Default thresholds are in `LabelThresholds` inside `fusion_data/dataset.py`.

## 6. Session-Level Split

Implemented in:

```text
fusion_data/splits.py
```

Command:

```powershell
python -m fusion_data.splits --data-dir data
```

Output:

```text
outputs/fusion_train/splits.json
```

Rules:

- split is by `session_id`;
- no session appears in more than one split;
- default ratios are `0.70 / 0.15 / 0.15`;
- split is deterministic with `seed=42`.

## 7. Runtime Smoke Test

Command:

```powershell
python scripts/fusion/run_fusion_dataset_smoke.py --data-dir data --limit-sessions 10
```

Output:

```text
outputs/fusion_train/dataset_smoke_summary.json
```

Checks:

- `mouse` shape `[1,64]`;
- `keyboard` shape `[1,64]`;
- `notif` shape `[1,32]`;
- `switching` shape `[1,64]`;
- no NaN/Inf;
- `fusion.global` shape `[1,11]`;
- `per_model` contains 4 tensors `[1,12]`;
- switching norms are tracked;
- cold starts are counted.

## 8. Training

Command:

```powershell
python scripts/fusion/train_fusion.py --data-dir data --epochs 50 --batch-size 8 --device cuda
```

If CUDA is unavailable, the script falls back to CPU.

Main model:

```text
InferrerFusion
  -> LowRankTuckerFusion
  -> predictive_models
  -> PoE
  -> EMA
```

Loss:

```python
global_factor_pred = output["global"][:, :5]
global_state_logits = output["global"][:, 5:10]

factor_loss = Huber(global_factor_pred, factor_labels)
state_loss = CrossEntropy(global_state_logits, state_label)

total_loss = 0.4 * factor_loss + 0.6 * state_loss
```

No softmax is applied before `CrossEntropy`.

Training outputs:

```text
outputs/fusion_train/checkpoints/best.pt
outputs/fusion_train/loss_history.csv
outputs/fusion_train/metrics.json
```

## 9. Tiny Overfit Test

Command:

```powershell
python scripts/fusion/overfit_tiny_batch.py --data-dir data --epochs 200
```

Pass condition:

```text
final_loss < 0.2 * initial_loss
```

This detects broken gradient flow, wrong shapes, or a bad loss connection.

## 10. Evaluation

Command:

```powershell
python scripts/fusion/evaluate_fusion.py --checkpoint outputs/fusion_train/checkpoints/best.pt --data-dir data --split test
```

Output:

```text
outputs/fusion_train/eval_metrics.json
```

Metrics:

- factor MAE;
- factor RMSE;
- state accuracy;
- macro F1;
- MCC;
- confusion matrix;
- entropy mean;
- margin mean.

## 11. Shape Contract Tests

Command:

```powershell
python -m unittest tests.test_fusion_shapes
```

Also useful:

```powershell
python test_fusion.py
python -m unittest tests.test_switching_encoder_integration
```

## 12. Known Limitations

Current limitations:

- `mouse` and `keyboard` real 64D embedders are not yet wired into the dataset,
  so they use zero fallback unless matching embedding columns exist.
- The GNN switching encoder is frozen and used only for embeddings, as required.
- The current training batches are session-split safe, but the model microstate
  is reset per batch to avoid hidden-state leakage through mixed batch positions.
- `SwitchingGRU` is wired but not yet trained before running `scripts/fusion/train_fusion.py`.

## 13. Recommended Next Step

Run in this order:

```powershell
python scripts/fusion/run_fusion_dataset_smoke.py --data-dir data --limit-sessions 10
python scripts/fusion/overfit_tiny_batch.py --data-dir data --epochs 200
python scripts/fusion/train_fusion.py --data-dir data --epochs 50 --batch-size 8 --device cuda
python scripts/fusion/evaluate_fusion.py --checkpoint outputs/fusion_train/checkpoints/best.pt --data-dir data --split test
```

Once mouse and keyboard embedders are available, replace their zero fallback
with real 64D vectors and rerun the same commands.
