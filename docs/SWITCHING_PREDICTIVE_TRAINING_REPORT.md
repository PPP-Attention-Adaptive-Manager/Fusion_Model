# Switching Predictive Model Training Report

Date: 2026-06-01

## 1. Data Shapes

Training folder used:

```text
data_training/
```

The scripts also accept `--data-dir data_for_training`; if that folder is not
present locally, they automatically fall back to `data_training/`.

Loaded arrays:

```text
tucker_slices.npy   shape = (371, 4, 512), dtype=float32
nasa_tlx_labels.npy shape = (371, 9),      dtype=float32
metadata.json       rows  = 371
```

## 2. Modality Index Used

Only the switching slice is used:

```python
X = tucker_slices[:, 3, :]
```

Resulting input:

```text
X shape = (371, 512)
```

Modality axis:

```text
0 = mouse
1 = keyboard
2 = notif
3 = switching
```

## 3. Model Architecture

Implemented file:

```text
predictive_models/switching/v1_gru_switching.py
```

Active integration:

```python
from .v1_gru_switching import SwitchingGRU as ActiveModel
```

Compatibility alias:

```text
predictive_models/switching/v1_switching_gru.py
```

Architecture:

```text
input x [B,512]
  -> BaseModalityModel.projector: 512 -> d_proj=256
  -> GRU: 256 -> hidden_dim=256
  -> factor_head: 256 -> 5
  -> state_head: 256 -> 5 raw logits
  -> compute_uncertainty(logits)
  -> output [B,12]
```

Output layout:

```text
dims 0-4  : [mental_demand, temporal_demand, effort, frustration, arousal_proxy]
dims 5-9  : raw logits [Flow, Neutral, Bored, Distracted, Overloaded]
dim 10    : H_norm
dim 11    : M margin
```

No softmax is applied to dims `5:10` before returning.

## 4. Target Preparation

Implemented in:

```text
scripts/switching/train_switching_predictive.py
```

Input label columns:

```text
0 mental_demand
1 physical_demand
2 temporal_demand
3 performance
4 effort
5 frustration
6 stress_self_report
7 valence
8 arousal
```

Factor targets:

```python
md = labels[:, 0]
td = labels[:, 2]
ef = labels[:, 4]
fr = labels[:, 5]
ar = (td + ef) / 2

factors = [md, td, ef, fr, ar] / 100
```

State labels:

```text
0 = Flow
1 = Neutral
2 = Bored
3 = Distracted
4 = Overloaded
```

Rules:

```python
high_demand = (md + ef) / 2 > 60
high_frust  = fr > 60
high_td     = td > 65
low_demand  = (md + ef) / 2 < 35
good_perf   = perf < 35

if high_demand and high_frust: return 4
if high_td and not high_frust: return 3
if not high_demand and good_perf and not high_frust: return 0
if low_demand and not good_perf: return 2
return 1
```

## 5. LOSO Split Details

Split key:

```text
metadata.json -> user_id
```

Number of users:

```text
10
```

For each fold:

```text
test_idx  = user_id == test_user
train_idx = users not in test_user and not in validation users
val_idx   = held-out train user(s), selected at user level
```

No random window split is used.

Normalization:

```text
mean/std fitted on X_train only
X_train, X_val, X_test transformed with train mean/std
```

## 6. Training Command

Full LOSO training was run on CPU to avoid the Windows CUDA pagefile issue seen
when loading PyTorch CUDA DLLs:

```powershell
python scripts\switching\train_switching_predictive.py --data-dir data_for_training --epochs 100 --batch-size 32 --device cpu --output-dir outputs\switching_predictive
```

Outputs:

```text
outputs/switching_predictive/fold_<user_id>/best.pt
outputs/switching_predictive/fold_<user_id>/history.csv
outputs/switching_predictive/train_loso_results.json
```

## 6.1 Final Single Model

The `fold_<user>/best.pt` files are evaluation checkpoints for LOSO
cross-validation. They are not meant to be the final deployment model.

For deployment-style use, one final model was trained on all samples:

```powershell
python scripts\switching\train_switching_final.py --data-dir data_for_training --epochs 100 --batch-size 32 --device cpu --output-dir outputs\switching_predictive\final
```

Final checkpoint:

```text
outputs/switching_predictive/final/best.pt
```

Final training summary:

```text
num_samples = 371
num_users = 10
trained_on_all_samples = true
best_epoch_from_validation = 1
validation_loss = 1.116426986560487
```

## 7. Evaluation Command

```powershell
python scripts\switching\evaluate_switching_predictive.py --data-dir data_for_training --checkpoint-dir outputs\switching_predictive --device cpu --batch-size 128
```

Outputs:

```text
outputs/switching_predictive/loso_results.json
outputs/switching_predictive/loso_summary.csv
outputs/switching_predictive/confusion_matrix.png
```

## 8. Per-User LOSO Results

| user | samples | factor MAE | factor RMSE | accuracy | macro F1 | MCC | kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| amrdroid | 16 | 0.1773 | 0.2215 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dem | 161 | 0.1346 | 0.1581 | 0.2112 | 0.0697 | 0.0000 | 0.0000 |
| ghoss | 15 | 0.2186 | 0.2746 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Graja | 11 | 0.2394 | 0.2780 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hffz | 41 | 0.3006 | 0.3162 | 0.0488 | 0.0186 | 0.0000 | 0.0000 |
| ignorant | 16 | 0.0596 | 0.0703 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ladabos | 16 | 0.2182 | 0.2626 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layss | 64 | 0.2207 | 0.3735 | 0.4062 | 0.1156 | 0.0000 | 0.0000 |
| Makki | 15 | 0.3624 | 0.3909 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mohanned | 16 | 0.4617 | 0.4651 | 1.0000 | 0.2000 | 0.0000 | 0.0000 |

Overall LOSO metrics:

```text
num_folds      = 10
num_samples    = 371
factor_mae     = 0.1998
factor_rmse    = 0.2685
state_accuracy = 0.2102
macro_f1       = 0.0695
mcc            = -0.2996
cohen_kappa    = -0.1825
per_class_f1   = [0.0, 0.3474, 0.0, 0.0, 0.0]
```

## 9. Confusion Matrix

Saved figure:

```text
outputs/switching_predictive/confusion_matrix.png
```

Matrix:

```text
              pred Flow  Neutral  Bored  Distracted  Overloaded
true Flow             0       39      0           0           0
true Neutral          0       78      0          47           0
true Bored            0       65      0           0           0
true Distracted       0      111      0           0           0
true Overloaded       0       31      0           0           0
```

## 10. test_fusion.py Result

Command:

```powershell
python test_fusion.py
```

Result:

```text
Ran 7 tests in 0.082s
OK
```

## 11. Tiny Overfit Result

Command:

```powershell
python scripts\switching\overfit_switching_tiny.py --data-dir data_for_training --device cpu --epochs 300 --samples 8
```

Result:

```text
initial_loss = 0.9886242151
final_loss   = 0.0000476969
ratio        = 0.0000482458
passed       = true
```

This confirms:

- model output contract is correct;
- target preparation is connected;
- loss can backpropagate through the switching GRU.

## 12. Known Limitations

- The classification metrics are weak in LOSO evaluation. The model mostly
  predicts `Neutral`, with some `Distracted`.
- The heuristic state labels may be imbalanced and noisy.
- Some users have very few samples, while `dem` dominates the dataset with
  161 windows.
- Validation user selection is user-level, but small user counts make validation
  unstable.
- Training was run on CPU because the current global Python CUDA install can
  hit Windows pagefile errors when loading CUDA DLLs.
- This trains only the switching predictive model on precomputed Tucker slices;
  it does not retrain the GNN, TFN, PoE, EMA, or switching pre-embedder.
