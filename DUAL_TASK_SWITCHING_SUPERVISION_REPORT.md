# Dual-Task Window-Level Supervision for the Switching Predictive Model

**Scope.** This experiment changes only the **labelling / supervision** of the switching predictive
model. The GraphSAGE-GAE encoder, the switching pre-embedder, the encoder export, the fusion core
(`fusion_model.py`), the TFN/Tucker generation, and the mouse/keyboard/notification models are all
**untouched**. Input to the model is still `tucker_slices[:, 3, :]` (512-d), and the model still
returns the `(B, 12)` contract; for dual-task regression we read `output[:, 0]` as the scalar load.

All numbers below are reproducible with the commands in the final section.

---

## 1. Why NASA-TLX session-level labels are weak for window-level training

NASA-TLX is answered **once per session** (post-hoc, retrospective). In `data_training/`, that single
TLX vector is copied to **every** 120s window of the session. A session with 16 windows therefore has
16 identical targets, even though the user's state plausibly changed window-to-window
(focused → distracted → idle → overloaded).

Consequences:

- The target has **zero within-session variance**, so a window-level model receives temporally
  constant, often self-contradictory supervision (different embeddings, identical label).
- The earlier NASA-based switching classifier collapsed: 5-class accuracy ≈ 0.21, macro-F1 ≈ 0.069,
  MCC ≈ −0.30, κ ≈ −0.18 (worse than chance on macro metrics). 3-class and regression-only variants
  did not fix it. The diagnosis was **labelling**, not the encoder.

## 2. Why dual-task gives better temporal supervision

The dual-task paradigm runs a secondary probe *during* the session. When the main task consumes more
cognitive resources, the secondary task degrades — measurable **at the moment of the probe**:

- `reaction_time_ms` rises,
- `miss_rate` rises,
- `error_rate` rises.

Because each probe carries its own timestamp, it can be aligned to the specific 120s window it falls
in, producing genuine **window-level** targets with within-session variance — exactly what NASA-TLX
lacks.

## 3. How window labels were computed

`scripts/switching/build_dual_task_window_labels.py`:

1. Loads `data_training/metadata.json` (N = 371 rows; aligned 1:1 with `tucker_slices.npy`).
2. For each row, resolves the session's `dual_task.csv` with a robust search
   (`./`, `raw/`, `events/`, `features/`, then recursive glob; first non-empty file wins, logged).
   In this dataset the selected file is consistently `data/<session>/raw/dual_task.csv`.
3. Selects events with `window_start <= timestamp < window_end`.
4. Computes per-window stats: `dual_task_count`, reaction-time mean/median/std/min/max,
   miss/error/success counts and rates (rates clamped to [0,1]).

Rules honoured:

- No dual-task file → `dual_task_available = false`, label fields `NaN`.
- File exists but no event in window → `dual_task_available = false`, `dual_task_count = 0`, `NaN`.
- Missed probes record `reaction_time_ms = 0`; these are **excluded from RT statistics** but still
  counted toward miss/error rates.

Outputs: `data_training/dual_task_window_labels.csv` (all columns, human-readable) and
`data_training/dual_task_window_labels.npy` (numeric columns, metadata order). The CSV also stores
**preliminary** global-scaled `dual_task_load_absolute/relative_prelim` for visualization only —
**training recomputes load with train-split-only scaling** (Section 6 / leakage rules).

## 4. Label coverage summary

| metric | value |
|---|---|
| total windows | **371** |
| windows with a dual-task probe | **97 (26.1%)** |
| windows missing labels | 274 |
| probes per covered window | **exactly 1** (distribution `{0: 274, 1: 97}`) |
| miss-only windows (no valid RT) | 28 of 97 |
| `error_rate` | **0.0 everywhere** (no errors recorded) |
| `miss_rate` | binary `{0.0, 1.0}` (one probe per window) |
| reaction_time_mean range | 900–2941 ms (median ≈ 1798 ms) |

**Coverage by user (available / total):**

| user | available | total |
|---|---|---|
| dem | 56 | 161 |
| hffz | 15 | 41 |
| layss | 6 | 64 |
| Makki | 5 | 15 |
| amrdroid | 5 | 16 |
| ghoss | 5 | 15 |
| ignorant | 5 | 16 |
| Graja | **0** | 11 |
| ladabos | **0** | 16 |
| mohanned | **0** | 16 |

Only **7 of 10 users** have any dual-task labels; **3 users are unevaluable** under LOSO. This is the
single most important fact in the report: the supervision is **sparse and shallow** (one binary-miss /
single-RT probe per labelled window, no error signal).

## 5. User imbalance summary

Windows per user are extremely skewed: **dem = 161**, layss = 64, hffz = 41, all others 11–16. Of the
*labelled* windows, dem alone holds 56 of 97 (58%). Without correction, dem dominates training. We
mitigate with **user-balanced loss** (Section 7).

## 6. Absolute vs relative target explanation

Both targets combine the same three indicators with fixed weights:

```
load = 0.60 * rt_norm + 0.25 * miss_rate + 0.15 * error_rate   (clamped to [0,1])
```

- **Absolute:** `rt_norm` = train-only robust-scaled (median/IQR → sigmoid) `reaction_time_mean`.
- **Relative (default):** removes per-user RT offsets. baseline_rt = median `reaction_time_mean` over
  that user's available windows; `relative_rt = (rt_mean − baseline_rt) / (baseline_rt + 1e-8)`; then
  train-only robust-scaled to [0,1]. This matters because baseline RT differs across users
  (e.g. a 450 ms user vs a 900 ms user are not comparable on raw RT).
- **Miss-only windows** (no valid RT, `miss_rate = 1`) have `rt_norm` imputed to **1.0** — a failed
  probe is treated as a maximum-latency / maximum-load proxy. With `error_rate ≡ 0`, such windows get
  load `≈ 0.6·1.0 + 0.25·1.0 = 0.85`.

**Leakage / personalization assumption (documented):** the train-only robust scaler is never fit on
test labels. The *per-user RT baseline* for the relative target **may** use a held-out user's own
available windows — this deliberately mimics a short personalization / calibration step and is the only
place a test user's own labels inform their target. The feature normalizer is fit on train only.

## 7. User-balanced loss explanation

`--sample-weighting user_balanced` sets `weight_i = 1 / (#available windows for user_i in the train
split)`, normalized so the mean weight is 1. The loss is `mean(weight_i · Huber(pred_i, y_i))`. This
prevents dem's 56 windows from dominating the gradient and gives small users (5–6 windows) comparable
influence.

## 8. LOSO results

`--split-mode loso`, relative target, user-balanced, GRU. 7 folds (one per labelled user; 0 skipped —
the 3 zero-coverage users are simply never testable). Aggregated over all 97 held-out windows:

| method | MAE ↓ | RMSE ↓ | Spearman ρ | R² |
|---|---|---|---|---|
| **switching model** | **0.2433** | 0.2896 | 0.258 | −0.127 |
| global_mean | 0.2844 | 0.3185 | −0.208 | −0.848 |
| train_user_mean | 0.2844 | 0.3185 | −0.208 | −0.848 |
| session_mean | 0.2844 | 0.3185 | −0.208 | −0.848 |
| previous_window | 0.2176 | 0.2973 | **0.386** | −0.606 |

Pearson r over all windows = 0.19. Per-user Pearson is mostly undefined (5–6 windows per user, near-
constant predictions). The model beats the **mean** baselines on MAE/RMSE but is **beaten on MAE by the
`previous_window` baseline**, and its R² is negative — i.e. it does not explain variance better than
predicting the test mean.

## 9. session_within_user results

`--split-mode session_within_user` (sessions split per user; no session shared across splits; users may
appear in train and test). Single fold, 36 held-out windows:

| method | MAE ↓ | RMSE ↓ | Spearman ρ | R² |
|---|---|---|---|---|
| **switching model** | **0.2001** | 0.2468 | −0.066 | −0.015 |
| global_mean | 0.2225 | 0.2561 | −0.066 | −0.094 |
| session_mean | 0.2225 | 0.2561 | −0.066 | −0.094 |
| train_user_mean | 0.2026 | **0.2383** | 0.249 | **+0.053** |
| previous_window | 0.2062 | 0.2827 | 0.304 | −0.333 |

Here the model's MAE (0.200) edges the global mean (0.222) and is roughly tied with `train_user_mean`
(0.203), but `train_user_mean` is the only predictor with **positive R²** and beats the model on RMSE.
The model's Pearson is NaN (prediction variance ≈ 0): it has essentially learned to output a near-
constant value close to the per-user mean.

## 10. Baseline comparison

Implemented in `evaluate_dual_task_switching.py`, all fit on train only, scored on test, aggregated
across folds: **global-mean**, **train-user-mean**, **session-mean**, **previous-window** (regression);
**majority-class** and **stratified-random** are covered by the binary metrics path. Results above.

## 11. Does switching beat the baselines?

**Honest answer: marginally, and not convincingly.**

- It beats the trivial **mean** baselines on MAE in both splits.
- It does **not** beat `previous_window` on MAE under LOSO, and does **not** beat `train_user_mean` on
  RMSE/R² under the within-user split.
- R² is negative (LOSO) or ≈ 0 (within-user); correlation is weak or undefined.

This is the behaviour of a model that has collapsed toward the conditional mean — there is little
exploitable window-level signal in the current labels for it to fit.

## 12. Is user personalization needed?

Within-user (0.200 MAE, R² ≈ −0.015) is modestly better than LOSO (0.243 MAE, R² ≈ −0.127), and the
`train_user_mean` baseline is the strongest single predictor within-user. That pattern points to a
**personalization / user-generalization** component: knowing the user's typical load helps more than the
switching embedding currently does. But the gap is small and the absolute quality is poor, so the
stronger conclusion is **weak signal + sparse labels**, with personalization a secondary effect.

## 13. Final recommendation

The report's required questions, answered directly:

- **Are enough dual-task labels available?** **No.** 97/371 windows (26%), one probe each,
  binary miss, zero errors, and 3 users with no labels. This is too sparse and too shallow for robust
  window-level supervision or for LOSO on most users.
- **Does switching predict dual-task load better than baselines?** **Only marginally** (beats the mean
  on MAE; loses to `previous_window` / `train_user_mean` on other metrics; negative/≈0 R²). Not a
  convincing win.
- **Does within-user beat LOSO?** **Yes, slightly** (MAE 0.200 vs 0.243), consistent with a
  personalization effect — but both are weak.
- **Is the problem user generalization or weak signal?** **Primarily weak signal / sparse labels**, with
  a secondary user-generalization component. If it were purely generalization, within-user would be
  strong; it is not.
- **Should switching remain a weak expert in fusion?** **Yes.** Keep the frozen GraphSAGE-GAE switching
  embedding as a **weak expert** with low/uncertainty-aware weight in fusion. Do not promote it to a
  confident window-level load predictor on this data.

**Next steps to make this conclusive:** (a) collect **denser dual-task probes** (several per 120s window
so RT stats and miss/error *rates* are meaningful), (b) capture genuine **error** events (currently all
zero), (c) ensure **every user** has probe coverage so LOSO is fully evaluable, and (d) re-run this exact
pipeline — it is already wired for it.

---

## Reproduce

```bash
# 1. Build window-level dual-task labels
python scripts/switching/build_dual_task_window_labels.py \
  --data-training-dir data_training --raw-data-dir data \
  --output data_training/dual_task_window_labels.csv

# 2. Train (LOSO, relative target, user-balanced)
python scripts/switching/train_switching_predictive.py \
  --data-dir data_training --task dual_task_regression \
  --dual-task-labels data_training/dual_task_window_labels.csv \
  --dual-task-target relative --split-mode loso \
  --sample-weighting user_balanced --device cuda \
  --output-dir outputs/switching_dual_task_loso

# 3. Evaluate (LOSO)
python scripts/switching/evaluate_dual_task_switching.py \
  --data-dir data_training --checkpoint-dir outputs/switching_dual_task_loso \
  --dual-task-labels data_training/dual_task_window_labels.csv \
  --split-mode loso --device cpu

# 4. Train (within-user/session split)
python scripts/switching/train_switching_predictive.py \
  --data-dir data_training --task dual_task_regression \
  --dual-task-labels data_training/dual_task_window_labels.csv \
  --dual-task-target relative --split-mode session_within_user \
  --sample-weighting user_balanced --device cuda \
  --output-dir outputs/switching_dual_task_within_user

# 5. Evaluate (within-user/session split)
python scripts/switching/evaluate_dual_task_switching.py \
  --data-dir data_training --checkpoint-dir outputs/switching_dual_task_within_user \
  --dual-task-labels data_training/dual_task_window_labels.csv \
  --split-mode session_within_user --device cpu
```

Notebook: `notebooks/dual_task_switching_supervision.ipynb` (coverage, distributions, LOSO vs within-
user, model-vs-baseline plots, per-user/per-session error).

> `--device cuda` falls back to CPU automatically when CUDA is unavailable. The numbers in this report
> were produced on CPU.
