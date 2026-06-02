# Final Mouse & Keyboard Embedders — Report

**Goal:** produce mouse and keyboard encoders that are *actually useful* — trained on a real
self-supervised objective, independently validated, exportable, and ready for Tucker fusion. (Tucker is
**not** rebuilt here; fusion is **not** trained here; switching/GNN and the notif encoder are untouched.)

**Result: both embedders PASS independent validation → `tucker_rebuild_allowed = true`.**

| Embedder | status | best behavioral \|r\| | variance | cold-start | dup | trained |
|---|---|---|---|---|---|---|
| keyboard | ✅ **PASS** | 0.41 (pause_ratio) | 1.02 | 0 % | 0.00 | true |
| mouse | ✅ **PASS** | 0.72 (click_rate) | 1.03 | 0 % | 0.00 | true |

## 1. Current problem summary
Tucker is interactional (outer product of 4 modalities) → a zero or degenerate embedder zeroes/contaminates
**all** slices. Previously: keyboard embedding was near-degenerate (trivial objective) and mouse had only
seeded (untrained) weights → both blocked a valid fusion. This task fixes both with real learned encoders.

## 2. Why the previous keyboard objective failed
The old contrastive loss used `labels = arange(batch)` with **no positive pairs** — each sample was its own
class. This gives fake 100 % accuracy and learns no behavioral structure. Validation confirmed it:
variance ≈ 5.6e-5, L2 ≈ 0.056, best behavioral \|r\| ≈ 0.07, consecutive cosine ≈ 0.034.

## 3. Why the previous mouse checkpoint failed
It was a **seeded initialization** (`trained=false`): never trained, near-constant output, and the
120 s-window extraction produced ~80 % cold-start zeros. An untrained encoder can never be allowed into
Tucker.

## 4. Final keyboard architecture
`KeyboardEncoderFinal` — **BiGRU(2×64)** over `(B, 20, 3)` keystroke windows `[hold, ikl, code]` →
last-step concat → `Linear → LayerNorm → 64-D`. SimCLR projection head + auxiliary behavioral head used in
training only; exported embedding = 64-D backbone.

## 5. Final keyboard loss
**NT-Xent (SimCLR)** on two augmented views (jitter hold/ikl, mask key code, event dropout) — real
positive pairs — **+ 0.2 × MSE** auxiliary regression of `[typing_speed, mean_hold, mean_ikl, std_hold,
std_ikl]`. The auxiliary head directly forces behavioral structure into the embedding (the previous gap).

## 6. Final keyboard training result
3303 windows / 63 sessions, 80/20 **session** split, 60 epochs (early-stop @50), best val 2.51 @ ep30.
Contrastive accuracy **0.20 → 0.86** (genuine instance discrimination, not the old fake 1.0).
Artifacts: `pre_embedders/keyboard/exports/keyboard_encoder_final/` (`encoder.pt`, `model_config.json`,
`loss_history.csv`, `train_config.json`).

## 7. Final keyboard validation result  (`outputs/embedders/final_validation/keyboard_validation.json`)
| metric | value | vs previous |
|---|---|---|
| variance_mean | **1.02** | 5.6e-5 (×~18000) |
| best behavioral \|r\| | **0.41** (pause_ratio) | 0.07 |
| features with \|r\| ≥ 0.2 | 5 / 7 (mean_hold .28, std_hold .36, mean_ikl .20, unique_key_ratio .32) | 0 |
| cold-start ratio | 0 % | — |
| duplicate_ratio | 0.00 | — |
| effective rank | 16.8 | — |
| consecutive cosine | 0.375 (healthy) | 0.034 |
→ **PASS**, `allowed_for_tucker = true`.

## 8. Final mouse architecture
`MouseEncoderFinal` (hybrid): **sequence branch** = 1D-CNN `(8→32→64→64)` + adaptive mean-pool over
`(B, 8, T)` channels `[speed, accel, jerk, dx, dy, is_idle, is_click, is_scroll]`; **stats branch** =
MLP over 8 standardized behavioral stats; **fusion** → `Linear → LayerNorm → 64-D`.

## 9. Final mouse loss
**NT-Xent (SimCLR)** on two augmented views (jitter speed/dx/dy, event dropout, temporal crop) **+ 0.2 ×
MSE** auxiliary regression of `[speed_mean, click_rate, idle_ratio, n_events, distance_total,
scroll_event_count]`.

**Cold-start fix:** event-count windows (256 events / stride 128, min 32) instead of 120 s windows →
guarantees evidence per window. Idle is a real behavior (`is_idle` channel + `idle_ratio`), not a cold
start.

## 10. Final mouse training result
16943 windows / 72 sessions (event-count windowing solved the data scarcity), 80/20 session split, 50
epochs (early-stop @24), best val 2.22 @ ep4; aux MSE → 0.018 (behavioral stats well captured).
Artifacts: `pre_embedders/mouse/exports/mouse_encoder_final/`.

## 11. Final mouse validation result  (`outputs/embedders/final_validation/mouse_validation.json`)
| metric | value |
|---|---|
| variance_mean | 1.03 |
| best behavioral \|r\| | **0.72** (click_rate) |
| other \|r\| | idle_ratio 0.65, distance_total 0.64, speed_mean 0.48, speed_std 0.45, speed_max 0.48 |
| cold-start ratio | 0 % (was ~80 %) |
| duplicate_ratio | 0.00 |
| effective rank | 33.2 |
| consecutive cosine | 0.715 (not collapsed) |
| n_valid | 16943 |
→ **PASS**, `allowed_for_tucker = true`.
> `n_events` and `scroll_event_count` show ≈0 correlation — expected: `n_events` is ~constant by
> construction (event-count windows) and scroll is sparse; neither blocks PASS (other features clear 0.2).

## 12. Notebook paths
- `notebooks/train_keyboard_final_embedder.ipynb`
- `notebooks/train_mouse_final_embedder.ipynb`

Config-cell driven (epochs, batch, lr, device, `RUN_TRAINING`). Plot loss curves, contrastive/aux
components, contrastive accuracy, embedding norm + PCA(2D) colored by behavior, per-feature correlation
bar chart, and the PASS/WARNING/FAIL gate. No manual code edits needed for normal use.

## 13. Is keyboard allowed into Tucker?
**Yes** — PASS on all criteria (trained, no NaN/Inf, n_valid 3303, cold-start 0 %, dup 0.00, variance ≫
previous, best \|r\| 0.41 ≥ 0.2).

## 14. Is mouse allowed into Tucker?
**Yes** — PASS (trained, no NaN/Inf, n_valid 16943, cold-start 0 %, dup 0.00, best \|r\| 0.72 ≥ 0.2,
temporal cosine 0.715 — not collapsed).

→ Combined gate: **`tucker_rebuild_allowed = true`**
(`outputs/embedders/final_validation/embedder_validation_summary.json`).

## 15. Remaining limitations
- **No human-labelled ground truth** for embedder quality — validation is self-supervised + behavioral
  proxy correlation, which is the right gate for an unsupervised embedder but not a task-accuracy proof.
- Optional NASA probe (`scripts/embedders/probe_embedders.py`) is **analysis only**, never a readiness
  criterion.
- Mouse contrastive accuracy saturates fast (windows are easily distinguishable); the *useful* signal is
  the auxiliary behavioral structure and the validation correlations, not the contrastive accuracy alone.
- `n_events`/`scroll` carry little embedding correlation (constant/sparse) — acceptable, but mouse scroll
  behavior is under-represented in this corpus.
- These embedders are trained per-window and pooled over time; they were validated independently — the
  *interaction* quality only becomes testable after the (separate, still-gated) Tucker rebuild.

## 16. Reproduce
```powershell
# train
python scripts/embedders/train_keyboard_final.py --data-dir data --epochs 100 --batch-size 128 --device cuda --output-dir pre_embedders/keyboard/exports/keyboard_encoder_final
python scripts/embedders/train_mouse_final.py    --data-dir data --epochs 100 --batch-size 128 --device cuda --output-dir pre_embedders/mouse/exports/mouse_encoder_final
# validate (Tucker gate)
python scripts/embedders/validate_embedders_final.py --data-dir data --device cuda \
  --keyboard-export pre_embedders/keyboard/exports/keyboard_encoder_final \
  --mouse-export pre_embedders/mouse/exports/mouse_encoder_final \
  --output-dir outputs/embedders/final_validation
# export smoke tests
python pre_embedders/keyboard/exports/keyboard_encoder_final/test_export.py
python pre_embedders/mouse/exports/mouse_encoder_final/test_export.py
# optional downstream probe (analysis only)
python scripts/embedders/probe_embedders.py --modality keyboard --device cpu
python scripts/embedders/probe_embedders.py --modality mouse --device cpu
```

> Note (Windows): the legacy keyboard `train.py` prints a non-cp1252 char; the FINAL scripts don't, but if
> you reuse older scripts run with `$env:PYTHONUTF8=1`. The numbers above were produced on CPU; CUDA is
> supported via `--device cuda`.
