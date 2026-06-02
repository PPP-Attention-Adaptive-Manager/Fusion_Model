# Mouse Encoder — FINAL (`mouse_encoder_final`)

Final, trained mouse embedder for Tucker fusion. **64-D float32** output.

## Files
- `encoder.pt` — `model_state_dict`, `model_config`, `stats_scaler`, `aux_scaler`, metadata.
- `model_config.json` — architecture + objective + `trained=true`.
- `output.py` — `load_model(export_dir, device)` / `get_output(session, payload)`.
- `test_export.py` — load → dummy payload → shape/finite/non-zero → deterministic reload.
- `loss_history.csv`, `train_config.json` — reproducibility.

## Architecture
`MouseEncoderFinal` (hybrid):
- **sequence branch**: `(B, 8, T)` per-event channels
  `[speed, accel, jerk, dx, dy, is_idle, is_click, is_scroll]` → 1D-CNN
  `(8→32→64→64)` → adaptive mean-pool.
- **stats branch**: 8 handcrafted behavioral stats (standardized by the saved
  `stats_scaler`) → MLP → 64.
- **fusion**: concat → `Linear → LayerNorm → 64-D`.

## Training objective
**NT-Xent (SimCLR)** on two augmented views (jitter speed/dx/dy, event dropout,
temporal crop) **+ 0.2 × MSE** auxiliary regression of behavioral stats
`[speed_mean, click_rate, idle_ratio, n_events, distance_total, scroll_event_count]`.

## Cold-start fix
Windows are **event-count based** (256 events / stride 128, min 32) instead of the
previous 120 s windows. This guarantees enough evidence per window, removing the
cold-start zeros that failed the earlier checkpoint. **Idle is a real behavior**
(`is_idle` channel + `idle_ratio` feature), not a cold start.

## Usage
```python
from pre_embedders.mouse.exports.mouse_encoder_final import output as ms
session = ms.load_model(device="auto")
res = ms.get_output(session, {"seq": seq_8xT, "stats": raw_8_features})
emb = res["embedding"]                            # (64,) float32
```
`stats` are the RAW behavioral features in `MOUSE_FEATURES` order; `output.py`
applies the saved `stats_scaler` internally.

## Notes
- Missing seq/stats → `zeros(64)` with `metadata.cold_start=True`.
- See `outputs/embedders/final_validation/mouse_validation.json` for the verdict.
