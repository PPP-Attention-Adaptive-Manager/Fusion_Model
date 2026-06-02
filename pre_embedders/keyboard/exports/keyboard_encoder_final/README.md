# Keyboard Encoder — FINAL (`keyboard_encoder_final`)

Final, validated keyboard embedder for Tucker fusion. **64-D float32** output.

## Files
- `encoder.pt` — `model_state_dict`, `model_config`, `aux_scaler`, training metadata.
- `model_config.json` — architecture + objective + `trained=true`.
- `output.py` — `load_model(export_dir, device)` / `get_output(session, payload)`.
- `test_export.py` — load → dummy window → shape/finite/non-zero → deterministic reload.
- `loss_history.csv`, `train_config.json` — reproducibility.

## Architecture
`KeyboardEncoderFinal`: **BiGRU(2×64)** over `(B, W=20, 3)` keystroke windows
`[hold, ikl, code]` → last-step concat → `Linear → LayerNorm → 64-D`. A SimCLR
projection head and an auxiliary behavioral-feature head are used **only during
training**; the exported embedding is the 64-D backbone (representation before the
projection head).

## Training objective
**NT-Xent (SimCLR)** on two augmented views of each window (real positive pairs:
jitter hold/ikl, mask key code, event dropout) **+ 0.2 × MSE** auxiliary regression
of behavioral features `[typing_speed, mean_hold, mean_ikl, std_hold, std_ikl]`.
This replaces the previous **trivial** contrastive objective (each sample = its own
class) that produced a near-degenerate embedding.

## Usage
```python
from pre_embedders.keyboard.exports.keyboard_encoder_final import output as kb
session = kb.load_model(device="auto")
res = kb.get_output(session, window_20x3)        # or a list of {code,hold,ikl} dicts
emb = res["embedding"]                            # (64,) float32
```

## Notes
- Missing/invalid input → `zeros(64)` with `metadata.cold_start=True`. A real (even
  short) window always produces a non-zero embedding.
- See `outputs/embedders/final_validation/keyboard_validation.json` for the
  PASS/WARNING/FAIL verdict and `allowed_for_tucker`.
