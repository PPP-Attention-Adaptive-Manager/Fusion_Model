# Keyboard Encoder Export (`keyboard_encoder`)

Stable, reloadable weights for the keyboard pre-embedder (`KeystrokeEncoder`).

## Files
- `encoder.pt` — checkpoint: `model_state_dict`, `model_config`, `encoder_class`, `variant`.
- `model_config.json` — architecture config (mirror of the checkpoint's `model_config`).
- `test_export.py` — smoke test (load -> dummy window -> shape/finite/non-zero -> deterministic reload).

## Architecture (unchanged)
`KeystrokeEncoder` — LSTM, input `(B, W, 3)` features
`[hold, ikl, code]`, output `(B, 64)` float32 (raw, **no L2 normalization**).

```json
{
  "input_size": 3,
  "hidden_size": 64,
  "num_layers": 2,
  "bidirectional": false,
  "dropout": 0.2,
  "window_size": 20,
  "stride": 10,
  "embedding_dim": 64
}
```

variant=`lstm`, trained=`True`, n_train_events=`33994`,
n_epochs=`8`.

## Load / inference
```python
from pre_embedders.keyboard.output import load_model, get_output
session = load_model("pre_embedders/keyboard/exports/keyboard_encoder", device="auto")
out = get_output(session, window_or_events)   # window: (20,3) array, or list of {code,hold,ikl}
emb = out["embedding"]                          # (64,) float32
```

## Notes
- Output is the encoder's native raw embedding; no L2 normalization is applied
  (the keyboard contract does not require it).
- Missing / invalid input -> zeros(64) with `metadata.cold_start=True`.
