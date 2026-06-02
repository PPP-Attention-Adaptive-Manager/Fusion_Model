# Embedder Weight Persistence Report (mouse + keyboard)

**Scope:** add minimal, coherent **weight save / load / export** for the existing mouse and keyboard
pre-embedders. No architecture redesign, no output-dim changes, no `fusion_model.py` changes, no Tucker
rebuild, no predictive-model training, no new embedding strategies, no deterministic-projection fallback,
no LSTM/BiLSTM or `MouseEncoderP2` replacement. All additions are **additive** — existing public APIs are
untouched.

## 1. Files changed / added

**Added (no existing file modified):**

| File | Purpose |
|---|---|
| `pre_embedders/keyboard/output.py` | `load_model()` + `get_output()` for the keyboard encoder |
| `pre_embedders/keyboard/exports/keyboard_encoder/encoder.pt` | trained checkpoint (state_dict + config) |
| `pre_embedders/keyboard/exports/keyboard_encoder/model_config.json` | architecture config |
| `pre_embedders/keyboard/exports/keyboard_encoder/README.md` | export docs |
| `pre_embedders/keyboard/exports/keyboard_encoder/test_export.py` | export smoke test |
| `pre_embedders/mouse/output.py` | `load_model()` + `get_output()` for the mouse encoder |
| `pre_embedders/mouse/exports/mouse_encoder/encoder.pt` | checkpoint (state_dict + config) |
| `pre_embedders/mouse/exports/mouse_encoder/model_config.json` | architecture config |
| `pre_embedders/mouse/exports/mouse_encoder/README.md` | export docs |
| `pre_embedders/mouse/exports/mouse_encoder/test_export.py` | export smoke test |
| `scripts/embedders/train_keyboard_encoder.py` | trains existing encoder on `data/` keyboard.csv, then exports |
| `scripts/embedders/export_mouse_encoder.py` | exports `MouseEncoderP2` weights (persistence only) |
| `scripts/embedders/check_embedder_weights.py` | discovery + verification harness |
| `outputs/embedders/embedder_weight_check.json` | machine-readable check result |

**Not modified (verified via `git status`):** `fusion_model.py`, `TFN/`, `predictive_models/`,
`data_training/`, `pre_embedders/switching/`, and all existing keyboard/mouse source
(`encoder.py`, `train.py`, `dataset.py`, `preprocess.py`, `mouse_encoder.py`).

## 2. Keyboard save / load path

- **Save:** `scripts/embedders/train_keyboard_encoder.py` aggregates keystroke events from
  `data/**/keyboard.csv` via the existing `parse_csv_events`, trains the existing `KeystrokeEncoder`
  through the existing `train()` (unchanged contrastive objective), then writes
  `pre_embedders/keyboard/exports/keyboard_encoder/encoder.pt` + `model_config.json` + `README.md`.
- **Load:** `pre_embedders/keyboard/output.py::load_model()` reads `encoder.pt`, rebuilds
  `KeystrokeEncoder` from `model_config`, loads the `state_dict`, sets `eval()`.
- **Inference:** `get_output(session, events_or_window)` accepts a `(W,3)` tensor/array (used as-is),
  a list of `{code,hold,ikl}` dicts (normalized via the existing `normalize_sequence`), or `None`
  (cold start). Returns `{"embedding": (64,) float32, "metadata": {...}}`. **No L2 normalization** —
  the keyboard contract uses the encoder's native raw embedding.

This export was produced on real data: **33,994 events from 68 sessions**, LSTM variant, 8 epochs.

## 3. Mouse save / load path

- **Save:** `scripts/embedders/export_mouse_encoder.py` instantiates the existing `MouseEncoderP2`
  (architecture unchanged) and writes `encoder.pt` + `model_config.json` + `README.md`. The repo has
  **no mouse training script / objective**, so — per task scope — this delivers the *persistence
  mechanism only*. The exported weights are the **seeded initialization** (seed 42) of `MouseEncoderP2`.
  A trained state_dict can be exported in the same format via `--state-dict <path>`.
- **Load:** `pre_embedders/mouse/output.py::load_model()` reads `encoder.pt`, rebuilds `MouseEncoderP2`
  from `model_config`, loads the `state_dict`, sets `eval()`.
- **Inference:** `get_output(session, payload)` accepts `{"seq": (1,8,T)|(8,T), "stats": (1,22)|(22,),
  "pre_click_seq": (n,1,20)|None}`, adds batch dims as needed, runs the encoder, returns
  `{"embedding": (64,) float32, "metadata": {...}}`. Missing/invalid payload → `zeros(64)` with
  `cold_start=True`.

## 4. Checkpoint format

Keyboard `encoder.pt`:
```python
{
  "model_state_dict": <KeystrokeEncoder state_dict>,
  "model_config": {"input_size": 3, "hidden_size": 64, "num_layers": 2,
                   "bidirectional": false, "dropout": 0.2,
                   "window_size": 20, "stride": 10, "embedding_dim": 64},
  "encoder_class": "KeystrokeEncoder", "variant": "lstm",
  "trained": true, "n_train_events": 33994, "n_epochs": 8, "exported_at": "..."
}
```

Mouse `encoder.pt`:
```python
{
  "model_state_dict": <MouseEncoderP2 state_dict>,
  "model_config": {"encoder_class": "MouseEncoderP2", "stats_dim": 22, "tcn_out_dim": 64,
                   "click_dim": 32, "mlp_hidden": 64, "fusion_dim": 64, "embedding_dim": 64},
  "trained": false, "init_seed": 42, "exported_at": "..."
}
```
`model_config.json` mirrors the checkpoint's `model_config` in each export dir. Optimizer state is not
saved (no resume requirement). Checkpoints are loaded with `weights_only=False` (trusted local files that
intentionally carry config dicts).

## 5. `test_export.py` results

Both run standalone and pass:

```
Keyboard export smoke test OK
variant=lstm  shape=(64,)  dtype=float32
l2_norm=0.175905  max_reload_diff=0.00e+00

Mouse export smoke test OK
shape=(64,)  dtype=float32
l2_norm=5.542763  max_reload_diff=0.00e+00
```

Each asserts: shape `(64,)`, dtype `float32`, no NaN, no Inf, not all-zero, `cold_start=False`,
deterministic output across a fresh reload (`max_reload_diff = 0`), and a working cold-start path.

## 6. `check_embedder_weights.py` result

`python scripts/embedders/check_embedder_weights.py` → `outputs/embedders/embedder_weight_check.json`:

```
all_passed = true
keyboard: encoder_pt_exists, load_model_ok, inference_ok, shape_ok, dtype_ok,
          no_nan, no_inf, not_all_zero, deterministic_after_reload  → passed
          (variant=lstm, l2_norm=0.1759, max_reload_diff=0.0)
mouse:    encoder_pt_exists, load_model_ok, inference_ok, shape_ok, dtype_ok,
          no_nan, no_inf, not_all_zero, deterministic_after_reload  → passed
          (l2_norm=5.5427, max_reload_diff=0.0)
```

Exit code 0. The script exits non-zero if any embedder fails any check.

## 7. Existing imports preserved (Part 8)

Verified importable and unchanged:
```python
from pre_embedders.keyboard import KeystrokeEncoder, build_lstm_encoder, build_bilstm_encoder
from pre_embedders.keyboard import (KeystrokeWindowDataset, StreamingWindowBuffer,
                                    parse_csv_events, normalize_sequence, train, extract_embeddings)
from pre_embedders.mouse.mouse_encoder import MouseEncoderP2
```
The new `output.py` modules are additive and do not alter `__init__.py` exports.

## 8. Limitations

1. **Mouse weights are an initialized (untrained) export.** No mouse training objective exists in the
   repo and inventing one was explicitly out of scope. The export delivers the *persistence mechanism*;
   re-export with `--state-dict` once trained weights exist. `metadata`/README/`encoder.pt` mark
   `trained=false`.
2. **Keyboard training is the existing self-supervised contrastive objective** (unchanged). Contrastive
   accuracy is 1.0 because each batch sample is its own class (tiny effective contrast set) — this report
   does not address embedding *quality*, only persistence. Output is the raw (un-normalized) embedding by
   contract; its L2 norm is small (~0.18) but stable and deterministic.
3. **Windows-console note:** the existing `train.py` prints a `→` character; on cp1252 consoles run with
   `PYTHONUTF8=1` (the wrapper does not modify `train.py`).
4. This change does **not** affect `data_training/` or the all-zero Tucker issue documented separately —
   wiring these embeddings into a dataset rebuild is a separate task (intentionally out of scope here).

## 9. Commands

```bash
# Keyboard: train on recorded data + export
PYTHONUTF8=1 python scripts/embedders/train_keyboard_encoder.py --raw-data-dir data --variant lstm --epochs 8

# Mouse: export weights (seeded init; or --state-dict <trained.pt>)
python scripts/embedders/export_mouse_encoder.py

# Verify both exports
python scripts/embedders/check_embedder_weights.py

# Per-export smoke tests
python pre_embedders/keyboard/exports/keyboard_encoder/test_export.py
python pre_embedders/mouse/exports/mouse_encoder/test_export.py
```
