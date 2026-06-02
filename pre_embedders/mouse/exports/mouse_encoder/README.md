# Mouse Encoder Export (`mouse_encoder`)

Stable, reloadable weights for the mouse pre-embedder (`MouseEncoderP2`).

## Files
- `encoder.pt` — checkpoint: `model_state_dict`, `model_config`.
- `model_config.json` — architecture config (mirror of the checkpoint's `model_config`).
- `test_export.py` — smoke test (load -> dummy payload -> shape/finite/non-zero -> deterministic reload).

## Architecture (unchanged)
`MouseEncoderP2`: TCN sequence branch + pre-click branch + stats MLP -> fusion_proj -> `(B, 64)` float32.

```json
{
  "encoder_class": "MouseEncoderP2",
  "stats_dim": 22,
  "tcn_out_dim": 64,
  "click_dim": 32,
  "mlp_hidden": 64,
  "fusion_dim": 64,
  "embedding_dim": 64
}
```

## Weights
**Initialized (untrained) weights**, seeded (42) for reproducibility. No mouse training objective exists in this repo; this export delivers the persistence mechanism. Re-export with `--state-dict` once trained weights exist.

## Load / inference
```python
from pre_embedders.mouse.output import load_model, get_output
session = load_model("pre_embedders/mouse/exports/mouse_encoder", device="auto")
out = get_output(session, {"seq": seq_1x8xT, "stats": stats_1x22, "pre_click_seq": clicks_nx1x20})
emb = out["embedding"]   # (64,) float32
```

## Notes
- Missing / invalid payload -> zeros(64) with `metadata.cold_start=True`.
- Architecture, output dim (64), and the 120 s fusion/Tucker logic are unchanged.
