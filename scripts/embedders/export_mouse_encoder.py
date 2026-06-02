"""Export weights for the mouse ``MouseEncoderP2`` pre-embedder.

The repository contains **no mouse training script / objective**, so — per the
task scope — this script only provides *weight persistence* (export + reload),
NOT a new training objective. It instantiates the existing ``MouseEncoderP2``
(architecture unchanged) under a fixed seed and writes a stable, reloadable
checkpoint:

    pre_embedders/mouse/exports/mouse_encoder/
        encoder.pt          (model_state_dict + model_config)
        model_config.json
        README.md

If/when a trained ``MouseEncoderP2`` state_dict becomes available, pass it with
``--state-dict path.pt`` to export real trained weights instead of the seeded
initialization (same checkpoint format).

Usage
-----
    python scripts/embedders/export_mouse_encoder.py
    python scripts/embedders/export_mouse_encoder.py --state-dict trained_mouse.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pre_embedders.mouse.mouse_encoder import MouseEncoderP2

DEFAULT_EXPORT_DIR = PROJECT_ROOT / "pre_embedders" / "mouse" / "exports" / "mouse_encoder"


def build_model_config() -> Dict[str, Any]:
    return {
        "encoder_class": "MouseEncoderP2",
        "stats_dim": 22,
        "tcn_out_dim": 64,
        "click_dim": 32,
        "mlp_hidden": 64,
        "fusion_dim": 64,
        "embedding_dim": 64,
    }


def export_mouse_encoder(
    export_dir: Path,
    *,
    state_dict_path: Optional[Path],
    seed: int,
) -> Dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    config = build_model_config()
    model = MouseEncoderP2(
        stats_dim=config["stats_dim"],
        tcn_out_dim=config["tcn_out_dim"],
        click_dim=config["click_dim"],
        mlp_hidden=config["mlp_hidden"],
        fusion_dim=config["fusion_dim"],
    )

    trained = False
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        trained = True

    model.eval()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "trained": trained,
        "init_seed": None if trained else seed,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, export_dir / "encoder.pt")
    (export_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_readme(export_dir, config, trained, seed)
    print(f"[mouse] exported -> {export_dir / 'encoder.pt'}  (trained={trained})")
    return checkpoint


def _write_readme(export_dir: Path, cfg: Dict[str, Any], trained: bool, seed: int) -> None:
    weight_note = (
        "Real trained weights (loaded via --state-dict)."
        if trained
        else f"**Initialized (untrained) weights**, seeded ({seed}) for reproducibility. "
        "No mouse training objective exists in this repo; this export delivers the "
        "persistence mechanism. Re-export with `--state-dict` once trained weights exist."
    )
    readme = f"""# Mouse Encoder Export (`mouse_encoder`)

Stable, reloadable weights for the mouse pre-embedder (`MouseEncoderP2`).

## Files
- `encoder.pt` — checkpoint: `model_state_dict`, `model_config`.
- `model_config.json` — architecture config (mirror of the checkpoint's `model_config`).
- `test_export.py` — smoke test (load -> dummy payload -> shape/finite/non-zero -> deterministic reload).

## Architecture (unchanged)
`MouseEncoderP2`: TCN sequence branch + pre-click branch + stats MLP -> fusion_proj -> `(B, 64)` float32.

```json
{json.dumps(cfg, indent=2)}
```

## Weights
{weight_note}

## Load / inference
```python
from pre_embedders.mouse.output import load_model, get_output
session = load_model("pre_embedders/mouse/exports/mouse_encoder", device="auto")
out = get_output(session, {{"seq": seq_1x8xT, "stats": stats_1x22, "pre_click_seq": clicks_nx1x20}})
emb = out["embedding"]   # (64,) float32
```

## Notes
- Missing / invalid payload -> zeros(64) with `metadata.cold_start=True`.
- Architecture, output dim (64), and the 120 s fusion/Tucker logic are unchanged.
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export the mouse encoder weights.")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--state-dict", type=Path, default=None,
                    help="Optional path to a trained MouseEncoderP2 state_dict / checkpoint.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    export_mouse_encoder(args.export_dir, state_dict_path=args.state_dict, seed=args.seed)


if __name__ == "__main__":
    main()
