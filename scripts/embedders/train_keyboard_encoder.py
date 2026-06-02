"""Train the keyboard KeystrokeEncoder on recorded data and EXPORT its weights.

This is a thin wrapper around the existing ``pre_embedders.keyboard.train`` — it
does NOT change the encoder architecture or the contrastive training objective.
Its only added responsibility is **weight persistence**: after training it writes

    pre_embedders/keyboard/exports/keyboard_encoder/
        encoder.pt          (model_state_dict + model_config + metadata)
        model_config.json
        README.md

Usage
-----
    python scripts/embedders/train_keyboard_encoder.py \\
        --raw-data-dir data --variant lstm --epochs 10
    # quick export from few events (CI / smoke):
    python scripts/embedders/train_keyboard_encoder.py --epochs 2 --max-sessions 8
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pre_embedders.keyboard import parse_csv_events, train

DEFAULT_EXPORT_DIR = PROJECT_ROOT / "pre_embedders" / "keyboard" / "exports" / "keyboard_encoder"
WINDOW_SIZE = 20
STRIDE = 10
EMBED_DIM = 64
NUM_LAYERS = 2


def collect_events(raw_data_dir: Path, max_sessions: int | None) -> List[Dict[str, Any]]:
    """Aggregate keystroke events across sessions' keyboard.csv files."""
    events: List[Dict[str, Any]] = []
    keyboard_csvs = sorted(raw_data_dir.glob("**/keyboard.csv"))
    used = 0
    for path in keyboard_csvs:
        try:
            rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        except OSError:
            continue
        if not rows:
            continue
        parsed = parse_csv_events(rows)
        if parsed:
            events.extend(parsed)
            used += 1
            if max_sessions is not None and used >= max_sessions:
                break
    print(f"[keyboard] collected {len(events)} events from {used} session(s)")
    return events


def export_keyboard_encoder(
    model: torch.nn.Module,
    *,
    export_dir: Path,
    variant: str,
    window_size: int,
    stride: int,
    embed_dim: int,
    num_layers: int,
    dropout: float,
    n_events: int,
    n_epochs: int,
) -> Dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "input_size": 3,
        "hidden_size": embed_dim,
        "num_layers": num_layers,
        "bidirectional": variant == "bilstm",
        "dropout": dropout,
        "window_size": window_size,
        "stride": stride,
        "embedding_dim": embed_dim,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "encoder_class": "KeystrokeEncoder",
        "variant": variant,
        "trained": n_events > 0 and n_epochs > 0,
        "n_train_events": n_events,
        "n_epochs": n_epochs,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, export_dir / "encoder.pt")
    (export_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    _write_readme(export_dir, variant, model_config, checkpoint)
    print(f"[keyboard] exported -> {export_dir / 'encoder.pt'}")
    return checkpoint


def _write_readme(export_dir: Path, variant: str, cfg: Dict[str, Any], ckpt: Dict[str, Any]) -> None:
    readme = f"""# Keyboard Encoder Export (`keyboard_encoder`)

Stable, reloadable weights for the keyboard pre-embedder (`KeystrokeEncoder`).

## Files
- `encoder.pt` — checkpoint: `model_state_dict`, `model_config`, `encoder_class`, `variant`.
- `model_config.json` — architecture config (mirror of the checkpoint's `model_config`).
- `test_export.py` — smoke test (load -> dummy window -> shape/finite/non-zero -> deterministic reload).

## Architecture (unchanged)
`KeystrokeEncoder` — {"BiLSTM" if variant == "bilstm" else "LSTM"}, input `(B, W, 3)` features
`[hold, ikl, code]`, output `(B, 64)` float32 (raw, **no L2 normalization**).

```json
{json.dumps(cfg, indent=2)}
```

variant=`{variant}`, trained=`{ckpt.get("trained")}`, n_train_events=`{ckpt.get("n_train_events")}`,
n_epochs=`{ckpt.get("n_epochs")}`.

## Load / inference
```python
from pre_embedders.keyboard.output import load_model, get_output
session = load_model("pre_embedders/keyboard/exports/keyboard_encoder", device="auto")
out = get_output(session, window_or_events)   # window: (20,3) array, or list of {{code,hold,ikl}}
emb = out["embedding"]                          # (64,) float32
```

## Notes
- Output is the encoder's native raw embedding; no L2 normalization is applied
  (the keyboard contract does not require it).
- Missing / invalid input -> zeros(64) with `metadata.cold_start=True`.
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train + export the keyboard encoder.")
    ap.add_argument("--raw-data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--variant", choices=["lstm", "bilstm"], default="lstm")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    events = collect_events(args.raw_data_dir, args.max_sessions)
    if len(events) <= args.window_size:
        raise SystemExit(
            f"Not enough keystroke events ({len(events)}) to form windows of "
            f"size {args.window_size}. Provide more data via --raw-data-dir."
        )

    model = train(
        events,
        bidirectional=(args.variant == "bilstm"),
        window_size=args.window_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    export_keyboard_encoder(
        model,
        export_dir=args.export_dir,
        variant=args.variant,
        window_size=args.window_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_events=len(events),
        n_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
