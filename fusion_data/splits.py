"""Session-level split helpers for fusion training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def list_session_ids(data_dir: str | Path) -> List[str]:
    root = Path(data_dir)
    sessions = sorted(path.name for path in root.glob("session_*") if path.is_dir())
    if sessions:
        return sessions
    return [root.name] if root.exists() else []


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("Split ratios must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def split_sessions(
    data_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: str | Path = "outputs/fusion_train/splits.json",
) -> Dict[str, List[str]]:
    """Split sessions into train/val/test without session leakage."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    sessions = list_session_ids(data_dir)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sessions))

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    split = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }

    all_seen = split["train"] + split["val"] + split["test"]
    if len(all_seen) != len(set(all_seen)):
        raise AssertionError("A session appeared in more than one split.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(Path(data_dir)),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "splits": split,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return split


def load_splits(path: str | Path = "outputs/fusion_train/splits.json") -> Dict[str, List[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: list(value) for key, value in payload["splits"].items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create session-level fusion train/val/test split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_train/splits.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = split_sessions(
        args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(split, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
