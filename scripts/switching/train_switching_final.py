"""Train one final switching predictive model on all available users.

LOSO creates one checkpoint per held-out user for evaluation. This script
creates the deployment-style checkpoint:

    outputs/switching_predictive/final/best.pt

It first uses a user-level validation split to choose an early-stopping epoch,
then refits a fresh model on all samples for that number of epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from predictive_models.switching.v1_gru_switching import SwitchingGRU
from scripts.switching.train_switching_predictive import (
    INPUT_FLAT_DIM,
    compute_loss,
    load_training_arrays,
    make_loader,
    prepare_targets,
    run_epoch,
    users_from_metadata,
)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _user_level_train_val(users: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unique_users = sorted(set(users.tolist()))
    if len(unique_users) < 2:
        indices = np.arange(users.shape[0])
        return indices, indices, unique_users, unique_users

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique_users))
    n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_ratio))))
    val_users = sorted(shuffled[:n_val])
    train_users = sorted(shuffled[n_val:])
    train_idx = np.where(np.isin(users, train_users))[0]
    val_idx = np.where(np.isin(users, val_users))[0]
    return train_idx, val_idx, train_users, val_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one final switching predictive model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_for_training"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/switching_predictive/final"))
    parser.add_argument("--d-proj", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    X, labels, metadata, resolved_dir = load_training_arrays(args.data_dir)
    factors, states = prepare_targets(labels)
    users = users_from_metadata(metadata)
    train_idx, val_idx, train_users, val_users = _user_level_train_val(users, args.val_ratio, args.seed)

    mean_train, std_train = _fit_normalizer(X[train_idx])
    X_train = _normalize(X[train_idx], mean_train, std_train)
    X_val = _normalize(X[val_idx], mean_train, std_train)
    train_loader = make_loader(X_train, factors[train_idx], states[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, factors[val_idx], states[val_idx], batch_size=args.batch_size, shuffle=False)

    model = SwitchingGRU(input_flat_dim=INPUT_FLAT_DIM, d_proj=args.d_proj, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 1
    patience_left = args.patience
    validation_history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        validation_history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Refit on all samples for the selected number of epochs.
    mean_all, std_all = _fit_normalizer(X)
    X_all = _normalize(X, mean_all, std_all)
    all_loader = make_loader(X_all, factors, states, batch_size=args.batch_size, shuffle=True)
    final_model = SwitchingGRU(input_flat_dim=INPUT_FLAT_DIM, d_proj=args.d_proj, hidden_dim=args.hidden_dim).to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    refit_history: List[Dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        metrics = run_epoch(final_model, all_loader, device=device, optimizer=final_optimizer)
        refit_history.append({"epoch": epoch, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "x_mean": mean_all,
            "x_std": std_all,
            "input_flat_dim": INPUT_FLAT_DIM,
            "d_proj": args.d_proj,
            "hidden_dim": args.hidden_dim,
            "best_epoch_from_validation": best_epoch,
            "validation_loss": best_val,
            "trained_on_all_samples": True,
            "all_users": sorted(set(users.tolist())),
            "validation_users_for_epoch_selection": val_users,
            "train_users_for_epoch_selection": train_users,
            "data_dir": str(resolved_dir),
        },
        checkpoint_path,
        _use_new_zipfile_serialization=False,
    )

    _write_csv(args.output_dir / "validation_history.csv", validation_history)
    _write_csv(args.output_dir / "refit_history.csv", refit_history)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(resolved_dir),
        "num_samples": int(X.shape[0]),
        "num_users": int(len(set(users.tolist()))),
        "best_epoch_from_validation": int(best_epoch),
        "validation_loss": float(best_val),
        "trained_on_all_samples": True,
        "device": str(device),
    }
    (args.output_dir / "final_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
