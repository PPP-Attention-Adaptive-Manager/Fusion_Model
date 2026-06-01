"""Train the switching predictive model from precomputed Tucker slices.

Expected data folder:

    data_for_training/
      tucker_slices.npy      # (N, 4, 512)
      nasa_tlx_labels.npy    # (N, 9)
      metadata.json          # list with user_id/session_id/window metadata

Only the switching slice is used:

    X = tucker_slices[:, 3, :]  # (N, 512)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from predictive_models.switching.v1_gru_switching import SwitchingGRU


STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]
SWITCHING_MODALITY_INDEX = 3
INPUT_FLAT_DIM = 512


def resolve_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if path.exists():
        return path
    fallback = Path("data_training")
    if str(path) == "data_for_training" and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Training data folder not found: {path}")


def load_training_arrays(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Path]:
    root = resolve_data_dir(data_dir)
    slices = np.load(root / "tucker_slices.npy").astype(np.float32)
    labels = np.load(root / "nasa_tlx_labels.npy").astype(np.float32)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if slices.ndim != 3 or slices.shape[1:] != (4, INPUT_FLAT_DIM):
        raise ValueError(f"Expected tucker_slices shape (N,4,512), got {slices.shape}")
    if labels.ndim != 2 or labels.shape[1] != 9:
        raise ValueError(f"Expected nasa_tlx_labels shape (N,9), got {labels.shape}")
    if len(metadata) != slices.shape[0] or labels.shape[0] != slices.shape[0]:
        raise ValueError("tucker_slices, labels and metadata must have the same N.")
    return slices[:, SWITCHING_MODALITY_INDEX, :], labels, metadata, root


def derive_state_label(row: np.ndarray) -> int:
    md = float(row[0])
    td = float(row[2])
    ef = float(row[4])
    fr = float(row[5])
    perf = float(row[3])

    high_demand = (md + ef) / 2.0 > 60.0
    high_frust = fr > 60.0
    high_td = td > 65.0
    low_demand = (md + ef) / 2.0 < 35.0
    good_perf = perf < 35.0

    if high_demand and high_frust:
        return 4
    if high_td and not high_frust:
        return 3
    if not high_demand and good_perf and not high_frust:
        return 0
    if low_demand and not good_perf:
        return 2
    return 1


def prepare_targets(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    md = labels[:, 0]
    td = labels[:, 2]
    ef = labels[:, 4]
    fr = labels[:, 5]
    ar = (td + ef) / 2.0
    factors = np.stack([md, td, ef, fr, ar], axis=1).astype(np.float32) / 100.0
    factors = np.clip(factors, 0.0, 1.0).astype(np.float32)
    states = np.asarray([derive_state_label(row) for row in labels], dtype=np.int64)
    return factors, states


def sanitize_user_id(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(user_id)).strip("_") or "unknown_user"


def users_from_metadata(metadata: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(item.get("user_id", "unknown_user")) for item in metadata], dtype=object)


def loso_folds(
    users: np.ndarray,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    unique_users = sorted(set(users.tolist()))
    folds: List[Dict[str, Any]] = []
    for fold_idx, test_user in enumerate(unique_users):
        train_val_users = [user for user in unique_users if user != test_user]
        if not train_val_users:
            continue
        rng = np.random.default_rng(seed + fold_idx)
        shuffled = list(rng.permutation(train_val_users))
        if len(shuffled) == 1:
            val_users = shuffled
            train_users = shuffled
        else:
            n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_ratio))))
            val_users = sorted(shuffled[:n_val])
            train_users = sorted(shuffled[n_val:])
        folds.append(
            {
                "fold_idx": fold_idx,
                "test_user": test_user,
                "train_users": train_users,
                "val_users": val_users,
                "train_idx": np.where(np.isin(users, train_users))[0],
                "val_idx": np.where(np.isin(users, val_users))[0],
                "test_idx": np.where(users == test_user)[0],
            }
        )
    return folds


def normalize_fold(
    X: np.ndarray,
    train_idx: np.ndarray,
    *other_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    mean = X[train_idx].mean(axis=0).astype(np.float32)
    std = X[train_idx].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    transformed = [((X[idx] - mean) / std).astype(np.float32) for idx in other_indices]
    return mean, std, transformed


def make_loader(
    X: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(factors.astype(np.float32)),
        torch.from_numpy(states.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_loss(
    output: torch.Tensor,
    factors_gt: torch.Tensor,
    states_gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    factor_loss = F.huber_loss(output[:, :5], factors_gt)
    state_loss = F.cross_entropy(output[:, 5:10], states_gt)
    loss = 0.4 * factor_loss + 0.6 * state_loss
    return {"loss": loss, "factor_loss": factor_loss.detach(), "state_loss": state_loss.detach()}


def _to_device(batch, device: torch.device):
    x, factors, states = batch
    return x.to(device), factors.to(device), states.to(device)


def run_epoch(
    model: SwitchingGRU,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "factor_loss": 0.0, "state_loss": 0.0, "correct": 0.0, "count": 0.0}

    for batch in loader:
        x, factors, states = _to_device(batch, device)
        model.reset_microstate()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(x)
            losses = compute_loss(output, factors, states)
            if training:
                losses["loss"].backward()
                optimizer.step()

        n = float(states.shape[0])
        totals["count"] += n
        totals["loss"] += float(losses["loss"].detach().item()) * n
        totals["factor_loss"] += float(losses["factor_loss"].item()) * n
        totals["state_loss"] += float(losses["state_loss"].item()) * n
        totals["correct"] += float((output[:, 5:10].detach().argmax(dim=-1) == states).sum().item())

    denom = max(totals["count"], 1.0)
    return {
        "loss": totals["loss"] / denom,
        "factor_loss": totals["factor_loss"] / denom,
        "state_loss": totals["state_loss"] / denom,
        "state_accuracy": totals["correct"] / denom,
        "num_samples": denom,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train switching predictive model with LOSO CV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_for_training"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/switching_predictive"))
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-folds", type=int, default=None)
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
    folds = loso_folds(users, seed=args.seed)
    if args.limit_folds is not None:
        folds = folds[: args.limit_folds]
    if not folds:
        raise RuntimeError("No LOSO folds could be created.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_fold_summaries: List[Dict[str, Any]] = []
    for fold in folds:
        test_user = str(fold["test_user"])
        safe_user = sanitize_user_id(test_user)
        fold_dir = args.output_dir / f"fold_{safe_user}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        test_idx = fold["test_idx"]
        mean, std, (X_train, X_val, X_test) = normalize_fold(X, train_idx, train_idx, val_idx, test_idx)

        train_loader = make_loader(X_train, factors[train_idx], states[train_idx], batch_size=args.batch_size, shuffle=True)
        val_loader = make_loader(X_val, factors[val_idx], states[val_idx], batch_size=args.batch_size, shuffle=False)

        model = SwitchingGRU(input_flat_dim=INPUT_FLAT_DIM, d_proj=args.d_proj, hidden_dim=args.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val = float("inf")
        best_epoch = -1
        patience_left = args.patience
        history: List[Dict[str, Any]] = []
        checkpoint_path = fold_dir / "best.pt"

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
            val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                patience_left = args.patience
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "x_mean": mean,
                        "x_std": std,
                        "test_user": test_user,
                        "train_users": list(map(str, fold["train_users"])),
                        "val_users": list(map(str, fold["val_users"])),
                        "input_flat_dim": INPUT_FLAT_DIM,
                        "d_proj": args.d_proj,
                        "hidden_dim": args.hidden_dim,
                        "epoch": epoch,
                        "val_loss": best_val,
                        "state_names": STATE_NAMES,
                        "data_dir": str(resolved_dir),
                    },
                    checkpoint_path,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        write_csv(fold_dir / "history.csv", history)
        summary = {
            "fold": safe_user,
            "test_user": test_user,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "checkpoint": str(checkpoint_path),
        }
        all_fold_summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))

    payload = {
        "data_dir": str(resolved_dir),
        "input_shape": list(X.shape),
        "label_shape": list(labels.shape),
        "modality_index": SWITCHING_MODALITY_INDEX,
        "folds": all_fold_summaries,
        "device": str(device),
    }
    (args.output_dir / "train_loso_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
