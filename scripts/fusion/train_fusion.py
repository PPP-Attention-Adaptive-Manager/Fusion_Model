"""Train the full fusion model on session-level splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fusion_data.dataset import (
    FusionWindowDataset,
    SessionBatchSampler,
    fusion_collate_fn,
)
from fusion_model import DEFAULT_D_DIMS, InferrerFusion
from fusion_data.splits import load_splits, split_sessions


def compute_fusion_loss(
    output: Dict[str, Any],
    factor_labels: torch.Tensor,
    state_label: torch.Tensor,
    factor_weight: float = 0.4,
    state_weight: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """Compute the requested fusion loss without applying softmax before CE."""

    global_factor_pred = output["global"][:, :5]
    global_state_logits = output["global"][:, 5:10]
    factor_loss = F.huber_loss(global_factor_pred, factor_labels)
    state_loss = F.cross_entropy(global_state_logits, state_label)
    total_loss = factor_weight * factor_loss + state_weight * state_loss
    return {
        "loss": total_loss,
        "factor_loss": factor_loss.detach(),
        "state_loss": state_loss.detach(),
    }


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in ["mouse", "keyboard", "notif", "switching", "factors", "state_label"]:
        moved[key] = batch[key].to(device)
    return moved


def _make_loader(
    dataset: FusionWindowDataset,
    batch_size: int,
    *,
    shuffle_sessions: bool,
    seed: int,
) -> DataLoader:
    sampler = SessionBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle_sessions=shuffle_sessions,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=fusion_collate_fn)


def _run_epoch(
    model: InferrerFusion,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_factor_loss = 0.0
    total_state_loss = 0.0
    total_samples = 0
    correct = 0
    factor_abs_error = 0.0

    for batch in loader:
        # Avoid hidden-state leakage across sessions and batch positions.
        model.reset_subject()
        batch = _batch_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model([batch["mouse"], batch["keyboard"], batch["notif"], batch["switching"]])
            losses = compute_fusion_loss(output, batch["factors"], batch["state_label"])
            if training:
                losses["loss"].backward()
                optimizer.step()

        batch_size = batch["state_label"].shape[0]
        total_samples += batch_size
        total_loss += float(losses["loss"].detach().item()) * batch_size
        total_factor_loss += float(losses["factor_loss"].item()) * batch_size
        total_state_loss += float(losses["state_loss"].item()) * batch_size

        state_scores = output["global"][:, 5:10].detach()
        pred = state_scores.argmax(dim=-1)
        correct += int((pred == batch["state_label"]).sum().item())
        factor_abs_error += float(torch.abs(output["global"][:, :5].detach() - batch["factors"]).sum().item())

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "factor_loss": total_factor_loss / denom,
        "state_loss": total_state_loss / denom,
        "state_accuracy": correct / denom,
        "factor_mae": factor_abs_error / (denom * 5),
        "num_samples": float(total_samples),
    }


def _write_history(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train InferrerFusion on session-level splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=Path, default=Path("outputs/fusion_train/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion_train"))
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if args.splits.is_file():
        splits = load_splits(args.splits)
    else:
        splits = split_sessions(args.data_dir, seed=args.seed, output_path=args.splits)

    train_dataset = FusionWindowDataset(
        args.data_dir,
        session_ids=splits["train"],
        limit_samples=args.limit_samples,
        switching_device="cpu",
    )
    val_dataset = FusionWindowDataset(
        args.data_dir,
        session_ids=splits["val"],
        limit_samples=args.limit_samples,
        switching_device="cpu",
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    if len(val_dataset) == 0:
        val_dataset = train_dataset

    train_loader = _make_loader(train_dataset, args.batch_size, shuffle_sessions=True, seed=args.seed)
    val_loader = _make_loader(val_dataset, args.batch_size, shuffle_sessions=False, seed=args.seed)

    model = InferrerFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history: list[Dict[str, Any]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "best.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "d_dims": DEFAULT_D_DIMS,
                    "splits": splits,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    _write_history(args.output_dir / "loss_history.csv", history)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint_path),
        "train_dataset": train_dataset.summary(),
        "val_dataset": val_dataset.summary(),
        "device": str(device),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
