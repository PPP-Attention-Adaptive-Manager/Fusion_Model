"""Evaluate LOSO checkpoints and diagnostic baselines for switching predictive models."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from predictive_models.switching.v1_gru_switching import SwitchingGRU
from predictive_models.switching.v1_mlp_switching import SwitchingMLP
from scripts.switching.train_switching_predictive import (
    INPUT_FLAT_DIM,
    STATE_NAMES_5CLASS,
    compute_class_weights,
    load_training_arrays,
    loso_folds,
    make_loader,
    normalize_fold,
    num_states_for_mode,
    prepare_targets,
    run_epoch,
    state_names_for_mode,
    users_from_metadata,
)


def _finite_metric(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_states: int) -> Dict[str, Any]:
    labels = list(range(num_states))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    return {
        "state_accuracy": _finite_metric(accuracy_score(y_true, y_pred)),
        "macro_f1": _finite_metric(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": _finite_metric(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": _finite_metric(kappa),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def _infer_state_mode(checkpoint_paths: List[Path], requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if not checkpoint_paths:
        return "5class"
    checkpoint = _torch_load(checkpoint_paths[0], device)
    return str(checkpoint.get("state_mode", "5class"))


def _predict(
    checkpoint_path: Path,
    X: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    users: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    fallback_state_mode: str,
) -> Dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path, device)
    test_user = str(checkpoint["test_user"])
    idx = np.where(users == test_user)[0]
    if idx.size == 0:
        raise ValueError(f"No samples found for test_user={test_user!r}")

    state_mode = str(checkpoint.get("state_mode", fallback_state_mode))
    num_states = int(checkpoint.get("num_states", num_states_for_mode(state_mode)))
    state_names = list(checkpoint.get("state_names", state_names_for_mode(state_mode)))

    mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    X_test = ((X[idx] - mean) / std).astype(np.float32)
    y_factors = factors[idx]
    y_states = states[idx]

    model = SwitchingGRU(
        input_flat_dim=int(checkpoint.get("input_flat_dim", INPUT_FLAT_DIM)),
        d_proj=int(checkpoint.get("d_proj", 256)),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        num_states=num_states,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    factor_preds: List[np.ndarray] = []
    state_preds: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    h_all: List[np.ndarray] = []
    margin_all: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], batch_size):
            batch = torch.from_numpy(X_test[start : start + batch_size]).to(device)
            model.reset_microstate()
            output = model(batch)
            factor_preds.append(output[:, :5].detach().cpu().numpy())
            logits = output[:, 5 : 5 + num_states].detach().cpu().numpy()
            logits_all.append(logits)
            state_preds.append(np.argmax(logits, axis=1))
            h_all.append(output[:, 5 + num_states].detach().cpu().numpy())
            margin_all.append(output[:, 6 + num_states].detach().cpu().numpy())

    y_factor_pred = np.vstack(factor_preds)
    y_state_pred = np.concatenate(state_preds)
    logits_np = np.vstack(logits_all)
    factor_error = y_factor_pred - y_factors
    metrics = _classification_metrics(y_states, y_state_pred, num_states)

    return {
        "test_user": test_user,
        "num_samples": int(idx.size),
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        **metrics,
        "y_true": y_states.tolist(),
        "y_pred": y_state_pred.tolist(),
        "factor_targets": y_factors.tolist(),
        "factor_preds": y_factor_pred.tolist(),
        "logits": logits_np.tolist(),
        "h_norm": np.concatenate(h_all).tolist(),
        "margin": np.concatenate(margin_all).tolist(),
        "checkpoint": str(checkpoint_path),
        "state_mode": state_mode,
        "state_names": state_names,
        "num_states": num_states,
        "task": checkpoint.get("task", "joint"),
        "class_weight_mode": checkpoint.get("class_weight_mode", "none"),
        "state_loss": checkpoint.get("state_loss", "ce"),
    }


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "test_user",
        "num_samples",
        "factor_mae",
        "factor_rmse",
        "state_accuracy",
        "macro_f1",
        "mcc",
        "cohen_kappa",
        "state_mode",
        "task",
        "class_weight_mode",
        "state_loss",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_confusion_matrix(path: Path, cm: np.ndarray, state_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Switching predictive LOSO confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=45, ha="right")
    ax.set_yticklabels(state_names)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(counts.size))


def _write_user_distribution_reports(
    output_dir: Path,
    metadata: List[Dict[str, Any]],
    states: np.ndarray,
    fold_results: List[Dict[str, Any]],
    state_names: List[str],
) -> None:
    num_states = len(state_names)
    users = users_from_metadata(metadata)
    label_rows: List[Dict[str, Any]] = []
    for user in sorted(set(users.tolist())):
        idx = np.where(users == user)[0]
        counts = np.bincount(states[idx], minlength=num_states)
        sessions = {
            str(metadata[i].get("session_id", "unknown_session"))
            for i in idx
        }
        majority_idx = int(np.argmax(counts)) if counts.sum() else 0
        row: Dict[str, Any] = {
            "user_id": user,
            "n_samples": int(idx.size),
            "n_sessions": int(len(sessions)),
            "majority_class": state_names[majority_idx],
            "entropy": _entropy_from_counts(counts),
        }
        row.update({state_names[i]: int(counts[i]) for i in range(num_states)})
        label_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_fields = ["user_id", "n_samples", "n_sessions", *state_names, "majority_class", "entropy"]
    with (output_dir / "user_label_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        writer.writerows(label_rows)

    pred_rows: List[Dict[str, Any]] = []
    confusion_by_user: Dict[str, Any] = {}
    for fold in fold_results:
        user = str(fold["test_user"])
        y_true = np.asarray(fold["y_true"], dtype=np.int64)
        y_pred = np.asarray(fold["y_pred"], dtype=np.int64)
        pred_counts = np.bincount(y_pred, minlength=num_states)
        metrics = _classification_metrics(y_true, y_pred, num_states)
        row = {
            "user_id": user,
            "n_samples": int(y_true.size),
            "state_accuracy": metrics["state_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
            "cohen_kappa": metrics["cohen_kappa"],
        }
        row.update({f"pred_{state_names[i]}": int(pred_counts[i]) for i in range(num_states)})
        pred_rows.append(row)
        confusion_by_user[user] = {
            "state_names": state_names,
            "confusion_matrix": metrics["confusion_matrix"],
            "metrics": {k: metrics[k] for k in ["state_accuracy", "macro_f1", "mcc", "cohen_kappa"]},
        }

    pred_fields = [
        "user_id",
        "n_samples",
        *[f"pred_{name}" for name in state_names],
        "state_accuracy",
        "macro_f1",
        "mcc",
        "cohen_kappa",
    ]
    with (output_dir / "user_prediction_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(pred_rows)
    (output_dir / "user_confusion_matrices.json").write_text(
        json.dumps(confusion_by_user, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _aggregate_baseline(name: str, fold_rows: List[Dict[str, Any]], num_states: int) -> Dict[str, Any]:
    all_true = np.asarray([label for row in fold_rows for label in row["y_true"]], dtype=np.int64)
    all_pred = np.asarray([label for row in fold_rows for label in row["y_pred"]], dtype=np.int64)
    metrics = _classification_metrics(all_true, all_pred, num_states)
    payload = {
        "name": name,
        "num_samples": int(all_true.size),
        **metrics,
        "folds": fold_rows,
    }
    factor_targets = [row for fold in fold_rows for row in fold.get("factor_targets", [])]
    factor_preds = [row for fold in fold_rows for row in fold.get("factor_preds", [])]
    if factor_targets and factor_preds:
        targets = np.asarray(factor_targets, dtype=np.float32)
        preds = np.asarray(factor_preds, dtype=np.float32)
        error = preds - targets
        payload["factor_mae"] = float(np.mean(np.abs(error)))
        payload["factor_rmse"] = float(np.sqrt(np.mean(error**2)))
    return payload


def _evaluate_mlp_fold(
    fold: Dict[str, Any],
    X: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    *,
    num_states: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    class_weight_mode: str,
    state_loss: str,
    task: str,
    focal_gamma: float,
) -> Dict[str, Any]:
    train_idx = fold["train_idx"]
    val_idx = fold["val_idx"]
    test_idx = fold["test_idx"]
    mean, std, (X_train, X_val, X_test) = normalize_fold(X, train_idx, train_idx, val_idx, test_idx)
    train_loader = make_loader(X_train, factors[train_idx], states[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, factors[val_idx], states[val_idx], batch_size=batch_size, shuffle=False)

    class_weight_np = compute_class_weights(states[train_idx], num_states, class_weight_mode)
    class_weight_tensor = (
        torch.from_numpy(class_weight_np).to(device)
        if class_weight_np is not None
        else None
    )
    torch.manual_seed(seed + int(fold["fold_idx"]))
    model = SwitchingMLP(input_flat_dim=INPUT_FLAT_DIM, hidden_dim=256, num_states=num_states).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = patience

    for epoch in range(1, epochs + 1):
        run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            num_states=num_states,
            class_weights=class_weight_tensor,
            state_loss=state_loss,
            task=task,
            focal_gamma=focal_gamma,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            num_states=num_states,
            class_weights=class_weight_tensor,
            state_loss=state_loss,
            task=task,
            focal_gamma=focal_gamma,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_left = patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    factor_preds: List[np.ndarray] = []
    state_preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], batch_size):
            batch = torch.from_numpy(X_test[start : start + batch_size]).to(device)
            output = model(batch)
            factor_preds.append(output[:, :5].cpu().numpy())
            state_preds.append(output[:, 5 : 5 + num_states].argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(state_preds)
    y_true = states[test_idx]
    y_factor_pred = np.vstack(factor_preds)
    y_factor = factors[test_idx]
    metrics = _classification_metrics(y_true, y_pred, num_states)
    factor_error = y_factor_pred - y_factor
    return {
        "test_user": str(fold["test_user"]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "num_samples": int(test_idx.size),
        **metrics,
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "factor_targets": y_factor.tolist(),
        "factor_preds": y_factor_pred.tolist(),
    }


def _evaluate_baselines(
    output_dir: Path,
    X: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    users: np.ndarray,
    *,
    num_states: int,
    state_names: List[str],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    folds = loso_folds(users, seed=args.seed)
    if args.limit_folds is not None:
        folds = folds[: args.limit_folds]

    baseline_rows: Dict[str, List[Dict[str, Any]]] = {
        "majority_class": [],
        "stratified_random": [],
    }
    if not args.skip_mlp_baseline:
        baseline_rows["simple_mlp"] = []

    for fold in folds:
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        y_true = states[test_idx]
        train_counts = np.bincount(states[train_idx], minlength=num_states)
        majority_class = int(np.argmax(train_counts))
        majority_pred = np.full_like(y_true, fill_value=majority_class)
        majority_metrics = _classification_metrics(y_true, majority_pred, num_states)
        baseline_rows["majority_class"].append(
            {
                "test_user": str(fold["test_user"]),
                "majority_class": state_names[majority_class],
                "train_class_counts": train_counts.astype(int).tolist(),
                "num_samples": int(test_idx.size),
                **majority_metrics,
                "y_true": y_true.tolist(),
                "y_pred": majority_pred.tolist(),
            }
        )

        rng = np.random.default_rng(args.seed + 10_000 + int(fold["fold_idx"]))
        probs = train_counts.astype(np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(num_states) / num_states
        stratified_pred = rng.choice(np.arange(num_states), size=y_true.shape[0], p=probs)
        stratified_metrics = _classification_metrics(y_true, stratified_pred, num_states)
        baseline_rows["stratified_random"].append(
            {
                "test_user": str(fold["test_user"]),
                "train_class_probs": probs.tolist(),
                "num_samples": int(test_idx.size),
                **stratified_metrics,
                "y_true": y_true.tolist(),
                "y_pred": stratified_pred.tolist(),
            }
        )

        if not args.skip_mlp_baseline:
            baseline_rows["simple_mlp"].append(
                _evaluate_mlp_fold(
                    fold,
                    X,
                    factors,
                    states,
                    num_states=num_states,
                    device=device,
                    batch_size=args.baseline_batch_size,
                    epochs=args.baseline_epochs,
                    patience=args.baseline_patience,
                    lr=args.baseline_lr,
                    weight_decay=args.baseline_weight_decay,
                    seed=args.seed,
                    class_weight_mode=args.baseline_class_weights,
                    state_loss=args.baseline_state_loss,
                    task=args.baseline_task,
                    focal_gamma=args.baseline_focal_gamma,
                )
            )

    baselines = {
        name: _aggregate_baseline(name, rows, num_states)
        for name, rows in baseline_rows.items()
    }
    global_counts = np.bincount(states, minlength=num_states)
    global_majority_idx = int(np.argmax(global_counts))
    global_majority_pred = np.full_like(states, fill_value=global_majority_idx)
    baselines["global_majority_reference"] = {
        "name": "global_majority_reference",
        "note": "Uses the full label distribution for interpretability only; LOSO-safe baseline is majority_class.",
        "majority_class": state_names[global_majority_idx],
        "global_class_counts": global_counts.astype(int).tolist(),
        "num_samples": int(states.shape[0]),
        **_classification_metrics(states, global_majority_pred, num_states),
    }
    payload = {
        "state_mode": args.resolved_state_mode,
        "state_names": state_names,
        "num_states": num_states,
        "baselines": baselines,
        "mlp_config": None
        if args.skip_mlp_baseline
        else {
            "epochs": args.baseline_epochs,
            "patience": args.baseline_patience,
            "batch_size": args.baseline_batch_size,
            "lr": args.baseline_lr,
            "weight_decay": args.baseline_weight_decay,
            "class_weights": args.baseline_class_weights,
            "state_loss": args.baseline_state_loss,
            "task": args.baseline_task,
            "focal_gamma": args.baseline_focal_gamma,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate switching predictive LOSO checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_for_training"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/switching_predictive"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--state-mode", choices=["auto", "5class", "3class"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-mlp-baseline", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--baseline-patience", type=int, default=12)
    parser.add_argument("--baseline-batch-size", type=int, default=32)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--baseline-weight-decay", type=float, default=1e-4)
    parser.add_argument("--baseline-class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--baseline-state-loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--baseline-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--baseline-task",
        choices=["joint", "regression_only", "classification_only"],
        default="joint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_paths = sorted(args.checkpoint_dir.glob("fold_*/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold_*/best.pt checkpoints found under {args.checkpoint_dir}")

    state_mode = _infer_state_mode(checkpoint_paths, args.state_mode, device)
    args.resolved_state_mode = state_mode
    num_states = num_states_for_mode(state_mode)
    state_names = state_names_for_mode(state_mode)

    X, labels, metadata, resolved_dir = load_training_arrays(args.data_dir)
    factors, states = prepare_targets(labels, state_mode=state_mode)
    users = users_from_metadata(metadata)

    if args.limit_folds is not None:
        checkpoint_paths = checkpoint_paths[: args.limit_folds]

    fold_results = [
        _predict(
            path,
            X,
            factors,
            states,
            users,
            device=device,
            batch_size=args.batch_size,
            fallback_state_mode=state_mode,
        )
        for path in checkpoint_paths
    ]

    all_true = np.asarray([label for row in fold_results for label in row["y_true"]], dtype=np.int64)
    all_pred = np.asarray([label for row in fold_results for label in row["y_pred"]], dtype=np.int64)
    all_factor_targets = np.asarray([row for fold in fold_results for row in fold["factor_targets"]], dtype=np.float32)
    all_factor_preds = np.asarray([row for fold in fold_results for row in fold["factor_preds"]], dtype=np.float32)
    factor_error = all_factor_preds - all_factor_targets
    class_metrics = _classification_metrics(all_true, all_pred, num_states)
    cm = np.asarray(class_metrics["confusion_matrix"], dtype=np.int64)

    summary = {
        "data_dir": str(resolved_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "state_mode": state_mode,
        "state_names": state_names,
        "num_states": num_states,
        "num_folds": len(fold_results),
        "num_samples": int(all_true.size),
        "factor_mae": float(np.mean(np.abs(factor_error))),
        "factor_rmse": float(np.sqrt(np.mean(factor_error**2))),
        **class_metrics,
        "folds": fold_results,
    }

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "loso_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(args.checkpoint_dir / "loso_summary.csv", fold_results)
    _plot_confusion_matrix(args.checkpoint_dir / "confusion_matrix.png", cm, state_names)
    _write_user_distribution_reports(args.checkpoint_dir, metadata, states, fold_results, state_names)
    if not args.skip_baselines:
        baselines = _evaluate_baselines(
            args.checkpoint_dir,
            X,
            factors,
            states,
            users,
            num_states=num_states,
            state_names=state_names,
            device=device,
            args=args,
        )
        summary["baselines_path"] = str(args.checkpoint_dir / "baselines.json")
        summary["baseline_names"] = sorted(baselines["baselines"].keys())

    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
