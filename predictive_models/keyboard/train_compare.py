"""predictive_models/keyboard/train_compare.py

Keyboard predictive model training + evaluation on Tucker slices.

This follows the teammate training guide:
- input: tucker_slices.npy (N, 4, 512) -> keyboard modality index 1
- labels: nasa_tlx_labels.npy (N, 9)
- splits: LOSO by user_id from metadata.json
- targets: factors (MD, TD, EF, FR, AR proxy), and 5-way cognitive state
- loss: 0.4 * huber(factors) + 0.6 * CE(logits)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


MODELS = {
    "gru": "v1_gru.KeyboardGRU",
    "tcn": "v2_tcn.KeyboardTCN",
    "transformer": "v3_transformer.KeyboardTransformer",
    "hybrid": "v4_hybrid.KeyboardHybrid",
}


KEYBOARD_MODALITY_IDX = 1


def derive_state_label(nasa_tlx_row: np.ndarray) -> int:
    """Derive cognitive state class from NASA-TLX scores (guide function)."""

    md = float(nasa_tlx_row[0])
    td = float(nasa_tlx_row[2])
    ef = float(nasa_tlx_row[4])
    fr = float(nasa_tlx_row[5])
    perf = float(nasa_tlx_row[3])

    high_demand = (md + ef) / 2 > 60
    high_frust = fr > 60
    high_td = td > 65
    low_demand = (md + ef) / 2 < 35
    good_perf = perf < 35

    if high_demand and high_frust:
        return 4  # Overloaded
    if high_td and not high_frust:
        return 3  # Distracted
    if not high_demand and good_perf and not high_frust:
        return 0  # Flow
    if low_demand and not good_perf:
        return 2  # Bored
    return 1  # Neutral


def prepare_targets(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training targets from nasa_tlx_labels.npy (guide function)."""

    md = labels[:, 0:1]
    td = labels[:, 2:3]
    ef = labels[:, 4:5]
    fr = labels[:, 5:6]
    ar = (labels[:, 2:3] + labels[:, 4:5]) / 2

    factors = np.concatenate([md, td, ef, fr, ar], axis=1).astype(np.float32)
    states = np.array([derive_state_label(row) for row in labels], dtype=np.int64)
    return factors, states


def compute_loss(
    output: torch.Tensor,
    factors_gt: torch.Tensor,
    states_gt: torch.Tensor,
) -> torch.Tensor:
    """Exact loss from the teammate guide (expects raw logits in dims 5–9)."""

    factor_loss = F.huber_loss(output[:, :5], factors_gt)
    state_loss = F.cross_entropy(output[:, 5:10], states_gt)
    return 0.4 * factor_loss + 0.6 * state_loss


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    labels = list(range(5))
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # Some folds can be single-class; kappa may become NaN.
    if not np.isfinite(kappa):
        kappa = 0.0
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_factors(factors_true_01: np.ndarray, factors_pred_01: np.ndarray) -> Dict:
    """Regression metrics for the 5 factor outputs.

    Targets are in 0–1 (because we normalize by /100 in training). We report
    errors in original 0–100 scale for interpretability.
    """

    y_true = factors_true_01.astype(np.float64)
    y_pred = factors_pred_01.astype(np.float64)

    # R² averaged across the 5 outputs
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

    mae_01 = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse_01 = float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))

    return {
        "factor_r2": float(r2),
        "factor_mae_01": float(mae_01),
        "factor_rmse_01": float(rmse_01),
        "factor_mae_0_100": float(mae_01 * 100.0),
        "factor_rmse_0_100": float(rmse_01 * 100.0),
    }


def _resolve_model(model_key: str):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {sorted(MODELS.keys())}")

    rel_path = MODELS[model_key]
    module_name, cls_name = rel_path.split(".")
    module = import_module(f"predictive_models.keyboard.{module_name}")
    return getattr(module, cls_name)


@dataclass(frozen=True)
class FoldData:
    x_train: np.ndarray
    x_test: np.ndarray
    factors_train: np.ndarray
    factors_test: np.ndarray
    states_train: np.ndarray
    states_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class EpochLog:
    model: str
    fold: str
    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    val_mcc: float
    val_kappa: float


def _normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-9
    return (x_train - mean) / std, (x_test - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    tucker_path = os.path.join(data_dir, "tucker_slices.npy")
    labels_path = os.path.join(data_dir, "nasa_tlx_labels.npy")
    meta_path = os.path.join(data_dir, "metadata.json")

    if not os.path.exists(tucker_path):
        raise FileNotFoundError(tucker_path)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(labels_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    tucker = np.load(tucker_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if tucker.ndim != 3 or tucker.shape[1] != 4:
        raise ValueError(f"Expected tucker (N, 4, 512), got {tucker.shape}")
    if labels.ndim != 2 or labels.shape[1] != 9:
        raise ValueError(f"Expected labels (N, 9), got {labels.shape}")
    if len(meta) != tucker.shape[0] or len(meta) != labels.shape[0]:
        raise ValueError(
            f"metadata length {len(meta)} must match N={tucker.shape[0]} (tucker) and N={labels.shape[0]} (labels)"
        )

    return tucker, labels, meta


def make_loso_folds(
    tucker: np.ndarray,
    labels: np.ndarray,
    meta: List[dict],
    modality_idx: int,
) -> Tuple[List[str], Dict[str, FoldData]]:
    users = sorted(set(m["user_id"] for m in meta))

    factors, states = prepare_targets(labels)
    factors = factors / 100.0  # normalize targets to [0,1]

    folds: Dict[str, FoldData] = {}
    for test_user in users:
        train_idx = [i for i, m in enumerate(meta) if m["user_id"] != test_user]
        test_idx = [i for i, m in enumerate(meta) if m["user_id"] == test_user]

        x_train = tucker[train_idx, modality_idx, :]
        x_test = tucker[test_idx, modality_idx, :]

        x_train, x_test, mean, std = _normalize_train_test(x_train, x_test)

        folds[test_user] = FoldData(
            x_train=x_train,
            x_test=x_test,
            factors_train=factors[train_idx],
            factors_test=factors[test_idx],
            states_train=states[train_idx],
            states_test=states[test_idx],
            mean=mean,
            std=std,
        )

    return users, folds


def _to_loader(
    x: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(factors),
        torch.from_numpy(states),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def eval_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    f_true: List[np.ndarray] = []
    f_pred: List[np.ndarray] = []

    with torch.no_grad():
        for xb, fb, sb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            sb = sb.to(device)
            if hasattr(model, "reset_microstate"):
                model.reset_microstate()

            out = model(xb)
            loss = compute_loss(out, fb, sb)

            total_loss += float(loss.detach().cpu())
            n_batches += 1

            logits = out[:, 5:10]
            pred = torch.argmax(logits, dim=-1)
            y_true.extend(sb.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())

            f_true.append(fb.detach().cpu().numpy())
            f_pred.append(out[:, :5].detach().cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    return (
        avg_loss,
        np.concatenate(f_true, axis=0).astype(np.float32) if f_true else np.zeros((0, 5), dtype=np.float32),
        np.concatenate(f_pred, axis=0).astype(np.float32) if f_pred else np.zeros((0, 5), dtype=np.float32),
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
    )


def train_one_fold(
    model: torch.nn.Module,
    fold: FoldData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, List[Tuple[float, float, Dict]]]:
    model.to(device)

    train_loader = _to_loader(fold.x_train, fold.factors_train, fold.states_train, batch_size=batch_size, shuffle=True)
    test_loader = _to_loader(fold.x_test, fold.factors_test, fold.states_test, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_history: List[Tuple[float, float, Dict]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, fb, sb in train_loader:
            xb = xb.to(device)
            fb = fb.to(device)
            sb = sb.to(device)

            # Important for sequence-based models: avoid leaking rolling history across random batches.
            if hasattr(model, "reset_microstate"):
                model.reset_microstate()

            out = model(xb)
            loss = compute_loss(out, fb, sb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_loss, f_true_np, f_pred_np, y_true_np, y_pred_np = eval_loader(model, test_loader, device)
        metrics = evaluate(y_true_np, y_pred_np)
        metrics.update(evaluate_factors(f_true_np, f_pred_np))
        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_macro_f1={metrics['macro_f1']:.4f} | val_acc={metrics['accuracy']:.4f} | "
            f"val_r2={metrics['factor_r2']:.3f}"
        )
        epoch_history.append((avg_loss, val_loss, metrics))

    # Final eval predictions are from last epoch
    _val_loss, f_true_np, f_pred_np, y_true_np, y_pred_np = eval_loader(model, test_loader, device)
    final_metrics = evaluate(y_true_np, y_pred_np)
    final_metrics.update(evaluate_factors(f_true_np, f_pred_np))
    return y_pred_np, f_true_np, f_pred_np, final_metrics, epoch_history


def train_full(
    model: torch.nn.Module,
    x: np.ndarray,
    factors: np.ndarray,
    states: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> List[Tuple[float, float, Dict]]:
    """Train a single global model on all windows (no LOSO)."""

    model.to(device)
    loader = _to_loader(x, factors, states, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_history: List[Tuple[float, float, Dict]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, fb, sb in loader:
            xb = xb.to(device)
            fb = fb.to(device)
            sb = sb.to(device)
            if hasattr(model, "reset_microstate"):
                model.reset_microstate()

            out = model(xb)
            loss = compute_loss(out, fb, sb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        # For full training we report train metrics on the same data (useful sanity check, not generalisation)
        val_loss, f_true_np, f_pred_np, y_true_np, y_pred_np = eval_loader(
            model,
            _to_loader(x, factors, states, batch_size=batch_size, shuffle=False),
            device,
        )
        metrics = evaluate(y_true_np, y_pred_np)
        metrics.update(evaluate_factors(f_true_np, f_pred_np))
        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | full_loss={val_loss:.4f} | "
            f"full_macro_f1={metrics['macro_f1']:.4f} | full_acc={metrics['accuracy']:.4f} | "
            f"full_r2={metrics['factor_r2']:.3f}"
        )
        epoch_history.append((train_loss, val_loss, metrics))

    return epoch_history


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder containing tucker_slices.npy, nasa_tlx_labels.npy, metadata.json",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="hybrid",
        help=f"Model key to train: one of {sorted(MODELS.keys())} or 'all'",
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of model keys (overrides --model). Example: --models gru tcn transformer hybrid",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument(
        "--max_folds",
        type=int,
        default=0,
        help="If >0, only run the first N LOSO folds (useful for quick smoke checks)",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="If set, saves a checkpoint per fold (state_dict + mean/std) into this directory",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="If set, writes an overall comparison CSV to this path",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="",
        help="If set, writes an overall comparison JSON to this path",
    )
    ap.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="If set, saves a bar chart PNG of overall metrics (requires matplotlib)",
    )
    ap.add_argument(
        "--curve_plot_path",
        type=str,
        default="",
        help="If set, saves learning-curve PNG(s) (loss + macro-F1 vs epoch) using matplotlib",
    )
    ap.add_argument(
        "--epoch_csv",
        type=str,
        default="",
        help="If set, writes per-epoch logs (train/val loss + metrics) to this CSV path",
    )
    ap.add_argument(
        "--train_full",
        action="store_true",
        help="Train one global model on ALL users/windows (no LOSO).",
    )
    ap.add_argument(
        "--full_ckpt_path",
        type=str,
        default="",
        help="If --train_full is set, saves one global checkpoint to this path",
    )
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tucker, labels, meta = load_dataset(args.data_dir)
    users, folds = make_loso_folds(tucker, labels, meta, modality_idx=KEYBOARD_MODALITY_IDX)

    # Full-data targets (for --train_full)
    full_factors, full_states = prepare_targets(labels)
    full_factors = (full_factors / 100.0).astype(np.float32)
    full_x = tucker[:, KEYBOARD_MODALITY_IDX, :].astype(np.float32)
    full_x, _unused, full_mean, full_std = _normalize_train_test(full_x, full_x)

    if args.max_folds and args.max_folds > 0:
        users = users[: args.max_folds]

    if args.models is not None:
        model_keys = args.models
    elif args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    model_keys = [m.strip().lower() for m in model_keys if m.strip()]
    for mk in model_keys:
        if mk not in MODELS:
            raise ValueError(f"Unknown model '{mk}'. Choices: {sorted(MODELS.keys())} or 'all'")

    summary_rows: List[Dict[str, float]] = []
    epoch_logs: List[EpochLog] = []

    if args.train_full:
        if args.full_ckpt_path and len(model_keys) != 1:
            raise ValueError("--full_ckpt_path only supports a single model; use --save_dir to save multiple.")

        for model_key in model_keys:
            ModelCls = _resolve_model(model_key)
            print(f"Global training (all users) | model={model_key} | device={device}")
            model = ModelCls(input_flat_dim=int(full_x.shape[1]))
            if hasattr(model, "reset_microstate"):
                model.reset_microstate()
            history = train_full(
                model=model,
                x=full_x,
                factors=full_factors,
                states=full_states,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            for epoch_i, (tr_loss, va_loss, mets) in enumerate(history, start=1):
                epoch_logs.append(
                    EpochLog(
                        model=model_key,
                        fold="FULL",
                        epoch=epoch_i,
                        train_loss=float(tr_loss),
                        val_loss=float(va_loss),
                        val_macro_f1=float(mets["macro_f1"]),
                        val_mcc=float(mets["mcc"]),
                        val_kappa=float(mets["kappa"]),
                    )
                )

            final_metrics = history[-1][2] if history else {}
            summary_rows.append(
                {
                    "model": model_key,
                    "macro_f1": float(final_metrics.get("macro_f1", 0.0)),
                    "accuracy": float(final_metrics.get("accuracy", 0.0)),
                    "mcc": float(final_metrics.get("mcc", 0.0)),
                    "kappa": float(final_metrics.get("kappa", 0.0)),
                    "factor_r2": float(final_metrics.get("factor_r2", 0.0)),
                    "factor_mae_0_100": float(final_metrics.get("factor_mae_0_100", 0.0)),
                    "factor_rmse_0_100": float(final_metrics.get("factor_rmse_0_100", 0.0)),
                }
            )

            # Save ckpt
            if args.full_ckpt_path:
                ckpt_path = args.full_ckpt_path
            elif args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"keyboard_{model_key}_global.pt")
            else:
                ckpt_path = ""

            if ckpt_path:
                os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)) or ".", exist_ok=True)
                torch.save(
                    {
                        "mode": "train_full",
                        "model_key": model_key,
                        "model_class": f"{ModelCls.__module__}.{ModelCls.__name__}",
                        "input_flat_dim": int(full_x.shape[1]),
                        "state_dict": model.state_dict(),
                        "norm_mean": full_mean,
                        "norm_std": full_std,
                        "metrics": final_metrics,
                    },
                    ckpt_path,
                )
                print(f"Saved global checkpoint: {ckpt_path}")

        # Skip LOSO loop entirely.
        model_keys = []

    for model_key in model_keys:
        ModelCls = _resolve_model(model_key)

        all_true: List[int] = []
        all_pred: List[int] = []
        all_f_true: List[np.ndarray] = []
        all_f_pred: List[np.ndarray] = []
        print(f"\n==============================")
        print(f"Keyboard LOSO training | model={model_key} | device={device} | folds={len(users)}")
        print(f"==============================")

        epoch_agg: Dict[int, List[float]] = {}
        epoch_agg_val: Dict[int, List[float]] = {}
        epoch_agg_f1: Dict[int, List[float]] = {}

        for fold_i, test_user in enumerate(users, start=1):
            fold = folds[test_user]

            model = ModelCls(input_flat_dim=fold.x_train.shape[1])
            if hasattr(model, "reset_microstate"):
                model.reset_microstate()

            print(
                f"\nFold {fold_i}/{len(users)} | test_user={test_user} | "
                f"train={len(fold.x_train)} | test={len(fold.x_test)}"
            )
            _y_pred, f_true_np, f_pred_np, metrics, history = train_one_fold(
                model=model,
                fold=fold,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            for epoch_i, (tr_loss, va_loss, mets) in enumerate(history, start=1):
                epoch_logs.append(
                    EpochLog(
                        model=model_key,
                        fold=test_user,
                        epoch=epoch_i,
                        train_loss=float(tr_loss),
                        val_loss=float(va_loss),
                        val_macro_f1=float(mets["macro_f1"]),
                        val_mcc=float(mets["mcc"]),
                        val_kappa=float(mets["kappa"]),
                    )
                )
                epoch_agg.setdefault(epoch_i, []).append(float(tr_loss))
                epoch_agg_val.setdefault(epoch_i, []).append(float(va_loss))
                epoch_agg_f1.setdefault(epoch_i, []).append(float(mets["macro_f1"]))

            print(
                "  metrics | "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"mcc={metrics['mcc']:.4f} "
                f"kappa={metrics['kappa']:.4f}"
            )
            print(f"  confusion_matrix:\n{metrics['confusion_matrix']}")

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"keyboard_{model_key}_loso_{test_user}.pt")
                torch.save(
                    {
                        "model_key": model_key,
                        "model_class": f"{ModelCls.__module__}.{ModelCls.__name__}",
                        "input_flat_dim": int(fold.x_train.shape[1]),
                        "state_dict": model.state_dict(),
                        "norm_mean": fold.mean,
                        "norm_std": fold.std,
                        "test_user": test_user,
                        "metrics": {
                            "macro_f1": float(metrics["macro_f1"]),
                            "accuracy": float(metrics["accuracy"]),
                            "mcc": float(metrics["mcc"]),
                            "kappa": float(metrics["kappa"]),
                            "factor_r2": float(metrics.get("factor_r2", 0.0)),
                            "factor_mae_0_100": float(metrics.get("factor_mae_0_100", 0.0)),
                            "factor_rmse_0_100": float(metrics.get("factor_rmse_0_100", 0.0)),
                            "confusion_matrix": metrics["confusion_matrix"].tolist(),
                        },
                    },
                    ckpt_path,
                )
                print(f"  saved_checkpoint={ckpt_path}")

            all_true.extend(fold.states_test.tolist())
            all_pred.extend(_y_pred.tolist())
            all_f_true.append(f_true_np)
            all_f_pred.append(f_pred_np)

        overall = evaluate(np.array(all_true, dtype=np.int64), np.array(all_pred, dtype=np.int64))
        if all_f_true and all_f_pred:
            overall.update(
                evaluate_factors(
                    np.concatenate(all_f_true, axis=0),
                    np.concatenate(all_f_pred, axis=0),
                )
            )
        print("\n=== Overall (all folds concatenated) ===")
        print(
            f"macro_f1={overall['macro_f1']:.4f} | "
            f"acc={overall['accuracy']:.4f} | "
            f"mcc={overall['mcc']:.4f} | "
            f"kappa={overall['kappa']:.4f}"
        )
        if "factor_r2" in overall:
            print(
                f"factors | r2={overall['factor_r2']:.4f} "
                f"mae(0-100)={overall['factor_mae_0_100']:.2f} "
                f"rmse(0-100)={overall['factor_rmse_0_100']:.2f}"
            )
        print(f"confusion_matrix:\n{overall['confusion_matrix']}")

        summary_rows.append(
            {
                "model": model_key,
                "macro_f1": float(overall["macro_f1"]),
                "accuracy": float(overall["accuracy"]),
                "mcc": float(overall["mcc"]),
                "kappa": float(overall["kappa"]),
                "factor_r2": float(overall.get("factor_r2", 0.0)),
                "factor_mae_0_100": float(overall.get("factor_mae_0_100", 0.0)),
                "factor_rmse_0_100": float(overall.get("factor_rmse_0_100", 0.0)),
            }
        )

        if epoch_agg:
            print("\n--- Per-epoch mean summary across folds ---")
            print("epoch\ttrain_loss_mean\tval_loss_mean\tval_macro_f1_mean")
            for epoch_i in sorted(epoch_agg.keys()):
                tr_mean = float(np.mean(epoch_agg[epoch_i]))
                va_mean = float(np.mean(epoch_agg_val.get(epoch_i, [0.0])))
                f1_mean = float(np.mean(epoch_agg_f1.get(epoch_i, [0.0])))
                print(f"{epoch_i}\t{tr_mean:.4f}\t{va_mean:.4f}\t{f1_mean:.4f}")

    if len(summary_rows) > 1:
        print("\n\n=== Summary (overall) ===")
        print("model\tmacro_f1\tacc\tmcc\tkappa\tfactor_r2\tmae(0-100)\trmse(0-100)")
        for row in summary_rows:
            print(
                f"{row.get('model','?')}\t"
                f"{row.get('macro_f1',0.0):.4f}\t"
                f"{row.get('accuracy',0.0):.4f}\t"
                f"{row.get('mcc',0.0):.4f}\t"
                f"{row.get('kappa',0.0):.4f}\t"
                f"{row.get('factor_r2',0.0):.4f}\t"
                f"{row.get('factor_mae_0_100',0.0):.2f}\t"
                f"{row.get('factor_rmse_0_100',0.0):.2f}"
            )

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "macro_f1",
                    "accuracy",
                    "mcc",
                    "kappa",
                    "factor_r2",
                    "factor_mae_0_100",
                    "factor_rmse_0_100",
                ],
            )
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"\nWrote CSV: {args.out_csv}")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"Wrote JSON: {args.out_json}")

    if args.plot_path:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(os.path.abspath(args.plot_path)) or ".", exist_ok=True)
            labels = [r["model"] for r in summary_rows]
            x = np.arange(len(labels))
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(x - width, [r["macro_f1"] for r in summary_rows], width, label="macro_f1")
            ax.bar(x, [r["mcc"] for r in summary_rows], width, label="mcc")
            ax.bar(x + width, [r["kappa"] for r in summary_rows], width, label="kappa")

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title("Keyboard model comparison (overall)")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(args.plot_path, dpi=200)
            plt.close(fig)
            print(f"Wrote plot: {args.plot_path}")
        except ImportError:
            print("matplotlib is not installed; skipping plot. Install via: pip install matplotlib")

    if args.curve_plot_path and epoch_logs:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Aggregate by model + epoch (mean across folds)
            per_model: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
            for r in epoch_logs:
                per_model.setdefault(r.model, {})
                per_model[r.model].setdefault(
                    r.epoch,
                    {"train_loss": [], "val_loss": [], "val_macro_f1": []},
                )
                per_model[r.model][r.epoch]["train_loss"].append(float(r.train_loss))
                per_model[r.model][r.epoch]["val_loss"].append(float(r.val_loss))
                per_model[r.model][r.epoch]["val_macro_f1"].append(float(r.val_macro_f1))

            root, ext = os.path.splitext(args.curve_plot_path)
            ext = ext or ".png"

            for model_key, epochs_dict in per_model.items():
                epochs_sorted = sorted(epochs_dict.keys())
                train_loss = [float(np.mean(epochs_dict[e]["train_loss"])) for e in epochs_sorted]
                val_loss = [float(np.mean(epochs_dict[e]["val_loss"])) for e in epochs_sorted]
                macro_f1 = [float(np.mean(epochs_dict[e]["val_macro_f1"])) for e in epochs_sorted]

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
                ax1.plot(epochs_sorted, train_loss, label="train_loss")
                ax1.plot(epochs_sorted, val_loss, label="val_loss")
                ax1.set_ylabel("loss")
                ax1.legend(loc="best")

                ax2.plot(epochs_sorted, macro_f1, label="macro_f1")
                ax2.set_ylim(0.0, 1.0)
                ax2.set_ylabel("macro_f1")
                ax2.set_xlabel("epoch")
                ax2.legend(loc="best")

                fig.suptitle(f"Learning curves: {model_key}")
                fig.tight_layout()

                out_path = f"{root}_{model_key}{ext}" if len(per_model) > 1 else f"{root}{ext}"
                os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"Wrote curves plot: {out_path}")

        except ImportError:
            print("matplotlib is not installed; skipping learning-curve plot. Install via: pip install matplotlib")

    if args.epoch_csv and epoch_logs:
        os.makedirs(os.path.dirname(os.path.abspath(args.epoch_csv)) or ".", exist_ok=True)
        with open(args.epoch_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "fold",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_macro_f1",
                    "val_mcc",
                    "val_kappa",
                ],
            )
            w.writeheader()
            for r in epoch_logs:
                w.writerow(r.__dict__)
        print(f"Wrote epoch CSV: {args.epoch_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())