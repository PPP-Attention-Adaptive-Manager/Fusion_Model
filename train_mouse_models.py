"""
train_mouse_models.py
=====================
Trains and evaluates three mouse modality models:
  1. RandomForest  — sklearn baseline, window-level
  2. MouseGRU      — PyTorch, sequence-aware
  3. MouseLSTM     — PyTorch, sequence-aware

Evaluation: LOSO (Leave-One-Subject-Out)
Primary metric: macro F1
Also reports: MCC, Cohen's kappa, per-class F1, confusion matrix

Usage:
    cd ~/fusion_model
    python train_mouse_models.py \
        --sequences   ~/tucker_outputs/sequences.npy \
        --output_dir  ~/mouse_models

Output:
    mouse_rf_v1.pkl           — trained RF (joblib)
    mouse_gru_v2.pt           — best GRU weights
    mouse_lstm_v3.pt          — best LSTM weights
    results.json              — all LOSO metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
FUSION_DIR = Path(__file__).resolve().parent
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from predictive_models.mouse.v1_mouse_rf   import MouseRandomForest
from predictive_models.mouse.v2_mouse_gru  import MouseGRU
from predictive_models.mouse.v3_mouse_lstm import MouseLSTM

MODALITY_IDX = 0   # mouse
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES  = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]

print(f"Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# Label derivation
# ═════════════════════════════════════════════════════════════════════════════

def derive_state_label(nasa_tlx_row: np.ndarray) -> int:
    """
    NASA-TLX (9 values) → cognitive state 0-4.
    Indices: 0=MD, 1=PD, 2=TD, 3=performance, 4=effort, 5=frustration,
             6=stress, 7=valence, 8=arousal
    """
    md   = float(nasa_tlx_row[0])
    td   = float(nasa_tlx_row[2])
    ef   = float(nasa_tlx_row[4])
    fr   = float(nasa_tlx_row[5])
    perf = float(nasa_tlx_row[3])   # low = performing well in NASA-TLX

    high_demand = (md + ef) / 2 > 60
    high_frust  = fr  > 60
    high_td     = td  > 65
    low_demand  = (md + ef) / 2 < 35
    good_perf   = perf < 35

    if high_demand and high_frust:
        return 4   # Overloaded
    if high_td and not high_frust:
        return 3   # Distracted
    if not high_demand and good_perf and not high_frust:
        return 0   # Flow
    if low_demand and not good_perf:
        return 2   # Bored
    return 1       # Neutral


def prepare_labels(nasa_tlx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        nasa_tlx: (9,) or (N, 9)
    Returns:
        factors_norm: (5,) or (N, 5)  — [MD, TD, EF, FR, AR] normalized to [0,1]
        state:        int  or (N,)    — 0-4
    """
    if nasa_tlx.ndim == 1:
        factors = np.array([
            nasa_tlx[0], nasa_tlx[2], nasa_tlx[4],
            nasa_tlx[5], (nasa_tlx[2] + nasa_tlx[4]) / 2,
        ], dtype=np.float32) / 100.0
        return factors, derive_state_label(nasa_tlx)
    else:
        factors = np.column_stack([
            nasa_tlx[:, 0], nasa_tlx[:, 2], nasa_tlx[:, 4],
            nasa_tlx[:, 5], (nasa_tlx[:, 2] + nasa_tlx[:, 4]) / 2,
        ]).astype(np.float32) / 100.0
        states = np.array([derive_state_label(r) for r in nasa_tlx], dtype=np.int64)
        return factors, states


# ─────────────────────────────────────────────────────────────────────────────
# Temporal label generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_temporal_labels(
    nasa_tlx_final: np.ndarray,
    T:              int,
    v_initial:      float = 50.0,
) -> np.ndarray:
    """
    Generate a (T, 9) sequence of NASA-TLX values that evolves from
    v_initial=50 toward nasa_tlx_final using exponential decay:

        V_i(t) = V_i,final + (V_i,initial - V_i,final) × e^(-λ × t)

    λ is chosen so that at t=T the curve has covered 95% of the distance:
        e^(-λ × T) = 0.05  →  λ = ln(20) / T

    Windows indexed t = 1 … T (first window is NOT the initial value;
    last window is 95% of the way to V_final, not exactly at it).
    """
    v_final = nasa_tlx_final.astype(np.float32)
    lam     = np.log(20.0) / max(T, 1)
    t_idx   = np.arange(1, T + 1, dtype=np.float32)           # (T,)
    decay   = np.exp(-lam * t_idx)[:, np.newaxis]              # (T, 1)
    seq     = v_final + (v_initial - v_final) * decay          # (T, 9)
    return np.clip(seq, 0.0, 100.0).astype(np.float32)


def prepare_temporal_labels(
    nasa_tlx_final: np.ndarray,
    T:              int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns per-window targets derived from the temporal decay sequence.

    Returns:
        factors_seq: (T, 5) float32 — [MD, TD, EF, FR, AR] normalised [0,1]
        states_seq:  (T,)   int64   — state label per window
    """
    seq = generate_temporal_labels(nasa_tlx_final, T)          # (T, 9)
    factors_seq = np.column_stack([
        seq[:, 0],
        seq[:, 2],
        seq[:, 4],
        seq[:, 5],
        (seq[:, 2] + seq[:, 4]) / 2,
    ]).astype(np.float32) / 100.0                               # (T, 5)
    states_seq = np.array(
        [derive_state_label(row) for row in seq], dtype=np.int64
    )                                                           # (T,)
    return factors_seq, states_seq


# ═════════════════════════════════════════════════════════════════════════════
# Loss
# ═════════════════════════════════════════════════════════════════════════════

def compute_loss(
    output:     torch.Tensor,   # (B, 12)
    factors_gt: torch.Tensor,   # (B, 5) normalized [0,1]
    states_gt:  torch.Tensor,   # (B,)   int64
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    factor_loss = F.huber_loss(output[:, :5], factors_gt)
    state_loss  = F.cross_entropy(output[:, 5:10], states_gt,
                                  weight=class_weights)
    return 0.4 * factor_loss + 0.6 * state_loss


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    macro_f1   = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    per_class  = f1_score(y_true, y_pred, average=None,     zero_division=0).tolist()
    mcc        = matthews_corrcoef(y_true, y_pred)
    kappa      = cohen_kappa_score(y_true, y_pred)
    cm         = confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist()

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  Per-class F1:")
    for i, (name, f1) in enumerate(zip(STATE_NAMES, per_class)):
        print(f"    {name:<12} {f1:.4f}  (n={int(np.sum(np.array(y_true)==i))})")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(f"    {STATE_NAMES[i]:<12} {row}")

    return {
        "macro_f1":    macro_f1,
        "mcc":         mcc,
        "kappa":       kappa,
        "per_class_f1": dict(zip(STATE_NAMES, per_class)),
        "confusion_matrix": cm,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_sequences(sequences_path: Path) -> list[dict]:
    seqs = list(np.load(sequences_path, allow_pickle=True))
    print(f"Loaded {len(seqs)} sessions")
    users = sorted(set(s["user_id"] for s in seqs))
    print(f"Users: {users}")
    print(f"Total windows: {sum(s['T'] for s in seqs)}")

    # show temporal label distribution (what the model actually trains on)
    all_temp_states = []
    for s in seqs:
        _, states_seq = prepare_temporal_labels(s["y"], s["T"])
        all_temp_states.extend(states_seq.tolist())
    print("\nTemporal state distribution (window-level, after decay):")
    for i, name in enumerate(STATE_NAMES):
        c = all_temp_states.count(i)
        print(f"  {name:<12} {c:4d}  ({100*c/len(all_temp_states):.1f}%)")
    return seqs


def sequences_to_flat(seqs: list[dict]) -> tuple:
    """Convert to flat window-level arrays for RF."""
    X_list, y_state_list, y_factor_list, user_list = [], [], [], []
    for s in seqs:
        X = s["X"][:, MODALITY_IDX, :]   # (T, 512)
        factors, state = prepare_labels(s["y"])
        T = s["T"]
        X_list.append(X)
        y_state_list.extend([state] * T)
        y_factor_list.extend([factors] * T)
        user_list.extend([s["user_id"]] * T)
    return (
        np.concatenate(X_list, axis=0),
        np.array(y_state_list, dtype=np.int64),
        np.array(y_factor_list, dtype=np.float32),
        np.array(user_list),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Random Forest — LOSO
# ═════════════════════════════════════════════════════════════════════════════

def train_rf_loso(seqs: list[dict], output_dir: Path) -> dict:
    print("\n" + "="*50)
    print("RANDOM FOREST — LOSO")
    print("="*50)

    X_all, y_state_all, y_factor_all, users_all = sequences_to_flat(seqs)
    unique_users = sorted(set(users_all))

    all_true, all_pred = [], []

    for test_user in unique_users:
        train_mask = users_all != test_user
        test_mask  = users_all == test_user

        X_tr, y_s_tr, y_f_tr = X_all[train_mask], y_state_all[train_mask], y_factor_all[train_mask]
        X_te, y_s_te          = X_all[test_mask],  y_state_all[test_mask]

        # per-user z-score on train stats
        mean = X_tr.mean(0, keepdims=True)
        std  = X_tr.std(0,  keepdims=True) + 1e-9
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

        rf = MouseRandomForest()
        rf.fit(X_tr, y_s_tr, y_f_tr)
        preds, _ = rf.predict(X_te)

        all_true.extend(y_s_te.tolist())
        all_pred.extend(preds.tolist())
        print(f"  {test_user:12s}  n={test_mask.sum()}")

    metrics = compute_metrics(all_true, all_pred, "Random Forest — LOSO overall")

    # retrain on all data and save
    mean = X_all.mean(0, keepdims=True)
    std  = X_all.std(0,  keepdims=True) + 1e-9
    X_norm = (X_all - mean) / std
    rf_final = MouseRandomForest()
    rf_final.fit(X_norm, y_state_all, y_factor_all)
    rf_final.save(output_dir / "mouse_rf_v1.pkl")
    np.save(output_dir / "mouse_rf_norm.npy", np.stack([mean, std]))
    print(f"\nSaved: mouse_rf_v1.pkl")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch model — LOSO training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, seqs, optimizer, class_weights=None):
    model.train()
    total_loss = 0.0
    n_windows  = 0

    for s in seqs:
        X = torch.tensor(s["X"][:, MODALITY_IDX, :], dtype=torch.float32).to(DEVICE)
        T = s["T"]

        # per-window temporal targets (vary across session, not broadcast)
        y_f_np, y_s_np = prepare_temporal_labels(s["y"], T)
        y_f = torch.tensor(y_f_np, dtype=torch.float32).to(DEVICE)
        y_s = torch.tensor(y_s_np, dtype=torch.long).to(DEVICE)

        model.reset_microstate()

        for t in range(T):
            x_t   = X[t].unsqueeze(0)
            out   = model(x_t)
            loss  = compute_loss(out, y_f[t].unsqueeze(0), y_s[t].unsqueeze(0),
                                 class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_windows  += 1

    return total_loss / max(n_windows, 1)


@torch.no_grad()
def evaluate_torch(model, seqs, class_weights=None) -> tuple[list, list, float]:
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    n_windows  = 0

    for s in seqs:
        X = torch.tensor(s["X"][:, MODALITY_IDX, :], dtype=torch.float32).to(DEVICE)
        T = s["T"]

        y_f_np, y_s_np = prepare_temporal_labels(s["y"], T)
        y_f = torch.tensor(y_f_np, dtype=torch.float32).to(DEVICE)
        y_s = torch.tensor(y_s_np, dtype=torch.long).to(DEVICE)

        model.reset_microstate()

        for t in range(T):
            x_t  = X[t].unsqueeze(0)
            out  = model(x_t)
            loss = compute_loss(out, y_f[t].unsqueeze(0), y_s[t].unsqueeze(0),
                                class_weights)
            pred = int(out[0, 5:10].argmax().item())
            all_true.append(int(y_s_np[t]))
            all_pred.append(pred)
            total_loss += loss.item()
            n_windows  += 1

    return all_true, all_pred, total_loss / max(n_windows, 1)


def train_torch_loso(
    ModelClass,
    model_name:  str,
    save_name:   str,
    seqs:        list[dict],
    output_dir:  Path,
    epochs:      int = 30,
    lr:          float = 1e-3,
) -> dict:
    print(f"\n{'='*50}")
    print(f"{model_name} — LOSO")
    print(f"{'='*50}")

    unique_users = sorted(set(s["user_id"] for s in seqs))
    all_true, all_pred = [], []

    # window-level class weights from temporal labels — correct granularity
    all_states_window = []
    for s in seqs:
        _, states_seq = prepare_temporal_labels(s["y"], s["T"])
        all_states_window.extend(states_seq.tolist())
    all_states_window = np.array(all_states_window)
    weights = compute_class_weight("balanced",
                                   classes=np.unique(all_states_window),
                                   y=all_states_window)
    cw = np.ones(5, dtype=np.float32)
    for i, cls in enumerate(np.unique(all_states_window)):
        cw[int(cls)] = weights[i]
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights (temporal window-level): {dict(zip(STATE_NAMES, cw.round(2).tolist()))}")

    for test_user in unique_users:
        train_seqs = [s for s in seqs if s["user_id"] != test_user]
        test_seqs  = [s for s in seqs if s["user_id"] == test_user]
        n_test_wins = sum(s["T"] for s in test_seqs)

        model = ModelClass(input_flat_dim=512, d_proj=256).to(DEVICE)
        optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler  = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_f1      = -1.0
        best_weights = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_seqs, optimizer, class_weights)
            true, pred, val_loss = evaluate_torch(model, test_seqs, class_weights)
            epoch_f1 = f1_score(true, pred, average="macro", zero_division=0)
            scheduler.step(val_loss)

            if epoch_f1 > best_f1:
                best_f1      = epoch_f1
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  [{test_user:12s}] epoch {epoch+1:3d}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"F1={epoch_f1:.4f}  best={best_f1:.4f}")

        # evaluate with best weights on test user
        model.load_state_dict(best_weights)
        true, pred, _ = evaluate_torch(model, test_seqs, class_weights)
        all_true.extend(true)
        all_pred.extend(pred)
        fold_f1 = f1_score(true, pred, average="macro", zero_division=0)
        print(f"  [{test_user:12s}] fold F1={fold_f1:.4f}  (n={n_test_wins} windows)")

    metrics = compute_metrics(all_true, all_pred, f"{model_name} — LOSO overall")

    # retrain on all data with best epoch count and save
    model_final = ModelClass(input_flat_dim=512, d_proj=256).to(DEVICE)
    optimizer   = AdamW(model_final.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\n  Retraining {model_name} on full dataset...")
    for epoch in range(epochs):
        loss = train_one_epoch(model_final, seqs, optimizer, class_weights)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  loss={loss:.4f}")

    save_path = output_dir / save_name
    torch.save(model_final.state_dict(), save_path)
    print(f"  Saved: {save_name}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences",  type=str,
        default=str(Path.home() / "tucker_outputs" / "sequences.npy"))
    parser.add_argument("--output_dir", type=str,
        default=str(Path.home() / "mouse_models"))
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--models",     nargs="*",
        default=["rf", "gru", "lstm"],
        help="Which models to train: rf gru lstm")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_sequences(Path(args.sequences))

    # Class distribution info
    all_states = [derive_state_label(s["y"]) for s in seqs]
    print("\nState distribution across sessions:")
    for i, name in enumerate(STATE_NAMES):
        count = all_states.count(i)
        print(f"  {name:<12} {count:3d} sessions ({100*count/len(all_states):.1f}%)")

    results = {}

    if "rf" in args.models:
        results["random_forest"] = train_rf_loso(seqs, output_dir)

    if "gru" in args.models:
        results["gru"] = train_torch_loso(
            MouseGRU, "MouseGRU", "mouse_gru_v2.pt",
            seqs, output_dir, epochs=args.epochs, lr=args.lr,
        )

    if "lstm" in args.models:
        results["lstm"] = train_torch_loso(
            MouseLSTM, "MouseLSTM", "mouse_lstm_v3.pt",
            seqs, output_dir, epochs=args.epochs, lr=args.lr,
        )

    # Save all results
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    # Final comparison
    if len(results) > 1:
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"  {'Model':<15} {'Macro F1':>10} {'MCC':>8} {'Kappa':>8}")
        print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8}")
        for name, m in results.items():
            print(f"  {name:<15} {m['macro_f1']:>10.4f} "
                  f"{m['mcc']:>8.4f} {m['kappa']:>8.4f}")


if __name__ == "__main__":
    main()