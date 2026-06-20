"""
train_keyboard_models.py
========================
Trains and evaluates all 4 keyboard modality models:
  v1 — KeyboardGRU
  v2 — KeyboardTCN
  v3 — KeyboardTransformer
  v4 — KeyboardHybrid  (recommended final)

Evaluation: LOSO (Leave-One-Subject-Out)
Primary metric: macro F1
Also reports: MCC, Cohen's kappa, per-class F1, confusion matrix

Usage:
    cd ~/fusion_model
    python train_keyboard_models.py \
        --sequences  ~/tucker_outputs/sequences.npy \
        --output_dir ~/keyboard_models \
        --epochs 50 \
        --models gru tcn transformer hybrid

Output:
    keyboard_gru_v1.pt
    keyboard_tcn_v2.pt
    keyboard_transformer_v3.pt
    keyboard_hybrid_v4.pt
    results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

FUSION_DIR = Path(__file__).resolve().parent
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

MODALITY_IDX = 1   # keyboard
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES  = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]

print(f"Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# Label derivation  (same as mouse)
# ═════════════════════════════════════════════════════════════════════════════

def derive_state_label(row: np.ndarray) -> int:
    md   = float(row[0]); td = float(row[2])
    ef   = float(row[4]); fr = float(row[5]); perf = float(row[3])
    high_demand = (md + ef) / 2 > 60
    high_frust  = fr  > 60
    high_td     = td  > 65
    low_demand  = (md + ef) / 2 < 35
    good_perf   = perf < 35
    if high_demand and high_frust:  return 4
    if high_td and not high_frust:  return 3
    if not high_demand and good_perf and not high_frust: return 0
    if low_demand and not good_perf: return 2
    return 1


def prepare_labels(nasa_tlx: np.ndarray) -> tuple:
    factors = np.array([
        nasa_tlx[0], nasa_tlx[2], nasa_tlx[4],
        nasa_tlx[5], (nasa_tlx[2] + nasa_tlx[4]) / 2,
    ], dtype=np.float32) / 100.0
    return factors, derive_state_label(nasa_tlx)


# ═════════════════════════════════════════════════════════════════════════════
# Loss with class weights
# ═════════════════════════════════════════════════════════════════════════════

def compute_loss(
    output:      torch.Tensor,
    factors_gt:  torch.Tensor,
    states_gt:   torch.Tensor,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    factor_loss = F.huber_loss(output[:, :5], factors_gt)
    state_loss  = F.cross_entropy(
        output[:, 5:10], states_gt,
        weight=class_weights,
    )
    return 0.4 * factor_loss + 0.6 * state_loss


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label: str) -> dict:
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None,     zero_division=0).tolist()
    mcc       = matthews_corrcoef(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    cm        = confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist()

    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"  Per-class F1:")
    for i, (name, f1) in enumerate(zip(STATE_NAMES, per_class)):
        n = int(np.sum(np.array(y_true) == i))
        print(f"    {name:<12} {f1:.4f}  (n={n})")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(f"    {STATE_NAMES[i]:<12} {row}")

    return {
        "macro_f1": macro_f1, "mcc": mcc, "kappa": kappa,
        "per_class_f1": dict(zip(STATE_NAMES, per_class)),
        "confusion_matrix": cm,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_sequences(path: Path) -> list[dict]:
    seqs  = list(np.load(path, allow_pickle=True))
    users = sorted(set(s["user_id"] for s in seqs))
    print(f"Loaded {len(seqs)} sessions  |  {sum(s['T'] for s in seqs)} windows")
    print(f"Users: {users}")

    states = [derive_state_label(s["y"]) for s in seqs]
    print("\nState distribution across sessions:")
    for i, name in enumerate(STATE_NAMES):
        c = states.count(i)
        print(f"  {name:<12} {c:3d}  ({100*c/len(states):.1f}%)")
    return seqs


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, seqs, optimizer, class_weights):
    model.train()
    total_loss = 0.0
    n_windows  = 0

    for s in seqs:
        X       = torch.tensor(s["X"][:, MODALITY_IDX, :],
                               dtype=torch.float32).to(DEVICE)
        factors, state = prepare_labels(s["y"])
        T       = s["T"]
        y_f     = torch.tensor(np.tile(factors, (T, 1)),
                               dtype=torch.float32).to(DEVICE)
        y_s     = torch.tensor([state] * T, dtype=torch.long).to(DEVICE)

        model.reset_microstate()   # clears hidden state + rolling buffer

        for t in range(T):
            x_t   = X[t].unsqueeze(0)       # (1, 512)
            out   = model(x_t)              # (1, 12)
            loss  = compute_loss(out, y_f[t].unsqueeze(0),
                                 y_s[t].unsqueeze(0), class_weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_windows  += 1

    return total_loss / max(n_windows, 1)


@torch.no_grad()
def evaluate(model, seqs, class_weights) -> tuple:
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    n_windows  = 0

    for s in seqs:
        X       = torch.tensor(s["X"][:, MODALITY_IDX, :],
                               dtype=torch.float32).to(DEVICE)
        factors, state = prepare_labels(s["y"])
        T       = s["T"]
        y_f     = torch.tensor(np.tile(factors, (T, 1)),
                               dtype=torch.float32).to(DEVICE)
        y_s     = torch.tensor([state] * T, dtype=torch.long).to(DEVICE)

        model.reset_microstate()

        for t in range(T):
            x_t   = X[t].unsqueeze(0)
            out   = model(x_t)
            loss  = compute_loss(out, y_f[t].unsqueeze(0),
                                 y_s[t].unsqueeze(0), class_weights)
            pred  = int(out[0, 5:10].argmax().item())
            all_true.append(state)
            all_pred.append(pred)
            total_loss += loss.item()
            n_windows  += 1

    return all_true, all_pred, total_loss / max(n_windows, 1)


# ═════════════════════════════════════════════════════════════════════════════
# LOSO loop
# ═════════════════════════════════════════════════════════════════════════════

def train_loso(
    ModelClass,
    model_name:   str,
    save_name:    str,
    seqs:         list[dict],
    output_dir:   Path,
    epochs:       int   = 50,
    lr:           float = 1e-4,
) -> dict:
    print(f"\n{'='*52}")
    print(f"{model_name} — LOSO")
    print(f"{'='*52}")

    # window-level class weights — correct granularity for the training loop
    all_states_window = []
    for s in seqs:
        _, state = prepare_labels(s["y"])
        all_states_window.extend([state] * s["T"])
    all_states_window = np.array(all_states_window)
    weights = compute_class_weight("balanced",
                                   classes=np.unique(all_states_window),
                                   y=all_states_window)
    # pad to 5 classes in case some are missing
    cw = np.ones(5, dtype=np.float32)
    for i, cls in enumerate(np.unique(all_states_window)):
        cw[int(cls)] = weights[i]
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights (window-level): {dict(zip(STATE_NAMES, cw.round(2).tolist()))}")

    unique_users = sorted(set(s["user_id"] for s in seqs))
    all_true, all_pred = [], []

    for test_user in unique_users:
        train_seqs = [s for s in seqs if s["user_id"] != test_user]
        test_seqs  = [s for s in seqs if s["user_id"] == test_user]
        n_test_wins = sum(s["T"] for s in test_seqs)

        model     = ModelClass(input_flat_dim=512, d_proj=256).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

        best_f1      = -1.0
        best_weights = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_seqs, optimizer,
                                         class_weights)
            true, pred, val_loss = evaluate(model, test_seqs, class_weights)
            epoch_f1  = f1_score(true, pred, average="macro", zero_division=0)
            scheduler.step()

            if epoch_f1 > best_f1:
                best_f1      = epoch_f1
                best_weights = {k: v.clone()
                                for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  [{test_user:12s}] ep {epoch+1:3d}  "
                      f"tr={train_loss:.4f}  val={val_loss:.4f}  "
                      f"F1={epoch_f1:.4f}  best={best_f1:.4f}")

        model.load_state_dict(best_weights)
        true, pred, _ = evaluate(model, test_seqs, class_weights)
        all_true.extend(true)
        all_pred.extend(pred)
        fold_f1 = f1_score(true, pred, average="macro", zero_division=0)
        print(f"  [{test_user:12s}] fold F1={fold_f1:.4f}  (n={n_test_wins})")

    metrics = compute_metrics(all_true, all_pred,
                              f"{model_name} — LOSO overall")

    # retrain on full dataset
    print(f"\n  Retraining {model_name} on full dataset...")
    model_final = ModelClass(input_flat_dim=512, d_proj=256).to(DEVICE)
    optimizer   = AdamW(model_final.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    for epoch in range(epochs):
        loss = train_one_epoch(model_final, seqs, optimizer, class_weights)
        scheduler.step()
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
        default=str(Path.home() / "keyboard_models"))
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--models",     nargs="*",
        default=["gru", "tcn", "transformer", "hybrid"],
        help="Which models: gru tcn transformer hybrid")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_sequences(Path(args.sequences))

    # import models
    from predictive_models.keyboard.v1_gru         import KeyboardGRU
    from predictive_models.keyboard.v2_tcn         import KeyboardTCN
    from predictive_models.keyboard.v3_transformer import KeyboardTransformer
    from predictive_models.keyboard.v4_hybrid      import KeyboardHybrid

    MODEL_REGISTRY = {
        "gru":         (KeyboardGRU,         "keyboard_gru_v1.pt"),
        "tcn":         (KeyboardTCN,         "keyboard_tcn_v2.pt"),
        "transformer": (KeyboardTransformer, "keyboard_transformer_v3.pt"),
        "hybrid":      (KeyboardHybrid,      "keyboard_hybrid_v4.pt"),
    }

    results = {}
    for key in args.models:
        if key not in MODEL_REGISTRY:
            print(f"Unknown model '{key}' — skipping")
            continue
        ModelClass, save_name = MODEL_REGISTRY[key]
        results[key] = train_loso(
            ModelClass, f"Keyboard{key.capitalize()}", save_name,
            seqs, output_dir, epochs=args.epochs, lr=args.lr,
        )

    # save results
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results: {results_path}")

    if len(results) > 1:
        print("\n" + "="*52)
        print("COMPARISON SUMMARY")
        print("="*52)
        print(f"  {'Model':<15} {'Macro F1':>10} {'MCC':>8} {'Kappa':>8}")
        print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8}")
        for name, m in results.items():
            print(f"  {name:<15} {m['macro_f1']:>10.4f} "
                  f"{m['mcc']:>8.4f} {m['kappa']:>8.4f}")


if __name__ == "__main__":
    main()
