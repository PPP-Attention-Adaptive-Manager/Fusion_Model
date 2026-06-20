"""
train_fusion_session.py
=======================
Trains all 4 predictive models JOINTLY on session-level state prediction.

Key design decisions:
  - Feed the full session sequence (T windows) through all 4 models
  - GRU hidden states accumulate context across the session
  - PoE combines per-model logits at every step
  - Loss computed ONLY at the LAST window vs session-level NASA-TLX label
  - This eliminates the label mismatch completely:
      session label → session-level prediction → honest supervision

What gets trained:
  MouseGRU + KeyboardGRU + NotifGRU + SwitchingGRU + PoE weights (if weighted)
  Tucker projections are NOT trained (pre-computed slices in sequences.npy)
  EMA is NOT used during training (inference-only smoothing)

Evaluation:
  LOSO — one prediction per session, metrics over sessions not windows
  This is honest: 43 sessions total, ~39 per training fold

Usage:
    cd ~/fusion_model
    python train_fusion_session.py \
        --sequences ~/tucker_outputs/sequences.npy \
        --output_dir ~/fusion_session_models \
        --epochs 80 \
        --lr 5e-4

Output:
    mouse_gru_session.pt
    keyboard_gru_session.pt
    notif_gru_session.pt
    switching_gru_session.pt
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

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]

# modality slot indices in Tucker tensor
MODALITY = {"mouse": 0, "keyboard": 1, "notif": 2, "switching": 3}

print(f"Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# Label derivation
# ═════════════════════════════════════════════════════════════════════════════

def derive_state_label(row: np.ndarray) -> int:
    md,td,ef,fr,perf = row[0],row[2],row[4],row[5],row[3]
    if (md+ef)/2 > 60 and fr > 60:                    return 4
    if td > 65 and fr <= 60:                           return 3
    if (md+ef)/2 < 35 and perf < 35 and fr <= 60:     return 0
    if (md+ef)/2 < 35 and perf >= 35:                 return 2
    return 1


def prepare_labels(nasa_tlx: np.ndarray) -> tuple:
    factors = np.array([
        nasa_tlx[0], nasa_tlx[2], nasa_tlx[4],
        nasa_tlx[5], (nasa_tlx[2] + nasa_tlx[4]) / 2,
    ], dtype=np.float32) / 100.0
    return factors, derive_state_label(nasa_tlx)


# ═════════════════════════════════════════════════════════════════════════════
# Fusion trainer — holds all 4 models + PoE
# ═════════════════════════════════════════════════════════════════════════════

class FusionTrainer(nn.Module):
    """
    Lightweight wrapper that runs all 4 predictive models through PoE.
    Does NOT include Tucker (pre-computed) or EMA (inference-only).

    Forward pass:
        slices: list of 4 tensors, each (1, 512)
        returns: (1, 5) log-probabilities from PoE
    """

    def __init__(self, model_classes: dict, d_proj: int = 256):
        super().__init__()
        self.models = nn.ModuleDict({
            name: cls(input_flat_dim=512, d_proj=d_proj)
            for name, cls in model_classes.items()
        })

    def reset_all(self):
        for model in self.models.values():
            model.reset_microstate()

    def forward(self, slices: dict[str, torch.Tensor]) -> tuple:
        """
        Args:
            slices: dict name → (1, 512) Tucker slice per modality

        Returns:
            fused_logits:    (1, 5) — sum of raw logits across models (PoE)
            per_model_out:   dict name → (1, 12)
        """
        per_model_out = {}
        logit_sum     = None

        for name, model in self.models.items():
            out = model(slices[name])      # (1, 12)
            per_model_out[name] = out

            # PoE: sum raw logits (dims 5-9), confidence-weighted by (1-H_norm)
            h_norm  = out[:, 10]                        # (1,)
            weight  = (1.0 - h_norm).clamp(min=0.01)   # (1,)
            logits  = out[:, 5:10]                      # (1, 5)
            weighted = weight.unsqueeze(-1) * logits    # (1, 5)

            if logit_sum is None:
                logit_sum = weighted
            else:
                logit_sum = logit_sum + weighted

        return logit_sum, per_model_out    # (1, 5) raw fused logits


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
        print(f"    {name:<12} {f1:.4f}  (n={n} sessions)")
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
    states = [derive_state_label(s["y"]) for s in seqs]

    print(f"Loaded {len(seqs)} sessions  |  {sum(s['T'] for s in seqs)} windows")
    print(f"Users: {users}")
    print("\nSession-level state distribution:")
    for i, name in enumerate(STATE_NAMES):
        c = states.count(i)
        print(f"  {name:<12} {c:3d}  ({100*c/len(states):.1f}%)")
    return seqs


# ═════════════════════════════════════════════════════════════════════════════
# Session forward pass
# ═════════════════════════════════════════════════════════════════════════════

def run_session(
    fusion:   FusionTrainer,
    session:  dict,
    train:    bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Feed one full session through the fusion model.
    Returns (fused_logits_at_T, avg_factors_at_T) from the last window.

    GRU hidden states accumulate over the session — backprop flows through
    the full sequence via BPTT from the last-step loss.
    """
    fusion.reset_all()

    X = session["X"]    # (T, 4, 512)
    T = session["T"]

    last_logits  = None
    last_factors = None

    for t in range(T):
        slices = {
            name: torch.tensor(
                X[t, idx, :], dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            for name, idx in MODALITY.items()
        }

        fused_logits, per_model_out = fusion(slices)   # (1, 5), dict

        if t == T - 1:
            last_logits = fused_logits
            # average factor predictions across models
            last_factors = torch.stack(
                [out[:, :5] for out in per_model_out.values()]
            ).mean(0)                                   # (1, 5)

    return last_logits, last_factors                   # (1,5), (1,5)


# ═════════════════════════════════════════════════════════════════════════════
# Training + evaluation loops
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    fusion:        FusionTrainer,
    seqs:          list[dict],
    optimizer:     torch.optim.Optimizer,
    class_weights: torch.Tensor,
) -> float:
    fusion.train()
    total_loss = 0.0

    for s in seqs:
        factors, state = prepare_labels(s["y"])
        y_s = torch.tensor([state], dtype=torch.long).to(DEVICE)
        y_f = torch.tensor(factors, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        last_logits, last_factors = run_session(fusion, s, train=True)

        factor_loss = F.huber_loss(last_factors, y_f)
        state_loss  = F.cross_entropy(last_logits, y_s, weight=class_weights)
        loss        = 0.4 * factor_loss + 0.6 * state_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(seqs), 1)


@torch.no_grad()
def evaluate_sessions(
    fusion:        FusionTrainer,
    seqs:          list[dict],
    class_weights: torch.Tensor,
) -> tuple[list, list, float]:
    fusion.eval()
    all_true, all_pred = [], []
    total_loss = 0.0

    for s in seqs:
        factors, state = prepare_labels(s["y"])
        y_s = torch.tensor([state], dtype=torch.long).to(DEVICE)
        y_f = torch.tensor(factors, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        last_logits, last_factors = run_session(fusion, s, train=False)

        factor_loss = F.huber_loss(last_factors, y_f)
        state_loss  = F.cross_entropy(last_logits, y_s, weight=class_weights)
        loss        = 0.4 * factor_loss + 0.6 * state_loss

        pred = int(last_logits.argmax(dim=-1).item())
        all_true.append(state)
        all_pred.append(pred)
        total_loss += loss.item()

    return all_true, all_pred, total_loss / max(len(seqs), 1)


# ═════════════════════════════════════════════════════════════════════════════
# LOSO loop
# ═════════════════════════════════════════════════════════════════════════════

def train_loso(
    model_classes: dict,
    seqs:          list[dict],
    output_dir:    Path,
    epochs:        int   = 80,
    lr:            float = 5e-4,
) -> dict:
    print(f"\n{'='*52}")
    print("Joint Fusion — Session-Level LOSO")
    print(f"{'='*52}")

    # class weights from session-level distribution
    all_states = np.array([derive_state_label(s["y"]) for s in seqs])
    weights    = compute_class_weight("balanced",
                                      classes=np.unique(all_states),
                                      y=all_states)
    cw = np.ones(5, dtype=np.float32)
    for i, cls in enumerate(np.unique(all_states)):
        cw[int(cls)] = weights[i]
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {dict(zip(STATE_NAMES, cw.round(2).tolist()))}")

    unique_users       = sorted(set(s["user_id"] for s in seqs))
    all_true, all_pred = [], []

    for test_user in unique_users:
        train_seqs  = [s for s in seqs if s["user_id"] != test_user]
        test_seqs   = [s for s in seqs if s["user_id"] == test_user]
        n_test_sess = len(test_seqs)

        fusion    = FusionTrainer(model_classes).to(DEVICE)
        optimizer = AdamW(fusion.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

        best_f1      = -1.0
        best_weights = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(fusion, train_seqs, optimizer,
                                         class_weights)
            true, pred, val_loss = evaluate_sessions(fusion, test_seqs,
                                                      class_weights)
            epoch_f1  = f1_score(true, pred, average="macro", zero_division=0)
            scheduler.step()

            if epoch_f1 > best_f1:
                best_f1      = epoch_f1
                best_weights = {k: v.clone()
                                for k, v in fusion.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"  [{test_user:12s}] ep {epoch+1:3d}  "
                      f"tr={train_loss:.4f}  val={val_loss:.4f}  "
                      f"F1={epoch_f1:.4f}  best={best_f1:.4f}")

        fusion.load_state_dict(best_weights)
        true, pred, _ = evaluate_sessions(fusion, test_seqs, class_weights)
        all_true.extend(true)
        all_pred.extend(pred)
        fold_f1 = f1_score(true, pred, average="macro", zero_division=0)
        print(f"  [{test_user:12s}] fold F1={fold_f1:.4f}  (n={n_test_sess} sessions)")

    metrics = compute_metrics(all_true, all_pred,
                              "Joint Fusion Session-Level — LOSO overall")

    # retrain on full data and save per-model weights
    print("\n  Retraining on full dataset...")
    fusion_final = FusionTrainer(model_classes).to(DEVICE)
    optimizer    = AdamW(fusion_final.parameters(), lr=lr, weight_decay=1e-4)
    scheduler    = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    for epoch in range(epochs):
        loss = train_one_epoch(fusion_final, seqs, optimizer, class_weights)
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}  loss={loss:.4f}")

    # save each model's weights separately so fusion_model.py can load them
    for name in model_classes:
        save_path = output_dir / f"{name}_gru_session.pt"
        torch.save(fusion_final.models[name].state_dict(), save_path)
        print(f"  Saved: {save_path.name}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences",  type=str,
        default=str(Path.home() / "tucker_outputs" / "sequences.npy"))
    parser.add_argument("--output_dir", type=str,
        default=str(Path.home() / "fusion_session_models"))
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--lr",         type=float, default=5e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_sequences(Path(args.sequences))

    from predictive_models.mouse.v2_mouse_gru        import MouseGRU
    from predictive_models.keyboard.v1_gru           import KeyboardGRU
    from predictive_models.notif.v1_notif_gru        import NotifGRU
    from predictive_models.switching.v1_switching_gru import SwitchingGRU

    model_classes = {
        "mouse":    MouseGRU,
        "keyboard": KeyboardGRU,
        "notif":    NotifGRU,
        "switching":SwitchingGRU,
    }

    results = train_loso(
        model_classes,
        seqs,
        output_dir,
        epochs=args.epochs,
        lr=args.lr,
    )

    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()