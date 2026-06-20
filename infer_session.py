"""
infer_session.py
================
Runs the full fusion pipeline on sessions from sequences.npy.
Outputs per-window predictions showing how cognitive state evolves.

Pipeline: Tucker slices → 4 models → PoE → EMA → prediction per window

Usage:
    cd ~/fusion_model

    # run on all sessions, print timelines
    python infer_session.py \
        --sequences ~/tucker_outputs/sequences.npy \
        --weights_dir ~/fusion_session_models

    # run on one specific user
    python infer_session.py \
        --sequences ~/tucker_outputs/sequences.npy \
        --weights_dir ~/fusion_session_models \
        --user dem

    # save predictions to CSV
    python infer_session.py \
        --sequences ~/tucker_outputs/sequences.npy \
        --weights_dir ~/fusion_session_models \
        --save_csv ~/fusion_session_models/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

FUSION_DIR = Path(__file__).resolve().parent
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES = ["Flow", "Neutral", "Bored", "Distracted", "Overloaded"]
STATE_ICONS = ["🟢", "⬜", "🔵", "🟡", "🔴"]
MODALITY    = {"mouse": 0, "keyboard": 1, "notif": 2, "switching": 3}

NASA_COLS   = ["MD", "TD", "EF", "FR", "AR_proxy"]


# ─────────────────────────────────────────────────────────────────────────────
# Label derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_state_label(row: np.ndarray) -> int:
    md,td,ef,fr,perf = row[0],row[2],row[4],row[5],row[3]
    if (md+ef)/2 > 60 and fr > 60:                    return 4
    if td > 65 and fr <= 60:                           return 3
    if (md+ef)/2 < 35 and perf < 35 and fr <= 60:     return 0
    if (md+ef)/2 < 35 and perf >= 35:                 return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(weights_dir: Path) -> dict:
    from predictive_models.mouse.v2_mouse_gru         import MouseGRU
    from predictive_models.keyboard.v1_gru            import KeyboardGRU
    from predictive_models.notif.v1_notif_gru         import NotifGRU
    from predictive_models.switching.v1_switching_gru import SwitchingGRU

    specs = {
        "mouse":    (MouseGRU,    "mouse_gru_session.pt"),
        "keyboard": (KeyboardGRU, "keyboard_gru_session.pt"),
        "notif":    (NotifGRU,    "notif_gru_session.pt"),
        "switching":(SwitchingGRU,"switching_gru_session.pt"),
    }

    models = {}
    for name, (cls, fname) in specs.items():
        model = cls(input_flat_dim=512, d_proj=256).to(DEVICE)
        weight_path = weights_dir / fname
        if weight_path.exists():
            model.load_state_dict(
                torch.load(weight_path, map_location=DEVICE)
            )
            print(f"  ✓ loaded {fname}")
        else:
            print(f"  ✗ {fname} not found — using random weights")
        model.eval()
        models[name] = model

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Single-step inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_step(
    models:   dict,
    ema_state: torch.Tensor,   # (5,) previous EMA state
    slices:   dict,            # name → (1, 512) tensor
    alpha:    float = 0.7,
) -> tuple[int, np.ndarray, np.ndarray, torch.Tensor]:
    """
    One inference tick through the full pipeline.

    Returns:
        predicted_state : int
        probabilities   : (5,) numpy — EMA-smoothed probs
        factors         : (5,) numpy — averaged factor scores
        new_ema_state   : (5,) tensor
    """
    logit_sum    = None
    all_factors  = []

    for name, model in models.items():
        out = model(slices[name])          # (1, 12)

        # collect factors
        all_factors.append(out[0, :5].cpu().numpy())

        # PoE: confidence-weighted logit sum
        h_norm  = out[0, 10].item()
        weight  = max(1.0 - h_norm, 0.01)
        logits  = out[0, 5:10]             # (5,)

        if logit_sum is None:
            logit_sum = weight * logits
        else:
            logit_sum = logit_sum + weight * logits

    # softmax → probabilities
    raw_probs  = torch.softmax(logit_sum, dim=-1)          # (5,)

    # EMA smoothing
    new_ema    = alpha * raw_probs + (1.0 - alpha) * ema_state
    probs_np   = new_ema.cpu().numpy()

    predicted  = int(np.argmax(probs_np))
    factors    = np.mean(all_factors, axis=0)

    return predicted, probs_np, factors, new_ema


# ─────────────────────────────────────────────────────────────────────────────
# Session inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_session(
    models:  dict,
    session: dict,
    alpha:   float = 0.7,
) -> list[dict]:
    """
    Run the full fusion pipeline over one session.
    Returns list of per-window prediction records.
    """
    # reset all model microstates
    for model in models.values():
        model.reset_microstate()

    # uniform EMA prior
    ema_state = torch.full((5,), 0.2, dtype=torch.float32).to(DEVICE)

    X       = session["X"]      # (T, 4, 512)
    T       = session["T"]
    starts  = session.get("window_starts", list(range(T)))

    records = []

    for t in range(T):
        slices = {
            name: torch.tensor(
                X[t, idx, :], dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            for name, idx in MODALITY.items()
        }

        pred, probs, factors, ema_state = infer_step(
            models, ema_state, slices, alpha
        )

        t_start = starts[t] if t < len(starts) else t * 120
        records.append({
            "window_idx":   t,
            "window_start": float(t_start),
            "minutes":      round(float(t_start) / 60, 1),
            "predicted":    pred,
            "state_name":   STATE_NAMES[pred],
            "probs":        probs.tolist(),
            "factors":      factors.tolist(),
            "confidence":   float(np.max(probs)),
            "margin":       float(np.max(probs) - np.sort(probs)[-2]),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def print_session_timeline(session: dict, records: list[dict]):
    true_state = derive_state_label(session["y"])
    nasa        = session["y"]

    print(f"\n{'═'*70}")
    print(f"  User: {session['user_id']}   Session: {session['session_id']}")
    print(f"  True label  : {STATE_ICONS[true_state]} {STATE_NAMES[true_state]}")
    print(f"  NASA-TLX    : MD={nasa[0]:.0f} TD={nasa[2]:.0f} "
          f"EF={nasa[4]:.0f} FR={nasa[5]:.0f} perf={nasa[3]:.0f}")
    print(f"  Windows     : {len(records)}")
    print(f"{'─'*70}")
    print(f"  {'t':>4}  {'min':>5}  {'state':<12}  {'conf':>5}  "
          f"  Flow  Neut  Bore  Dist  Over")
    print(f"{'─'*70}")

    for r in records:
        bar = "  ".join(f"{p*100:4.1f}" for p in r["probs"])
        icon = STATE_ICONS[r["predicted"]]
        print(f"  {r['window_idx']:>4}  {r['minutes']:>4.1f}m  "
              f"{icon} {r['state_name']:<10}  {r['confidence']*100:4.1f}%"
              f"    {bar}")

    # final prediction
    final = records[-1]
    match = "✓" if final["predicted"] == true_state else "✗"
    print(f"{'─'*70}")
    print(f"  Final prediction: {STATE_ICONS[final['predicted']]} "
          f"{STATE_NAMES[final['predicted']]}  {match} "
          f"(true: {STATE_NAMES[true_state]})")


def print_summary(all_records: list[tuple]):
    """Print cross-session summary table."""
    print(f"\n{'═'*70}")
    print("  FULL DATASET PREDICTION SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'user':<12}  {'session':<10}  {'true':<12}  "
          f"{'predicted':<12}  {'match'}")
    print(f"{'─'*70}")

    correct = 0
    total   = 0
    for session, records in all_records:
        true  = derive_state_label(session["y"])
        pred  = records[-1]["predicted"]
        match = "✓" if pred == true else "✗"
        if pred == true:
            correct += 1
        total += 1
        print(f"  {session['user_id']:<12}  "
              f"{session['session_id'][-8:]:<10}  "
              f"{STATE_NAMES[true]:<12}  "
              f"{STATE_NAMES[pred]:<12}  {match}")

    print(f"{'─'*70}")
    print(f"  Session accuracy: {correct}/{total} = {100*correct/total:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences",   type=str,
        default=str(Path.home() / "tucker_outputs" / "sequences.npy"))
    parser.add_argument("--weights_dir", type=str,
        default=str(Path.home() / "fusion_session_models"))
    parser.add_argument("--user",        type=str, default=None,
        help="Filter to one user (e.g. --user dem)")
    parser.add_argument("--session",     type=str, default=None,
        help="Filter to one session ID substring")
    parser.add_argument("--alpha",       type=float, default=0.7,
        help="EMA smoothing factor (default 0.7)")
    parser.add_argument("--save_csv",    type=str, default=None,
        help="Save all predictions to a CSV file")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    sequences_path = Path(args.sequences)

    print("Loading models...")
    models = load_models(weights_dir)

    print(f"\nLoading sequences from {sequences_path}")
    seqs = list(np.load(sequences_path, allow_pickle=True))

    # filter
    if args.user:
        seqs = [s for s in seqs if s["user_id"] == args.user]
        print(f"Filtered to user '{args.user}': {len(seqs)} sessions")
    if args.session:
        seqs = [s for s in seqs if args.session in s["session_id"]]
        print(f"Filtered to session '{args.session}': {len(seqs)} sessions")

    if not seqs:
        print("No sessions found. Check --user / --session filters.")
        return

    # run inference
    all_records = []
    for session in seqs:
        records = infer_session(models, session, alpha=args.alpha)
        print_session_timeline(session, records)
        all_records.append((session, records))

    if len(all_records) > 1:
        print_summary(all_records)

    # optional CSV export
    if args.save_csv:
        csv_path = Path(args.save_csv)
        with csv_path.open("w", newline="") as f:
            fieldnames = [
                "user_id", "session_id", "window_idx", "minutes",
                "true_state", "predicted_state",
                "prob_flow", "prob_neutral", "prob_bored",
                "prob_distracted", "prob_overloaded",
                "confidence", "margin",
                "factor_md", "factor_td", "factor_ef",
                "factor_fr", "factor_ar",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for session, records in all_records:
                true_state = derive_state_label(session["y"])
                for r in records:
                    writer.writerow({
                        "user_id":           session["user_id"],
                        "session_id":        session["session_id"],
                        "window_idx":        r["window_idx"],
                        "minutes":           r["minutes"],
                        "true_state":        STATE_NAMES[true_state],
                        "predicted_state":   r["state_name"],
                        "prob_flow":         round(r["probs"][0], 4),
                        "prob_neutral":      round(r["probs"][1], 4),
                        "prob_bored":        round(r["probs"][2], 4),
                        "prob_distracted":   round(r["probs"][3], 4),
                        "prob_overloaded":   round(r["probs"][4], 4),
                        "confidence":        round(r["confidence"], 4),
                        "margin":            round(r["margin"], 4),
                        "factor_md":         round(r["factors"][0], 4),
                        "factor_td":         round(r["factors"][1], 4),
                        "factor_ef":         round(r["factors"][2], 4),
                        "factor_fr":         round(r["factors"][3], 4),
                        "factor_ar":         round(r["factors"][4], 4),
                    })
        print(f"\nPredictions saved → {csv_path}")


if __name__ == "__main__":
    main()