"""OPTIONAL downstream probe for the final embedders (analysis only).

Freezes an embedder, builds windows, attaches NASA-TLX session labels to each
window (via metadata.json session->label), and fits a small ridge linear probe
embedding -> NASA factor. Reports MAE/RMSE/R2 with a session-disjoint split.

This is NOT a training signal and NOT a readiness criterion. Behavioral
validation (validate_embedders_final.py) remains the gate. A probe result is
reported only as additional context.

Usage:
  python scripts/embedders/probe_embedders.py --modality keyboard --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.embedders import embedder_common as ec

NASA_COLS = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort",
             "frustration", "stress_self_report", "valence", "arousal"]


def _load_out(export_dir: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_probe_out", export_dir / "output.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _session_nasa(data_dir: Path):
    """session_id -> NASA vector (9,) from raw/labels.csv (first row)."""
    import csv
    out = {}
    for lab in sorted(data_dir.glob("session_*/raw/labels.csv")):
        sid = lab.parents[1].name
        row = next(csv.DictReader(lab.open("r", encoding="utf-8", newline="")), None)
        if not row:
            continue
        try:
            out[sid] = np.array([float(row.get(c, "nan")) for c in NASA_COLS], np.float64)
        except ValueError:
            continue
    return out


def _ridge_probe(E, y, sessions, lam=10.0, seed=42):
    uniq = sorted(set(sessions))
    rng = np.random.default_rng(seed); perm = list(rng.permutation(uniq))
    n_te = max(1, int(round(len(perm) * 0.3)))
    te = set(perm[:n_te])
    tr_m = np.array([s not in te for s in sessions]); te_m = ~tr_m
    if tr_m.sum() < 10 or te_m.sum() < 5:
        return None
    Xtr, Xte = E[tr_m], E[te_m]; ytr, yte = y[tr_m], y[te_m]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ (ytr - ytr.mean()))
    b = ytr.mean()
    pred = Xte @ W + b
    err = pred - yte
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": (1 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan"),
            "n_train": int(tr_m.sum()), "n_test": int(te_m.sum())}


def main():
    ap = argparse.ArgumentParser(description="Optional downstream probe (analysis only).")
    ap.add_argument("--modality", choices=["keyboard", "mouse"], required=True)
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/embedders/probe")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nasa = _session_nasa(args.data_dir)
    if args.modality == "keyboard":
        export = PROJECT_ROOT / "pre_embedders/keyboard/exports/keyboard_encoder_final"
        mod = _load_out(export, "keyboard"); sess = mod.load_model(export_dir=export, device=args.device)
        X, F, sessions, _ = ec.build_keyboard_windows(args.data_dir)
        E = np.stack([mod.get_output(sess, X[i])["embedding"] for i in range(len(X))])
    else:
        export = PROJECT_ROOT / "pre_embedders/mouse/exports/mouse_encoder_final"
        mod = _load_out(export, "mouse"); sess = mod.load_model(export_dir=export, device=args.device)
        SEQ, F, sessions, _ = ec.build_mouse_windows(args.data_dir, max_sessions=20)
        E = np.stack([mod.get_output(sess, {"seq": SEQ[i], "stats": F[i]})["embedding"] for i in range(len(SEQ))])

    keep = np.array([s in nasa and np.isfinite(nasa[s]).all() for s in sessions])
    if keep.sum() < 20:
        print(f"[probe] too few labelled windows ({int(keep.sum())}); skipping."); return
    E, sessions = E[keep], [s for s, k in zip(sessions, keep) if k]
    Y = np.stack([nasa[s] for s in sessions])

    results = {}
    for j, col in enumerate(NASA_COLS):
        r = _ridge_probe(E, Y[:, j], sessions)
        if r:
            results[col] = r
    payload = {"modality": args.modality, "n_labelled_windows": int(keep.sum()),
               "note": "ANALYSIS ONLY — not a readiness criterion; behavioral validation is the gate.",
               "probe": results}
    (args.output_dir / f"{args.modality}_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
