"""Independent validation of the FINAL mouse + keyboard embedders (Tucker gate).

Runs each EXPORTED encoder over real raw session windows, computes health +
behavioral metrics, and assigns PASS / WARNING / FAIL. Tucker rebuild is allowed
only if BOTH embedders are PASS.

Usage:
  python scripts/embedders/validate_embedders_final.py --data-dir data --device cpu \
    --keyboard-export pre_embedders/keyboard/exports/keyboard_encoder_final \
    --mouse-export pre_embedders/mouse/exports/mouse_encoder_final \
    --output-dir outputs/embedders/final_validation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.embedders import embedder_common as ec

PREV_KEYBOARD_VARIANCE = 5.6e-5  # previous (WARNING) version, to require improvement


def _load_output_module(export_dir: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_final_output", export_dir / "output.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pearson(a, b):
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _effective_rank(E: np.ndarray) -> float:
    if E.shape[0] < 2:
        return 0.0
    Ec = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(Ec, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _duplicate_ratio(valid: np.ndarray) -> float:
    if valid.shape[0] < 2:
        return 0.0
    nrm = valid / (np.linalg.norm(valid, axis=1, keepdims=True) + 1e-12)
    if nrm.shape[0] > 6000:
        nrm = nrm[np.random.default_rng(0).choice(nrm.shape[0], 6000, replace=False)]
    sim = nrm @ nrm.T
    np.fill_diagonal(sim, -1.0)
    return float(np.mean(sim.max(1) > 0.9999))


def _temporal(E, sessions):
    by: Dict[str, List[int]] = {}
    for i, s in enumerate(sessions):
        by.setdefault(s, []).append(i)
    cos = []
    for idxs in by.values():
        for a, b in zip(idxs[:-1], idxs[1:]):
            na, nb = np.linalg.norm(E[a]), np.linalg.norm(E[b])
            if na > 1e-9 and nb > 1e-9:
                cos.append(float(E[a] @ E[b] / (na * nb)))
    if not cos:
        return {"consecutive_cosine_mean": float("nan"), "consecutive_cosine_std": float("nan"), "n_pairs": 0}
    return {"consecutive_cosine_mean": float(np.mean(cos)),
            "consecutive_cosine_std": float(np.std(cos)), "n_pairs": len(cos)}


def _behavioral(E, F, feature_names):
    """Max single-dim |Pearson r| between each behavioral feature and any embedding
    dim (comparable to the previous validation metric)."""
    row_nonzero = np.abs(E).sum(1) > 0
    Ev, Fv = E[row_nonzero], F[row_nonzero]
    per = {}
    best_abs, best_feat = 0.0, None
    for j, name in enumerate(feature_names):
        col = Fv[:, j]
        best = 0.0
        for d in range(Ev.shape[1]):
            r = _pearson(Ev[:, d], col)
            if not np.isnan(r) and abs(r) > best:
                best = abs(r)
        per[name] = best
        if best > best_abs:
            best_abs, best_feat = best, name
    return {"best_abs_r": float(best_abs), "best_feature": best_feat, "per_feature": per}


def _health(E):
    n = int(E.shape[0])
    nz = np.abs(E).sum(1) > 0
    valid = E[nz]
    norms = np.linalg.norm(E, axis=1)
    return {
        "n_windows": n, "n_valid": int(nz.sum()), "n_cold_start_zero": int((~nz).sum()),
        "cold_start_ratio": float((~nz).mean()) if n else 1.0,
        "nan_count": int(np.isnan(E).sum()), "inf_count": int(np.isinf(E).sum()),
        "all_zero_rows": int((~nz).sum()),
        "duplicate_ratio": _duplicate_ratio(valid),
        "l2_norm_mean": float(norms[nz].mean()) if nz.any() else 0.0,
        "l2_norm_std": float(norms[nz].std()) if nz.any() else 0.0,
        "l2_norm_min": float(norms[nz].min()) if nz.any() else 0.0,
        "l2_norm_max": float(norms[nz].max()) if nz.any() else 0.0,
        "variance_mean": float(valid.var()) if valid.size else 0.0,
        "per_dim_var_min": float(valid.var(0).min()) if valid.size else 0.0,
        "per_dim_var_max": float(valid.var(0).max()) if valid.size else 0.0,
        "effective_rank": _effective_rank(valid),
    }


def _verdict_keyboard(trained, h, temporal, behav):
    reasons = []
    ok = True
    if trained is not True:
        ok = False; reasons.append("trained != true")
    if h["nan_count"] or h["inf_count"]:
        ok = False; reasons.append("NaN/Inf present")
    if h["n_valid"] < 100:
        ok = False; reasons.append(f"n_valid {h['n_valid']} < 100")
    if h["cold_start_ratio"] >= 0.05:
        ok = False; reasons.append(f"cold-start ratio {h['cold_start_ratio']:.2%} >= 5%")
    if h["duplicate_ratio"] >= 0.2:
        ok = False; reasons.append(f"duplicate_ratio {h['duplicate_ratio']:.2f} >= 0.2")
    if h["variance_mean"] <= PREV_KEYBOARD_VARIANCE:
        ok = False; reasons.append(f"variance_mean {h['variance_mean']:.2e} not > previous {PREV_KEYBOARD_VARIANCE:.2e}")
    if behav["best_abs_r"] < 0.2:
        ok = False; reasons.append(f"best behavioral |r| {behav['best_abs_r']:.3f} < 0.2")
    return ("PASS" if ok else "FAIL"), reasons


def _verdict_mouse(trained, h, temporal, behav):
    reasons = []
    ok = True
    if trained is not True:
        ok = False; reasons.append("trained != true")
    if h["nan_count"] or h["inf_count"]:
        ok = False; reasons.append("NaN/Inf present")
    if h["n_valid"] < 100:
        ok = False; reasons.append(f"n_valid {h['n_valid']} < 100")
    if h["cold_start_ratio"] >= 0.10:
        ok = False; reasons.append(f"cold-start ratio {h['cold_start_ratio']:.2%} >= 10%")
    if h["duplicate_ratio"] >= 0.2:
        ok = False; reasons.append(f"duplicate_ratio {h['duplicate_ratio']:.2f} >= 0.2")
    if behav["best_abs_r"] < 0.2:
        ok = False; reasons.append(f"best behavioral |r| {behav['best_abs_r']:.3f} < 0.2")
    tc = temporal["consecutive_cosine_mean"]
    if np.isnan(tc) or tc < 0.02 or tc > 0.98:
        ok = False; reasons.append(f"temporal cosine collapsed (mean={tc})")
    return ("PASS" if ok else "FAIL"), reasons


def validate_keyboard(data_dir, export_dir, device, out_dir):
    mod = _load_output_module(export_dir, "keyboard")
    session = mod.load_model(export_dir=export_dir, device=device)
    trained = bool(session["config"].get("trained", False))
    X, F, sessions, _ = ec.build_keyboard_windows(data_dir)
    embs, meta = [], []
    for i in range(len(X)):
        r = mod.get_output(session, X[i])
        embs.append(r["embedding"])
        meta.append({"session_id": sessions[i], "cold_start": r["metadata"]["cold_start"]})
    E = np.asarray(embs, np.float32)
    np.save(out_dir / "keyboard_embeddings.npy", E)
    ec.write_csv(out_dir / "keyboard_windows.csv", meta)
    h = _health(E); temporal = _temporal(E, sessions); behav = _behavioral(E, F, ec.KB_FEATURES)
    status, reasons = _verdict_keyboard(trained, h, temporal, behav)
    return {"embedder": "keyboard", "trained": trained, "n_sessions": len(set(sessions)),
            "health": h, "temporal_stability": temporal, "behavioral_correlation": behav,
            "status": status, "reasons": reasons,
            "allowed_for_tucker": status == "PASS"}


def validate_mouse(data_dir, export_dir, device, out_dir, max_sessions=None):
    mod = _load_output_module(export_dir, "mouse")
    session = mod.load_model(export_dir=export_dir, device=device)
    trained = bool(session["config"].get("trained", False))
    SEQ, F, sessions, _ = ec.build_mouse_windows(data_dir, max_sessions=max_sessions)
    embs, meta = [], []
    for i in range(len(SEQ)):
        r = mod.get_output(session, {"seq": SEQ[i], "stats": F[i]})
        embs.append(r["embedding"])
        meta.append({"session_id": sessions[i], "n_events": float(F[i][ec.MOUSE_FEATURES.index("n_events")]),
                     "cold_start": r["metadata"]["cold_start"]})
    E = np.asarray(embs, np.float32)
    np.save(out_dir / "mouse_embeddings.npy", E)
    ec.write_csv(out_dir / "mouse_windows.csv", meta)
    h = _health(E); temporal = _temporal(E, sessions); behav = _behavioral(E, F, ec.MOUSE_FEATURES)
    status, reasons = _verdict_mouse(trained, h, temporal, behav)
    return {"embedder": "mouse", "trained": trained, "n_sessions": len(set(sessions)),
            "health": h, "temporal_stability": temporal, "behavioral_correlation": behav,
            "status": status, "reasons": reasons,
            "allowed_for_tucker": status == "PASS"}


def parse_args():
    ap = argparse.ArgumentParser(description="Validate final mouse/keyboard embedders (Tucker gate).")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--keyboard-export", type=Path,
                    default=PROJECT_ROOT / "pre_embedders/keyboard/exports/keyboard_encoder_final")
    ap.add_argument("--mouse-export", type=Path,
                    default=PROJECT_ROOT / "pre_embedders/mouse/exports/mouse_encoder_final")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/embedders/final_validation")
    ap.add_argument("--mouse-max-sessions", type=int, default=None)
    ap.add_argument("--only", choices=["keyboard", "mouse", "both"], default="both")
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.only in ("keyboard", "both"):
        print("[validate-final] keyboard ...")
        kb = validate_keyboard(args.data_dir, args.keyboard_export, args.device, out)
        (out / "keyboard_validation.json").write_text(json.dumps(kb, indent=2), encoding="utf-8")
        print(f"  keyboard: {kb['status']} | best|r|={kb['behavioral_correlation']['best_abs_r']:.3f} "
              f"var={kb['health']['variance_mean']:.2e} cold={kb['health']['cold_start_ratio']:.2%}")
        results["keyboard"] = kb
    if args.only in ("mouse", "both"):
        print("[validate-final] mouse ...")
        ms = validate_mouse(args.data_dir, args.mouse_export, args.device, out, args.mouse_max_sessions)
        (out / "mouse_validation.json").write_text(json.dumps(ms, indent=2), encoding="utf-8")
        print(f"  mouse: {ms['status']} | best|r|={ms['behavioral_correlation']['best_abs_r']:.3f} "
              f"cold={ms['health']['cold_start_ratio']:.2%} "
              f"tcos={ms['temporal_stability']['consecutive_cosine_mean']}")
        results["mouse"] = ms

    summary = {
        "keyboard": {"status": results.get("keyboard", {}).get("status"),
                     "allowed_for_tucker": results.get("keyboard", {}).get("allowed_for_tucker")},
        "mouse": {"status": results.get("mouse", {}).get("status"),
                  "allowed_for_tucker": results.get("mouse", {}).get("allowed_for_tucker")},
    }
    summary["tucker_rebuild_allowed"] = bool(
        summary["keyboard"].get("allowed_for_tucker") and summary["mouse"].get("allowed_for_tucker"))
    (out / "embedder_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["tucker_rebuild_allowed"]:
        print("\n[GATE] Tucker rebuild BLOCKED — both embedders must PASS.")


if __name__ == "__main__":
    main()
