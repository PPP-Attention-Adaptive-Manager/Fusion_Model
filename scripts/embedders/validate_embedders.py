"""Independent, per-embedder validation stage (run BEFORE any Tucker rebuild).

Rationale
---------
The Low-Rank Tucker fusion is *interactional*: it forms an outer product across
all four modalities, so a single invalid / degenerate embedder contaminates
EVERY modality slice. Therefore each embedder must be validated **in isolation**
on real raw session data before it is ever wired into Tucker.

What this does, per embedder (mouse, keyboard):
  1. run the EXPORTED encoder over real raw session data
  2. save embeddings + per-window metadata
  3. compute health stats: non-zero / NaN / Inf / variance / duplicate ratio / L2-norm
  4. test temporal stability (consecutive-window cosine within a session)
  5. test behavioral correlation (embedding vs simple hand features)
  6. read trained/pretrained status from the checkpoint / model_config.json
  7. assign a verdict: PASS / WARNING / FAIL

Hard rules (encoded below):
  * keyboard CAN be validated (it has trained weights).
  * mouse MUST be FAIL/WARNING while trained=false / pretrained=false — an
    untrained encoder is never allowed to PASS, regardless of mechanical stats.
  * this script NEVER rebuilds Tucker and NEVER uses a seeded mouse in Tucker.

Outputs (outputs/embedders/validation/):
  keyboard_embeddings.npy, keyboard_windows.csv
  mouse_embeddings.npy,    mouse_windows.csv
  embedder_validation.json

Usage:
  python scripts/embedders/validate_embedders.py --raw-data-dir data --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

OUT_DIR = PROJECT_ROOT / "outputs" / "embedders" / "validation"

# Verdict thresholds (documented in the report).
VARIANCE_MIN = 1e-6            # below -> collapsed
DUP_COSINE = 0.9999           # near-identical embeddings
DUP_RATIO_FAIL = 0.90         # >90% duplicates -> collapse FAIL
DUP_RATIO_WARN = 0.50
CORR_WARN = 0.20              # |r| below -> behavioral signal too weak (WARNING)
TEMPORAL_CONST_COS = 0.9999  # consecutive cosine ~1 everywhere -> constant output
MIN_VALID = 30               # too few windows to judge -> WARNING


# --------------------------------------------------------------------------- #
# generic metric helpers
# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def health_stats(E: np.ndarray) -> Dict[str, Any]:
    n = int(E.shape[0])
    row_nonzero = np.abs(E).sum(axis=1) > 0
    valid = E[row_nonzero]
    norms = np.linalg.norm(E, axis=1)
    out: Dict[str, Any] = {
        "n_windows": n,
        "n_cold_start_zero": int((~row_nonzero).sum()),
        "n_valid": int(row_nonzero.sum()),
        "has_nan": bool(np.isnan(E).any()),
        "has_inf": bool(np.isinf(E).any()),
        "variance_mean": float(valid.var()) if valid.size else 0.0,
        "per_dim_var_min": float(valid.var(axis=0).min()) if valid.size else 0.0,
        "per_dim_var_max": float(valid.var(axis=0).max()) if valid.size else 0.0,
        "l2_norm_mean": float(norms[row_nonzero].mean()) if row_nonzero.any() else 0.0,
        "l2_norm_std": float(norms[row_nonzero].std()) if row_nonzero.any() else 0.0,
        "l2_norm_min": float(norms[row_nonzero].min()) if row_nonzero.any() else 0.0,
        "l2_norm_max": float(norms[row_nonzero].max()) if row_nonzero.any() else 0.0,
    }
    out["duplicate_ratio"] = duplicate_ratio(valid)
    return out


def duplicate_ratio(valid: np.ndarray) -> float:
    """Fraction of valid embeddings whose nearest neighbor cosine > DUP_COSINE."""
    if valid.shape[0] < 2:
        return 0.0
    norm = valid / (np.linalg.norm(valid, axis=1, keepdims=True) + 1e-12)
    # cap memory for very large N
    if norm.shape[0] > 6000:
        rng = np.random.default_rng(0)
        norm = norm[rng.choice(norm.shape[0], 6000, replace=False)]
    sim = norm @ norm.T
    np.fill_diagonal(sim, -1.0)
    nn = sim.max(axis=1)
    return float(np.mean(nn > DUP_COSINE))


def temporal_stability(E: np.ndarray, sessions: List[str]) -> Dict[str, float]:
    """Mean cosine similarity between consecutive windows within each session."""
    cos_vals: List[float] = []
    by_sess: Dict[str, List[int]] = {}
    for i, s in enumerate(sessions):
        by_sess.setdefault(s, []).append(i)
    for idxs in by_sess.values():
        for a, b in zip(idxs[:-1], idxs[1:]):
            va, vb = E[a], E[b]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 1e-9 and nb > 1e-9:
                cos_vals.append(float(va @ vb / (na * nb)))
    if not cos_vals:
        return {"consecutive_cosine_mean": float("nan"), "n_pairs": 0}
    return {
        "consecutive_cosine_mean": float(np.mean(cos_vals)),
        "consecutive_cosine_std": float(np.std(cos_vals)),
        "n_pairs": len(cos_vals),
    }


def behavioral_correlation(E: np.ndarray, B: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Correlate the embedding's 1st principal component with each hand feature.

    Returns the best |Pearson r| over features and a per-feature breakdown.
    """
    row_nonzero = np.abs(E).sum(axis=1) > 0
    Ev = E[row_nonzero]
    if Ev.shape[0] < 5:
        return {"best_abs_r": float("nan"), "best_feature": None, "per_feature": {}}
    # PC1 score of valid embeddings
    Ec = Ev - Ev.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
        pc1 = Ec @ Vt[0]
    except np.linalg.LinAlgError:
        return {"best_abs_r": float("nan"), "best_feature": None, "per_feature": {}}
    per_feature: Dict[str, float] = {}
    best_abs, best_feat = 0.0, None
    for name, vals in B.items():
        v = np.asarray(vals, dtype=np.float64)[row_nonzero]
        r = _pearson(pc1, v)
        per_feature[name] = r
        if not np.isnan(r) and abs(r) > best_abs:
            best_abs, best_feat = abs(r), name
    return {"best_abs_r": float(best_abs), "best_feature": best_feat, "per_feature": per_feature}


# --------------------------------------------------------------------------- #
# KEYBOARD: run exported encoder on real keystroke windows
# --------------------------------------------------------------------------- #
def collect_keyboard(raw_dir: Path, device: str, window_size: int = 20, stride: int = 10):
    from pre_embedders.keyboard import parse_csv_events
    from pre_embedders.keyboard.output import get_output, load_model

    session = load_model(device=device)
    config = session.get("model_config", {})
    trained = _checkpoint_flag(PROJECT_ROOT / "pre_embedders/keyboard/exports/keyboard_encoder/encoder.pt",
                               keys=("trained",))

    embs: List[np.ndarray] = []
    sessions: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    B = {"mean_hold": [], "mean_ikl": [], "typing_speed": [], "std_hold": []}

    for kb_csv in sorted(raw_dir.glob("**/keyboard.csv")):
        sid = kb_csv.parents[1].name if kb_csv.parent.name in ("raw", "labels", "events", "features") else kb_csv.parent.name
        try:
            rows = list(csv.DictReader(kb_csv.open("r", encoding="utf-8", newline="")))
        except OSError:
            continue
        events = parse_csv_events(rows)
        if len(events) < window_size:
            continue
        for start in range(0, len(events) - window_size + 1, stride):
            chunk = events[start:start + window_size]
            res = get_output(session, chunk)
            emb = res["embedding"]
            holds = np.array([e["hold"] for e in chunk], dtype=np.float64)
            ikls = np.array([e["ikl"] for e in chunk], dtype=np.float64)
            mean_ikl = float(np.mean(ikls)) if ikls.size else 0.0
            embs.append(emb)
            sessions.append(sid)
            B["mean_hold"].append(float(np.mean(holds)))
            B["mean_ikl"].append(mean_ikl)
            B["typing_speed"].append(1000.0 / mean_ikl if mean_ikl > 1e-6 else 0.0)
            B["std_hold"].append(float(np.std(holds)))
            meta_rows.append({
                "session_id": sid, "window_start_event": start, "window_size": window_size,
                "mean_hold": B["mean_hold"][-1], "mean_ikl": mean_ikl,
                "cold_start": res["metadata"]["cold_start"],
            })

    E = np.asarray(embs, dtype=np.float32) if embs else np.zeros((0, 64), np.float32)
    Bn = {k: np.asarray(v, dtype=np.float64) for k, v in B.items()}
    return E, sessions, Bn, meta_rows, {"config": config, "trained": trained, "variant": session.get("variant")}


# --------------------------------------------------------------------------- #
# MOUSE: run exported encoder on real 120s windows
# --------------------------------------------------------------------------- #
def _fast_load_mouse_csv(path: Path):
    """Fast MouseEvent loader (csv.DictReader) — avoids pandas iterrows()."""
    from pre_embedders.mouse.mouse_encoder import MouseEvent

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    events = []
    with path.open("r", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            ts = _f(row.get("timestamp"), None)
            if ts is None:
                continue
            btn = row.get("button")
            events.append(MouseEvent(
                timestamp=ts, x=_f(row.get("x")), y=_f(row.get("y")),
                dx=_f(row.get("delta_x")), dy=_f(row.get("delta_y")),
                speed=_f(row.get("speed")),
                event_type=str(row.get("event_type", "mouse_move")),
                button=btn if btn not in (None, "", "nan") else None,
            ))
    events.sort(key=lambda e: e.timestamp)
    return events


def collect_mouse(raw_dir: Path, device: str, min_events: int = 20,
                  max_sessions: Optional[int] = 20, max_windows: Optional[int] = 40):
    from pre_embedders.mouse.mouse_encoder import (
        build_pre_click_subsequence,
        build_tcn_sequence,
        compute_per_event_derivatives,
        extract_window_stats,
    )
    from pre_embedders.mouse.output import get_output, load_model

    # Mouse runs on CPU on purpose: single-sample TCN conv forwards are faster on
    # CPU and avoid the CUDA conv-autotune stall. Mouse FAILs the trained gate
    # regardless, so a bounded mechanical sample is sufficient.
    session = load_model(device="cpu")
    config = session.get("model_config", {})
    trained = _checkpoint_flag(PROJECT_ROOT / "pre_embedders/mouse/exports/mouse_encoder/encoder.pt",
                               keys=("trained", "pretrained"))

    def _graph_windows(session_dir: Path) -> List[Tuple[float, float]]:
        gdir = session_dir / "data_graph" / "data_graph_120s"
        wins = []
        for g in sorted(gdir.glob("graph_*.json")):
            try:
                w = json.loads(g.read_text(encoding="utf-8")).get("window", {})
                ws, we = w.get("window_start"), w.get("window_end")
                if ws is not None and we is not None:
                    wins.append((float(ws), float(we)))
            except (OSError, json.JSONDecodeError):
                continue
        return wins

    embs: List[np.ndarray] = []
    sessions: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    B = {"speed_mean": [], "click_rate": [], "idle_ratio": [], "n_events": []}

    mouse_csvs = sorted(raw_dir.glob("**/raw/mouse.csv"))
    processed = 0
    for mouse_csv in mouse_csvs:
        sid = mouse_csv.parents[1].name
        session_dir = mouse_csv.parents[1]
        wins = _graph_windows(session_dir)
        if not wins:
            continue
        if max_sessions is not None and processed >= max_sessions:
            break
        try:
            events = _fast_load_mouse_csv(mouse_csv)
        except Exception:  # noqa: BLE001
            continue
        if not events:
            continue
        if max_windows is not None and len(embs) >= max_windows:
            break
        processed += 1
        print(f"[validate]   mouse session {processed}: {sid} "
              f"({len(events)} events, {len(wins)} windows) | total emb so far={len(embs)}", flush=True)
        ev_ts = np.array([e.timestamp for e in events])
        for (ws, we) in wins:
            if max_windows is not None and len(embs) >= max_windows:
                break
            lo = int(np.searchsorted(ev_ts, ws, side="left"))
            hi = int(np.searchsorted(ev_ts, we, side="left"))
            win_events = events[lo:hi]
            if len(win_events) < min_events:
                continue
            try:
                d = compute_per_event_derivatives(win_events)
                stats = extract_window_stats(win_events)
                if stats is None:
                    continue
                seq = build_tcn_sequence(win_events, d)
                pre_click = build_pre_click_subsequence(win_events, d)
                stats_arr = torch.tensor(stats.to_array(), dtype=torch.float32).unsqueeze(0)
                payload = {"seq": seq, "stats": stats_arr, "pre_click_seq": pre_click}
                res = get_output(session, payload)
            except Exception:  # noqa: BLE001
                continue
            emb = res["embedding"]
            embs.append(emb)
            sessions.append(sid)
            sa = stats.to_array()
            B["speed_mean"].append(float(sa[0]))
            B["click_rate"].append(float(sa[18]) if len(sa) > 18 else 0.0)
            B["idle_ratio"].append(float(sa[16]) if len(sa) > 16 else 0.0)
            B["n_events"].append(float(len(win_events)))
            meta_rows.append({
                "session_id": sid, "window_start": ws, "window_end": we,
                "n_events": len(win_events), "cold_start": res["metadata"]["cold_start"],
            })

    E = np.asarray(embs, dtype=np.float32) if embs else np.zeros((0, 64), np.float32)
    Bn = {k: np.asarray(v, dtype=np.float64) for k, v in B.items()}
    return E, sessions, Bn, meta_rows, {"config": config, "trained": trained}


def _checkpoint_flag(path: Path, keys: Tuple[str, ...]) -> Optional[bool]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    for k in keys:
        if isinstance(ckpt, dict) and k in ckpt:
            return bool(ckpt[k])
    return None


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def decide_verdict(name: str, trained: Optional[bool], stats: Dict[str, Any],
                   temporal: Dict[str, float], behav: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    verdict = "PASS"

    # Hard gate: untrained encoder can never PASS.
    if trained is not True:
        verdict = "FAIL"
        reasons.append(f"trained/pretrained status is {trained!r} (untrained encoder cannot pass)")

    # Mechanical health (can force FAIL even if trained).
    if stats["has_nan"] or stats["has_inf"]:
        verdict = "FAIL"; reasons.append("NaN/Inf in embeddings")
    if stats["n_valid"] == 0:
        verdict = "FAIL"; reasons.append("all embeddings are zero (cold start)")
    elif stats["variance_mean"] < VARIANCE_MIN:
        verdict = "FAIL"; reasons.append(f"variance≈0 ({stats['variance_mean']:.2e}) -> collapsed")
    if stats["duplicate_ratio"] > DUP_RATIO_FAIL:
        verdict = "FAIL"; reasons.append(f"duplicate_ratio {stats['duplicate_ratio']:.2f} > {DUP_RATIO_FAIL}")

    # Quality warnings (do not downgrade an already-FAIL).
    warn: List[str] = []
    if stats["n_valid"] < MIN_VALID:
        warn.append(f"only {stats['n_valid']} valid windows (< {MIN_VALID})")
    tc = temporal.get("consecutive_cosine_mean", float("nan"))
    if not np.isnan(tc) and tc > TEMPORAL_CONST_COS:
        warn.append(f"consecutive cosine ≈ {tc:.4f} -> near-constant output")
    br = behav.get("best_abs_r", float("nan"))
    if np.isnan(br) or br < CORR_WARN:
        warn.append(f"weak behavioral correlation (best |r|={br})")
    if stats["duplicate_ratio"] > DUP_RATIO_WARN:
        warn.append(f"elevated duplicate_ratio {stats['duplicate_ratio']:.2f}")

    if verdict == "PASS" and warn:
        verdict = "WARNING"
    reasons.extend(warn)
    return {"verdict": verdict, "reasons": reasons}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def validate_one(name: str, E, sessions, B, meta_rows, info) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}_embeddings.npy", E)
    _write_csv(OUT_DIR / f"{name}_windows.csv", meta_rows)

    stats = health_stats(E)
    temporal = temporal_stability(E, sessions)
    behav = behavioral_correlation(E, B)
    trained = info.get("trained")
    decision = decide_verdict(name, trained, stats, temporal, behav)

    result = {
        "embedder": name,
        "trained_or_pretrained": trained,
        "model_config": info.get("config"),
        "n_sessions": len(set(sessions)),
        "health": stats,
        "temporal_stability": temporal,
        "behavioral_correlation": behav,
        **decision,
    }
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Independent per-embedder validation (pre-Tucker).")
    ap.add_argument("--raw-data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mouse-max-sessions", type=int, default=20,
                    help="cap mouse sessions (mouse FAILs the trained gate anyway; subset is enough)")
    ap.add_argument("--mouse-max-windows", type=int, default=40,
                    help="hard cap on total mouse windows (bounded mechanical sample)")
    ap.add_argument("--skip-keyboard", action="store_true",
                    help="reuse the saved keyboard_validation.json instead of recomputing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kb_saved = OUT_DIR / "keyboard_validation.json"
        if args.skip_keyboard and kb_saved.is_file():
            kb_result = json.loads(kb_saved.read_text(encoding="utf-8"))
            print(f"[validate] keyboard: reused saved result -> {kb_result['verdict']}")
        else:
            print("[validate] keyboard: running exported encoder on real keystroke windows...")
            kb = collect_keyboard(args.raw_data_dir, args.device)
            kb_result = validate_one("keyboard", *kb)
            print(f"[validate] keyboard: {kb_result['verdict']} "
                  f"({kb_result['health']['n_valid']} valid windows)")
            # Persist keyboard result immediately so a slow mouse pass can't lose it.
            kb_saved.write_text(json.dumps(kb_result, indent=2), encoding="utf-8")

        print(f"[validate] mouse: running exported encoder on real 120s windows "
              f"(max {args.mouse_max_sessions} sessions)...")
        ms = collect_mouse(args.raw_data_dir, args.device,
                           max_sessions=args.mouse_max_sessions, max_windows=args.mouse_max_windows)
        ms_result = validate_one("mouse", *ms)
        print(f"[validate] mouse: {ms_result['verdict']} "
              f"({ms_result['health']['n_valid']} valid windows)")

    summary = {
        "tucker_rebuild_allowed": kb_result["verdict"] in ("PASS", "WARNING")
                                   and ms_result["verdict"] in ("PASS", "WARNING"),
        "embedders": {"keyboard": kb_result, "mouse": ms_result},
    }
    out_path = OUT_DIR / "embedder_validation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {out_path}")
    if not summary["tucker_rebuild_allowed"]:
        print("\n[GATE] Tucker rebuild is BLOCKED — at least one embedder did not pass.")


if __name__ == "__main__":
    main()
