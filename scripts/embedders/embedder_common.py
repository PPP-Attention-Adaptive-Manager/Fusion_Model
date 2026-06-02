"""Shared utilities for the FINAL mouse/keyboard embedders.

Contains everything reused by training, validation, and export:
  * device / seed helpers, session-level split
  * NT-Xent (SimCLR) contrastive loss with REAL positive pairs
  * a numpy RobustScaler (median/IQR) for auxiliary-target standardization
  * window builders that turn raw session CSVs into model-ready tensors +
    per-window behavioral features, for BOTH modalities
  * augmentation functions for contrastive views

Design choices (justified in FINAL_MOUSE_KEYBOARD_EMBEDDER_REPORT.md):
  - Keyboard windows: 20 keystrokes / stride 10 (the native cadence), features
    [hold, ikl, code] normalized per window via the existing normalize_sequence.
  - Mouse windows: fixed EVENT-COUNT windows (default 256 events / stride 128).
    Event-count windowing guarantees enough evidence per window -> removes the
    cold-start zeros that broke the previous 120s-based extraction. Idle is kept
    as a real behavior (is_idle channel + idle_ratio feature), not a cold start.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- keyboard / mouse behavioral feature names (aux targets) ----
KB_FEATURES = ["typing_speed", "mean_hold", "mean_ikl", "std_hold", "std_ikl",
               "pause_ratio", "unique_key_ratio"]
KB_AUX = ["typing_speed", "mean_hold", "mean_ikl", "std_hold", "std_ikl"]

MOUSE_FEATURES = ["speed_mean", "speed_std", "speed_max", "click_rate", "idle_ratio",
                  "n_events", "distance_total", "scroll_event_count"]
MOUSE_AUX = ["speed_mean", "click_rate", "idle_ratio", "n_events", "distance_total", "scroll_event_count"]

MOUSE_SEQ_CHANNELS = ["speed", "accel", "jerk", "dx", "dy", "is_idle", "is_click", "is_scroll"]
MOUSE_SEQ_LEN = 256


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def session_split(sessions: np.ndarray, val_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """80/20 split BY SESSION (no window leakage across train/val)."""
    uniq = sorted(set(sessions.tolist()))
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(uniq))
    n_val = max(1, int(round(len(perm) * val_frac)))
    val_sess = set(perm[:n_val])
    train_idx = np.array([i for i, s in enumerate(sessions) if s not in val_sess], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(sessions) if s in val_sess], dtype=np.int64)
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# NT-Xent (SimCLR) — real positive pairs
# --------------------------------------------------------------------------- #
class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy on two augmented views.

    z1, z2: (B, d) projections of view1/view2 of the SAME windows. Positives are
    (i in z1 <-> i in z2); all other 2B-2 samples are negatives.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.t = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        b = z1.shape[0]
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)      # (2B, d)
        sim = z @ z.t() / self.t                                # (2B, 2B)
        sim.fill_diagonal_(float("-inf"))
        targets = torch.arange(2 * b, device=z.device)
        targets = (targets + b) % (2 * b)                       # i <-> i+B
        return F.cross_entropy(sim, targets)


def contrastive_accuracy(z1: torch.Tensor, z2: torch.Tensor) -> float:
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t()
    sim.fill_diagonal_(float("-inf"))
    pred = sim.argmax(dim=1)
    targets = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
    return float((pred == targets).float().mean().item())


# --------------------------------------------------------------------------- #
# Robust scaler (median / IQR), fit on train aux targets
# --------------------------------------------------------------------------- #
class RobustScaler:
    """Median/IQR standardization with hard clipping to +/-clip.

    Clipping is essential: behavioral features (typing_speed, std_ikl, ...) are
    heavy-tailed; without it a few outliers blow up the MSE auxiliary loss and
    destabilize the stats-branch input.
    """

    def __init__(self, median=None, scale=None, clip: float = 5.0):
        self.median = median
        self.scale = scale
        self.clip = clip

    def fit(self, X: np.ndarray) -> "RobustScaler":
        self.median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale = np.where(iqr < 1e-8, 1.0, iqr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.median) / self.scale
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {"median": np.asarray(self.median).tolist(), "scale": np.asarray(self.scale).tolist(),
                "clip": float(self.clip)}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "RobustScaler":
        return cls(median=np.asarray(d["median"], np.float64), scale=np.asarray(d["scale"], np.float64),
                   clip=float(d.get("clip", 5.0)))


# --------------------------------------------------------------------------- #
# KEYBOARD windows
# --------------------------------------------------------------------------- #
def build_keyboard_windows(data_dir: Path, window_size: int = 20, stride: int = 10,
                           log_rows: Optional[List[Dict[str, Any]]] = None
                           ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return (X (N,W,3) normalized, F (N,7) behavioral feats, sessions, meta-array).

    Uses the existing parse_csv_events + normalize_sequence. Logs per-session
    issues into log_rows if provided.
    """
    from pre_embedders.keyboard import normalize_sequence, parse_csv_events

    X: List[np.ndarray] = []
    feats: List[List[float]] = []
    sessions: List[str] = []
    for kb in sorted(Path(data_dir).glob("session_*/raw/keyboard.csv")):
        sid = kb.parents[1].name
        try:
            rows = list(csv.DictReader(kb.open("r", encoding="utf-8", newline="")))
            events = parse_csv_events(rows)
        except Exception as exc:  # noqa: BLE001
            if log_rows is not None:
                log_rows.append({"session_id": sid, "status": "error", "detail": str(exc), "events": 0, "windows": 0})
            continue
        n_win = 0
        for start in range(0, len(events) - window_size + 1, stride):
            chunk = events[start:start + window_size]
            X.append(normalize_sequence(chunk))
            feats.append(_kb_features(chunk))
            sessions.append(sid)
            n_win += 1
        if log_rows is not None:
            log_rows.append({"session_id": sid, "status": "ok" if n_win else "too_few_events",
                             "detail": "", "events": len(events), "windows": n_win})
    Xa = np.asarray(X, dtype=np.float32) if X else np.zeros((0, window_size, 3), np.float32)
    Fa = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, len(KB_FEATURES)), np.float32)
    return Xa, Fa, sessions, np.arange(len(sessions))


def _kb_features(chunk: List[Dict[str, Any]]) -> List[float]:
    holds = np.array([e["hold"] for e in chunk], dtype=np.float64)
    ikls = np.array([e["ikl"] for e in chunk], dtype=np.float64)
    codes = [e["code"] for e in chunk]
    dur_s = max(np.sum(np.clip(ikls, 0, None)) / 1000.0, 1e-6)
    return [
        len(chunk) / dur_s,                                   # typing_speed (keys/s)
        float(np.mean(holds)),                                # mean_hold
        float(np.mean(ikls)),                                 # mean_ikl
        float(np.std(holds)),                                 # std_hold
        float(np.std(ikls)),                                  # std_ikl
        float(np.mean(ikls > 1000.0)),                        # pause_ratio
        len(set(codes)) / max(len(codes), 1),                 # unique_key_ratio
    ]


def augment_keyboard(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Augment a batch of keyboard windows (B,W,3): jitter hold/ikl, mask code, dropout events."""
    b, w, _ = x.shape
    out = x.clone()
    # jitter hold(0) & ikl(1)
    noise = torch.randn(b, w, 2, generator=rng, device=x.device) * 0.10
    out[:, :, :2] += noise
    # randomly mask code channel (2) for ~10% of events
    mask = (torch.rand(b, w, generator=rng, device=x.device) < 0.10)
    out[:, :, 2] = torch.where(mask, torch.zeros_like(out[:, :, 2]), out[:, :, 2])
    # event dropout: zero ~8% of timesteps (model still sees the slot)
    drop = (torch.rand(b, w, generator=rng, device=x.device) < 0.08).unsqueeze(-1)
    out = torch.where(drop, torch.zeros_like(out), out)
    return out


# --------------------------------------------------------------------------- #
# MOUSE windows (event-count based -> no cold start)
# --------------------------------------------------------------------------- #
def _fast_load_mouse(path: Path):
    from pre_embedders.mouse.mouse_encoder import MouseEvent

    def _f(v, d=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d
    events = []
    with path.open("r", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            ts = _f(row.get("timestamp"), None)
            if ts is None:
                continue
            btn = row.get("button")
            events.append(MouseEvent(
                timestamp=ts, x=_f(row.get("x")), y=_f(row.get("y")),
                dx=_f(row.get("delta_x")), dy=_f(row.get("delta_y")), speed=_f(row.get("speed")),
                event_type=str(row.get("event_type", "mouse_move")),
                button=btn if btn not in (None, "", "nan") else None,
            ))
    events.sort(key=lambda e: e.timestamp)
    return events


def _mouse_window_tensors(win_events, seq_len: int = MOUSE_SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Build (8, seq_len) sequence + behavioral feature vector for one window."""
    from pre_embedders.mouse.mouse_encoder import compute_per_event_derivatives
    d = compute_per_event_derivatives(win_events)
    n = len(win_events)
    speed = d["speed"].astype(np.float32)
    accel = np.abs(d["accel"]).astype(np.float32)
    jerk = np.abs(d["jerk"]).astype(np.float32)
    dx = np.array([e.dx for e in win_events], dtype=np.float32)
    dy = np.array([e.dy for e in win_events], dtype=np.float32)
    is_idle = d["is_idle"].astype(np.float32)
    is_click = np.array([1.0 if e.event_type == "mouse_press" else 0.0 for e in win_events], np.float32)
    is_scroll = np.array([1.0 if e.event_type == "scroll" else 0.0 for e in win_events], np.float32)
    seq = np.stack([speed, accel, jerk, dx, dy, is_idle, is_click, is_scroll], axis=0)  # (8, n)
    # per-channel robust scale within window (continuous channels only)
    for ch in range(5):
        m = np.abs(seq[ch]).max()
        if m > 1e-9:
            seq[ch] = seq[ch] / m
    # pad / truncate to seq_len
    if n < seq_len:
        seq = np.concatenate([seq, np.zeros((8, seq_len - n), np.float32)], axis=1)
    else:
        seq = seq[:, :seq_len]

    ts = d["ts"]
    dur = max(float(ts[-1] - ts[0]), 1e-6)
    dist = float(np.sum(np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)))
    feats = np.array([
        float(np.mean(d["speed"])),                              # speed_mean
        float(np.std(d["speed"])),                               # speed_std
        float(np.max(d["speed"])) if n else 0.0,                 # speed_max
        float(is_click.sum()) / dur,                             # click_rate
        float(np.mean(is_idle)),                                 # idle_ratio
        float(n),                                                # n_events
        dist,                                                    # distance_total
        float(is_scroll.sum()),                                  # scroll_event_count
    ], dtype=np.float32)
    return seq.astype(np.float32), feats


def build_mouse_windows(data_dir: Path, w_events: int = MOUSE_SEQ_LEN, stride: int = 128,
                        min_events: int = 32, max_sessions: Optional[int] = None,
                        log_rows: Optional[List[Dict[str, Any]]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return (SEQ (N,8,w_events), F (N,8) feats, sessions, meta).

    Event-count windows guarantee >= min_events evidence -> no cold start.
    """
    SEQ: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    sessions: List[str] = []
    n_sess = 0
    for mc in sorted(Path(data_dir).glob("session_*/raw/mouse.csv")):
        sid = mc.parents[1].name
        if max_sessions is not None and n_sess >= max_sessions:
            break
        try:
            events = _fast_load_mouse(mc)
        except Exception as exc:  # noqa: BLE001
            if log_rows is not None:
                log_rows.append({"session_id": sid, "status": "error", "detail": str(exc), "events": 0, "windows": 0})
            continue
        if len(events) < min_events:
            if log_rows is not None:
                log_rows.append({"session_id": sid, "status": "too_few_events", "detail": "",
                                 "events": len(events), "windows": 0})
            continue
        n_sess += 1
        n_win = 0
        starts = list(range(0, max(1, len(events) - w_events + 1), stride))
        if not starts:
            starts = [0]
        for s in starts:
            win = events[s:s + w_events]
            if len(win) < min_events:
                continue
            try:
                seq, fv = _mouse_window_tensors(win, w_events)
            except Exception:  # noqa: BLE001
                continue
            SEQ.append(seq)
            feats.append(fv)
            sessions.append(sid)
            n_win += 1
        if log_rows is not None:
            log_rows.append({"session_id": sid, "status": "ok" if n_win else "no_window", "detail": "",
                             "events": len(events), "windows": n_win})
    SEQa = np.asarray(SEQ, dtype=np.float32) if SEQ else np.zeros((0, 8, w_events), np.float32)
    Fa = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, len(MOUSE_FEATURES)), np.float32)
    return SEQa, Fa, sessions, np.arange(len(sessions))


def augment_mouse(seq: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Augment a batch of mouse sequences (B,8,T): jitter continuous channels,
    event dropout, temporal crop+repad. Never flips idle<->active wholesale."""
    b, c, t = seq.shape
    out = seq.clone()
    # jitter continuous channels 0..4 (speed,accel,jerk,dx,dy)
    out[:, :5, :] += torch.randn(b, 5, t, generator=rng, device=seq.device) * 0.08
    # event (timestep) dropout ~8%
    drop = (torch.rand(b, 1, t, generator=rng, device=seq.device) < 0.08)
    out = torch.where(drop, torch.zeros_like(out), out)
    # temporal crop: zero a random 0-15% contiguous tail/head segment
    if t > 16:
        crop = int(t * (0.0 + 0.15 * torch.rand(1, generator=rng).item()))
        if crop > 0:
            if torch.rand(1, generator=rng).item() < 0.5:
                out[:, :, :crop] = 0.0
            else:
                out[:, :, t - crop:] = 0.0
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
