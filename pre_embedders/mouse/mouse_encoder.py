"""
AAM Mouse Encoder — Final Version
===================================
Self-contained. Replaces phase1_features.py + phase2_encoder.py entirely.

Architecture decisions locked:
  - Stats features : 22 (pruned from 33, see MouseWindowStats.feature_names())
  - TCN            : Option C — deterministic frozen filters
                     first block: principled filterbank (Gabor + gradient + peak + ...)
                     deeper blocks: near-identity frozen init
  - Output m_t     : (B, 64) torch.Tensor → Inférer fusion layer (Window K)
                     matches keyboard (64) and GNN (64) — mouse is not more
                     semantically complex than those modalities

Trainable components:
  - TCN.projection (Linear hidden→64)
  - TCN downsample 1×1 convs (channel mixing)
  - PreClickSubsequenceEncoder (full)
  - stats_mlp (full)
  - fusion_proj (full)

Frozen components:
  - All CausalConv1d weights and biases inside FrozenDeterministicBlock
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from typing import Optional
import threading
import queue
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

IDLE_SPEED_THRESHOLD  = 5.0    # px/s — below this = idle
PRE_CLICK_WINDOW_MS   = 200.0  # ms  — lookback for pre-click velocity
SPATIAL_GRID_CELLS    = 20     # NxN grid for spatial entropy
SAMPEN_M              = 2      # SampEn template length
SAMPEN_R_FACTOR       = 0.2    # SampEn tolerance = r * std(series)
MIN_EVENTS_FOR_WINDOW = 10     # discard windows with fewer events
TCN_INPUT_DIM         = 8      # channels in T×8 sequence
STATS_DIM             = 22     # handcrafted scalar feature dimension
TCN_HIDDEN            = 64     # TCN internal channel width
TCN_OUT_DIM           = 64     # TCN pooled output dim — matches keyboard/GNN
CLICK_DIM             = 32     # pre-click subsequence encoder output
STATS_MLP_HIDDEN      = 64     # stats MLP hidden width
FUSION_DIM            = 64     # m_t output dimension — matches keyboard/GNN


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class MouseEvent:
    timestamp:  float
    x:          float
    y:          float
    dx:         float
    dy:         float
    speed:      float          # px/s, pre-computed by libinput
    event_type: str            # mouse_move | mouse_press | mouse_release | scroll
    button:     Optional[str]  # left | right | middle | None


@dataclass
class MouseWindowStats:
    """
    22-feature handcrafted window descriptor.
    Pruned from 33 — removed statistically redundant features.
    """
    # Group A — kinematics (5)
    speed_mean:           float = 0.0
    speed_std:            float = 0.0
    speed_max:            float = 0.0   # kept: captures burst peaks lost in mean+std
    jerk_mean:            float = 0.0
    curvature_mean:       float = 0.0
    # Group A/B — angle (1)
    angle_delta_mean:     float = 0.0
    # Group B — trajectory geometry (3)
    path_efficiency:      float = 0.0   # B3: displacement / path_length
    direction_reversals:  int   = 0     # B5: sign flips in vx + vy
    sub_movement_count:   int   = 0     # B11: local minima in speed profile
    # Group B extended (1)
    convex_hull_area:     float = 0.0   # B6: spatial extent of activity
    # Group C — complexity (3)
    spatial_entropy:      float = 0.0   # C1: Shannon entropy over 20×20 position grid
    sample_entropy:       float = 0.0   # C4: speed signal unpredictability
    fractal_dimension:    float = 1.0   # C6: Higuchi FD — replaces skew+kurt+autocorr2,3
    # Group C extended (1)
    autocorr_lag1:        float = 0.0   # C5: serial dependence in speed (lag 1 only)
    # Group D — idle (2)
    idle_ratio:           float = 0.0   # D1: total_idle / window_duration
    mean_idle_duration:   float = 0.0   # D3: depth of disengagement
    # Group E — click dynamics (4)
    click_rate:           float = 0.0   # E1: clicks per second
    ici_cv:               float = 0.0   # E6: CoV of inter-click intervals
    click_hesitation:     float = 0.0   # E7: idle→move to click time
    post_click_pause:     float = 0.0   # E10: idle duration after click
    # Group F — scroll (2)
    scroll_reversal_count: int  = 0     # F2: direction reversals in scroll dy
    scroll_velocity:       float = 0.0  # F1: total |dy| / scroll duration

    def to_array(self) -> np.ndarray:
        return np.array([
            self.speed_mean,       self.speed_std,        self.speed_max,
            self.jerk_mean,        self.curvature_mean,   self.angle_delta_mean,
            self.path_efficiency,
            float(self.direction_reversals), float(self.sub_movement_count),
            self.convex_hull_area,
            self.spatial_entropy,  self.sample_entropy,   self.fractal_dimension,
            self.autocorr_lag1,
            self.idle_ratio,       self.mean_idle_duration,
            self.click_rate,       self.ici_cv,
            self.click_hesitation, self.post_click_pause,
            float(self.scroll_reversal_count), self.scroll_velocity,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "speed_mean",          "speed_std",           "speed_max",
            "jerk_mean",           "curvature_mean",      "angle_delta_mean",
            "path_efficiency",     "direction_reversals", "sub_movement_count",
            "convex_hull_area",
            "spatial_entropy",     "sample_entropy",      "fractal_dimension",
            "autocorr_lag1",
            "idle_ratio",          "mean_idle_duration",
            "click_rate",          "ici_cv",
            "click_hesitation",    "post_click_pause",
            "scroll_reversal_count", "scroll_velocity",
        ]


# ─────────────────────────────────────────────────────────────
# PER-EVENT DERIVATIVE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_per_event_derivatives(events: list[MouseEvent]) -> dict:
    """
    Compute per-event kinematic arrays from a window.
    Returns dict of np.ndarray, all length N (events count).
    """
    n     = len(events)
    ts    = np.array([e.timestamp for e in events], dtype=np.float64)
    dx    = np.array([e.dx        for e in events], dtype=np.float64)
    dy    = np.array([e.dy        for e in events], dtype=np.float64)
    speed = np.array([e.speed     for e in events], dtype=np.float64)

    dt = np.diff(ts)
    dt = np.where(dt < 1e-9, 1e-9, dt)   # guard div-by-zero

    # acceleration
    accel = np.zeros(n)
    accel[1:] = np.diff(speed) / dt

    # jerk
    jerk = np.zeros(n)
    jerk[2:] = np.diff(accel[1:]) / dt[1:]

    # angle delta (unwrapped)
    theta       = np.arctan2(dy, dx)
    angle_delta = np.zeros(n)
    angle_delta[1:] = np.abs(np.diff(np.unwrap(theta)))

    # angular velocity
    ang_vel = np.zeros(n)
    ang_vel[1:] = angle_delta[1:] / dt

    # velocity components for curvature and reversals
    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[1:] = dx[1:] / dt
    vy[1:] = dy[1:] / dt

    # curvature = |vx*ay - vy*ax| / speed³
    ax = np.zeros(n); ay = np.zeros(n)
    ax[1:] = np.diff(vx) / dt
    ay[1:] = np.diff(vy) / dt
    cross       = np.abs(vx * ay - vy * ax)
    speed_cubed = np.where(speed**3 < 1e-9, 1e-9, speed**3)
    curvature   = cross / speed_cubed

    is_idle = (speed < IDLE_SPEED_THRESHOLD).astype(np.float32)

    return {
        "ts": ts, "speed": speed, "accel": accel, "jerk": jerk,
        "curvature": curvature, "angle_delta": angle_delta,
        "ang_vel": ang_vel, "is_idle": is_idle,
        "vx": vx, "vy": vy, "dt": dt,
    }


# ─────────────────────────────────────────────────────────────
# SCALAR FEATURE HELPERS
# ─────────────────────────────────────────────────────────────

def _sample_entropy(series: np.ndarray, m: int, r: float) -> float:
    """O(n²) Sample Entropy. Run on window-level speed series."""
    n = len(series)
    if n < m + 2 or r <= 0:
        return 0.0

    def _count(length: int) -> int:
        count = 0
        for i in range(n - length):
            template = series[i:i + length]
            for j in range(i + 1, n - length + 1):
                if np.max(np.abs(template - series[j:j + length])) < r:
                    count += 1
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def _higuchi_fd(series: np.ndarray, k_max: int = 8) -> float:
    """
    Higuchi Fractal Dimension on a 1D series. FD in [1, 2].
    Validated in biosignal (EEG/EMG) literature as a workload correlate.
    """
    N = len(series)
    if N < k_max * 2:
        return 1.0
    L = []
    x = np.arange(1, k_max + 1)
    for k in x:
        Lk = 0.0
        for m in range(1, k + 1):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            Lm = np.sum(np.abs(np.diff(series[idxs])))
            Lm *= (N - 1) / (len(idxs) * k)
            Lk += Lm
        L.append(Lk / k)
    L = np.array(L)
    valid = L > 0
    if valid.sum() < 2:
        return 1.0
    slope, _ = np.polyfit(np.log(1.0 / x[valid]), np.log(L[valid]), 1)
    return float(np.clip(slope, 1.0, 2.0))


# ─────────────────────────────────────────────────────────────
# WINDOW FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_window_stats(events: list[MouseEvent]) -> Optional[MouseWindowStats]:
    """
    Extract all 22 handcrafted features from one window.
    Returns None if window is too small or degenerate.
    """
    if len(events) < MIN_EVENTS_FOR_WINDOW:
        logger.debug(f"Window too small ({len(events)} events), skipping.")
        return None
    window_duration = events[-1].timestamp - events[0].timestamp
    if window_duration < 1e-9:
        return None

    d       = compute_per_event_derivatives(events)
    ts      = d["ts"]
    speed   = d["speed"]
    jerk    = d["jerk"]
    curv    = d["curvature"]
    ang_d   = d["angle_delta"]
    is_idle = d["is_idle"]
    vx      = d["vx"]
    vy      = d["vy"]
    xs      = np.array([e.x for e in events])
    ys      = np.array([e.y for e in events])

    f = MouseWindowStats()

    # ── Group A — kinematics
    f.speed_mean       = float(np.mean(speed))
    f.speed_std        = float(np.std(speed))
    f.speed_max        = float(np.max(speed))
    f.jerk_mean        = float(np.mean(np.abs(jerk)))
    f.curvature_mean   = float(np.mean(curv))
    f.angle_delta_mean = float(np.mean(ang_d))

    # ── Group B — trajectory geometry
    diffs        = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    path_length  = float(np.sum(diffs))
    displacement = float(np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2))
    f.path_efficiency = (
        float(np.clip(displacement / path_length, 0.0, 1.0))
        if path_length > 1e-9 else 1.0
    )
    f.direction_reversals = (
        int(np.sum(np.diff(np.sign(vx)) != 0)) +
        int(np.sum(np.diff(np.sign(vy)) != 0))
    )
    f.sub_movement_count = int(np.sum(
        (speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:])
    ))

    # ── B6 — convex hull area
    pts = np.stack([xs, ys], axis=1)
    try:
        hull = ConvexHull(pts)
        f.convex_hull_area = float(hull.volume)   # volume = area in 2D
    except Exception:
        f.convex_hull_area = float(
            (xs.max() - xs.min()) * (ys.max() - ys.min())
        )

    # ── Group C — complexity
    H, _, _ = np.histogram2d(xs, ys, bins=SPATIAL_GRID_CELLS)
    H_flat  = H.flatten()
    H_flat  = H_flat[H_flat > 0]
    f.spatial_entropy   = float(scipy_entropy(H_flat / H_flat.sum()))
    f.sample_entropy    = _sample_entropy(
        speed, SAMPEN_M, SAMPEN_R_FACTOR * (np.std(speed) + 1e-9)
    )
    f.fractal_dimension = _higuchi_fd(speed)

    # C5 — autocorrelation at lag 1
    speed_c = speed - speed.mean()
    var     = np.var(speed)
    if len(speed) > 1 and var > 1e-12:
        f.autocorr_lag1 = float(np.mean(speed_c[:-1] * speed_c[1:]) / var)

    # ── Group D — idle
    idle_durations = []
    in_idle = False
    idle_start = 0.0
    for i, flag in enumerate(is_idle):
        if flag and not in_idle:
            in_idle = True
            idle_start = ts[i]
        elif not flag and in_idle:
            in_idle = False
            idle_durations.append(ts[i] - idle_start)
    if in_idle:
        idle_durations.append(ts[-1] - idle_start)
    f.idle_ratio         = float(sum(idle_durations) / window_duration)
    f.mean_idle_duration = float(np.mean(idle_durations)) if idle_durations else 0.0

    # ── Group E — click dynamics
    click_events = [
        (i, e) for i, e in enumerate(events)
        if e.event_type == "mouse_press" and e.button == "left"
    ]
    f.click_rate = len(click_events) / window_duration

    if len(click_events) >= 2:
        click_ts = np.array([ts[i] for i, _ in click_events])
        icis     = np.diff(click_ts)
        mu       = float(np.mean(icis))
        f.ici_cv = float(np.std(icis) / mu) if mu > 1e-9 else 0.0

    hesitations = []
    for idx, _ in click_events:
        for j in range(idx - 1, max(idx - 50, -1), -1):
            if is_idle[j] == 1.0:
                hesitations.append(ts[idx] - ts[j + 1] if j + 1 < idx else 0.0)
                break
    f.click_hesitation = float(np.mean(hesitations)) if hesitations else 0.0

    post_pauses = []
    for idx, _ in click_events:
        for j in range(idx + 1, min(idx + 100, len(events))):
            if is_idle[j] == 0.0:
                post_pauses.append(ts[j] - ts[idx])
                break
    f.post_click_pause = float(np.mean(post_pauses)) if post_pauses else 0.0

    # ── Group F — scroll
    scroll_evts = [(e.timestamp, e.dy) for e in events if e.event_type == "scroll"]
    if len(scroll_evts) >= 2:
        dy_signs = np.sign([dy for _, dy in scroll_evts])
        f.scroll_reversal_count = int(np.sum(np.diff(dy_signs) != 0))
        ts_sc  = np.array([t for t, _ in scroll_evts])
        dy_sc  = np.array([dy for _, dy in scroll_evts])
        dur_sc = ts_sc[-1] - ts_sc[0] + 1e-9
        f.scroll_velocity = float(np.sum(np.abs(dy_sc)) / dur_sc)

    return f


# ─────────────────────────────────────────────────────────────
# TCN SEQUENCE BUILDERS
# ─────────────────────────────────────────────────────────────

def build_tcn_sequence(
    events: list[MouseEvent],
    d:      dict,
    max_len: int = 512,
) -> torch.Tensor:
    """
    Build the T×8 input tensor for MouseTCNEncoder.
    Channel layout:
      0: speed      1: |accel|    2: |jerk|     3: ang_vel
      4: curvature  5: is_idle    6: angle_delta 7: is_click
    Returns (1, 8, T) — channels-first, batch=1.
    """
    is_click = np.array(
        [1.0 if e.event_type == "mouse_press" else 0.0 for e in events],
        dtype=np.float32,
    )
    seq = np.stack([
        d["speed"].astype(np.float32),
        np.abs(d["accel"]).astype(np.float32),
        np.abs(d["jerk"]).astype(np.float32),
        d["ang_vel"].astype(np.float32),
        d["curvature"].astype(np.float32),
        d["is_idle"].astype(np.float32),
        d["angle_delta"].astype(np.float32),
        is_click,
    ], axis=1)   # (T, 8)

    # per-channel max normalization within window
    for ch in range(seq.shape[1]):
        ch_max = seq[:, ch].max()
        if ch_max > 1e-9:
            seq[:, ch] /= ch_max

    # pad / truncate to max_len
    T = seq.shape[0]
    if T < max_len:
        seq = np.vstack([seq, np.zeros((max_len - T, TCN_INPUT_DIM), dtype=np.float32)])
    else:
        seq = seq[:max_len]

    return torch.tensor(seq.T, dtype=torch.float32).unsqueeze(0)   # (1, 8, T)


def build_pre_click_subsequence(
    events:      list[MouseEvent],
    d:           dict,
    subseq_len:  int = 20,
) -> Optional[torch.Tensor]:
    """
    E9: speed profile in PRE_CLICK_WINDOW_MS before each left-click.
    Returns (n_clicks, 1, subseq_len) or None if no left-clicks.
    """
    ts    = d["ts"]
    speed = d["speed"]
    pre_click_sec = PRE_CLICK_WINDOW_MS / 1000.0
    subsequences  = []

    for i, e in enumerate(events):
        if e.event_type != "mouse_press" or e.button != "left":
            continue
        mask = (ts >= ts[i] - pre_click_sec) & (ts < ts[i])
        sub  = speed[mask].astype(np.float32)
        if len(sub) < 3:
            continue
        # resample to fixed length
        sub = np.interp(
            np.linspace(0, 1, subseq_len),
            np.linspace(0, 1, len(sub)),
            sub,
        )
        s_max = sub.max()
        if s_max > 1e-9:
            sub /= s_max
        subsequences.append(sub)

    if not subsequences:
        return None
    return torch.tensor(
        np.stack(subsequences)[:, np.newaxis, :], dtype=torch.float32
    )   # (n_clicks, 1, subseq_len)


# ─────────────────────────────────────────────────────────────
# OPTION C — DETERMINISTIC FROZEN TCN
# ─────────────────────────────────────────────────────────────

def _gabor_kernel(f_hz: float, fs: float = 100.0, dilation: int = 1, k: int = 3) -> np.ndarray:
    t      = np.arange(k) * dilation / fs
    sigma  = (k * dilation) / (4.0 * fs)
    kernel = np.cos(2 * np.pi * f_hz * t) * np.exp(-t**2 / (2 * sigma**2))
    return kernel.astype(np.float32)


def _build_first_layer_filters(
    in_ch: int = 8,
    out_ch: int = 64,
    k: int = 3,
    fs: float = 100.0,
) -> torch.Tensor:
    """
    Build (out_ch, in_ch, k) weight tensor for the first TCN block.
    8 filter types × 8 input channels = 64 output filters.
    Each filter is applied to exactly one input channel (cross-channel = 0).

    Filter types:
      0 — causal gradient     : detects increase / acceleration onset
      1 — causal neg gradient : detects deceleration (pre-click ramp)
      2 — Laplacian           : sub-movement peak detector
      3 — inverted Laplacian  : trough / idle valley detector
      4 — smoothing           : low-pass noise suppression
      5 — causal identity     : raw passthrough
      6 — onset emphasis      : detects sudden onset from rest
      7 — Gabor 10Hz          : physiological tremor band
    """
    filters_per_ch = out_ch // in_ch   # must be 8

    templates = np.array([
        [0.0,  -1.0,   1.0],                       # 0: causal gradient
        [0.0,   1.0,  -1.0],                        # 1: causal neg gradient
        [-1.0,  2.0,  -1.0],                        # 2: Laplacian
        [1.0,  -2.0,   1.0],                        # 3: inverted Laplacian
        [0.25,  0.5,   0.25],                       # 4: smoothing
        [0.0,   0.0,   1.0],                        # 5: causal identity
        [-0.5, -0.5,   1.0],                        # 6: onset emphasis
        _gabor_kernel(10.0, fs=fs),                 # 7: 10Hz tremor
    ], dtype=np.float32)   # (8, 3)

    W = np.zeros((out_ch, in_ch, k), dtype=np.float32)
    for ch in range(in_ch):
        for fi, filt in enumerate(templates):
            W[ch * filters_per_ch + fi, ch, :] = filt

    # L2-normalize each filter
    norms = np.linalg.norm(W.reshape(out_ch, -1), axis=1, keepdims=True).reshape(out_ch, 1, 1)
    W /= np.where(norms < 1e-9, 1.0, norms)
    return torch.tensor(W)


def _build_deep_layer_filters(ch: int = 64, k: int = 3) -> torch.Tensor:
    """
    Near-identity init for deeper frozen blocks (in_ch == out_ch).
    Center tap = scaled identity matrix; side taps = 0.
    Residual connection carries the signal; conv adds marginal refinement.
    """
    W = torch.zeros(ch, ch, k)
    W[:, :, k // 2] = torch.eye(ch) * (1.0 / (ch ** 0.5))
    return W


class CausalConv1d(nn.Module):
    """Causal 1D conv — output at t uses only inputs ≤ t."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class FrozenDeterministicBlock(nn.Module):
    """
    TCN residual block with frozen conv weights (Option C).

    Frozen:    conv1, conv2 (CausalConv1d weights + biases)
    Trainable: downsample 1×1 conv when in_ch != out_ch
               (learns to mix filter responses into useful channels)
    """
    def __init__(
        self,
        in_ch:          int,
        out_ch:         int,
        kernel_size:    int,
        dilation:       int,
        is_first_block: bool  = False,
        fs:             float = 100.0,
    ):
        super().__init__()
        self.conv1     = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2     = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu      = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        with torch.no_grad():
            if is_first_block:
                self.conv1.conv.weight.copy_(
                    _build_first_layer_filters(in_ch, out_ch, kernel_size, fs)
                )
                self.conv2.conv.weight.copy_(
                    _build_deep_layer_filters(out_ch, kernel_size)
                )
            else:
                self.conv1.conv.weight.copy_(
                    _build_deep_layer_filters(out_ch, kernel_size)
                )
                self.conv2.conv.weight.copy_(
                    _build_deep_layer_filters(out_ch, kernel_size)
                )
            nn.init.zeros_(self.conv1.conv.bias)
            nn.init.zeros_(self.conv2.conv.bias)

        # freeze conv weights and biases
        for p in list(self.conv1.parameters()) + list(self.conv2.parameters()):
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return self.relu(out + residual)


class MouseTCNEncoder(nn.Module):
    """
    Input:  (B, 8, T)
    Output: (B, TCN_OUT_DIM=64)

    5 FrozenDeterministicBlocks with dilations [1, 2, 4, 8, 16].
    Receptive field ≈ 63 timesteps ≈ 630ms at 100Hz.
    Only the final Linear projection is trained.
    """
    def __init__(
        self,
        in_channels: int   = TCN_INPUT_DIM,
        hidden_dim:  int   = TCN_HIDDEN,
        out_dim:     int   = TCN_OUT_DIM,
        kernel_size: int   = 3,
        n_layers:    int   = 5,
        fs:          float = 100.0,
    ):
        super().__init__()
        dilations = [2 ** i for i in range(n_layers)]
        blocks = []
        ch = in_channels
        for idx, d in enumerate(dilations):
            blocks.append(FrozenDeterministicBlock(
                ch, hidden_dim, kernel_size, d,
                is_first_block=(idx == 0), fs=fs,
            ))
            ch = hidden_dim
        self.network    = nn.Sequential(*blocks)
        self.projection = nn.Linear(hidden_dim, out_dim)   # trainable
        self.out_dim    = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.network(x)                  # (B, hidden_dim, T)
        return self.projection(h.mean(dim=2))   # (B, out_dim)


class PreClickSubsequenceEncoder(nn.Module):
    """
    Small 1D CNN for E9 pre-click velocity profiles.
    Input:  (n_clicks, 1, subseq_len)
    Output: (1, click_dim) — mean across all clicks in window
    Fully trainable.
    """
    def __init__(self, subseq_len: int = 20, click_dim: int = CLICK_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, click_dim),
        )
        self.click_dim = click_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(0)   # (click_dim,)


# ─────────────────────────────────────────────────────────────
# MOUSE ENCODER P2 — FINAL OUTPUT CONTRACT
# ─────────────────────────────────────────────────────────────

class MouseEncoderP2(nn.Module):
    """
    Full mouse modality encoder.

    Input branches:
      seq           (1, 8, 512)         → MouseTCNEncoder  → (1, 64)
      pre_click_seq (n_clicks, 1, 20)   → ClickEncoder     → (1, 32)
      stats         (1, 22)             → stats_mlp        → (1, 64)

    Output:
      m_t  (B, 64) — dense embedding, matches keyboard (64) and GNN (64).
      Ready for Inférer Window-K fusion and Low-Rank Tucker (W_mouse: 64×R).

    Frozen:  all CausalConv1d inside FrozenDeterministicBlock
    Trained: TCN.projection, TCN downsample 1×1 convs,
             PreClickSubsequenceEncoder (full),
             stats_mlp (full), fusion_proj: Linear(160→64) + LN + ReLU
    """
    def __init__(
        self,
        stats_dim:   int = STATS_DIM,         # 22
        tcn_out_dim: int = TCN_OUT_DIM,       # 64
        click_dim:   int = CLICK_DIM,         # 32
        mlp_hidden:  int = STATS_MLP_HIDDEN,  # 64
        fusion_dim:  int = FUSION_DIM,        # 64
    ):
        super().__init__()
        self.tcn       = MouseTCNEncoder(out_dim=tcn_out_dim)
        self.click_enc = PreClickSubsequenceEncoder(click_dim=click_dim)
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        combined_dim = tcn_out_dim + click_dim + mlp_hidden   # 64+32+64 = 160
        self.fusion_proj = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.output_dim = fusion_dim

    def forward(
        self,
        seq:           torch.Tensor,                    # (B, 8, T)
        stats:         torch.Tensor,                    # (B, 22)
        pre_click_seq: Optional[torch.Tensor] = None,   # (n_clicks, 1, 20) | None
    ) -> torch.Tensor:
        tcn_out   = self.tcn(seq)                       # (B, 64)
        stats_out = self.stats_mlp(stats)               # (B, 64)

        if pre_click_seq is not None and pre_click_seq.shape[0] > 0:
            click_out = self.click_enc(pre_click_seq).unsqueeze(0)   # (1, 32)
        else:
            click_out = torch.zeros(
                seq.shape[0], self.click_enc.click_dim, device=seq.device
            )

        combined = torch.cat([tcn_out, click_out, stats_out], dim=1)   # (B, 160)
        return self.fusion_proj(combined)               # (B, 64)


# ─────────────────────────────────────────────────────────────
# PER-USER NORMALIZER
# ─────────────────────────────────────────────────────────────

class PerUserNormalizer:
    """
    Per-user z-score normalization.
    Non-negotiable: individual motor baselines vary significantly across subjects.
    Applies to the stats branch only (sequence branch uses within-window max norm).
    """
    def __init__(self):
        self._stats: dict[str, dict] = {}

    def fit(self, user_id: str, X: np.ndarray):
        """X: (n_windows, 22)"""
        self._stats[user_id] = {
            "mean": np.mean(X, axis=0),
            "std":  np.std(X,  axis=0) + 1e-9,
        }

    def transform(self, user_id: str, X: np.ndarray) -> np.ndarray:
        if user_id not in self._stats:
            raise ValueError(f"User '{user_id}' not fitted. Call fit() first.")
        s = self._stats[user_id]
        return (X - s["mean"]) / s["std"]

    def fit_transform(self, user_id: str, X: np.ndarray) -> np.ndarray:
        self.fit(user_id, X)
        return self.transform(user_id, X)


# ─────────────────────────────────────────────────────────────
# SLIDING WINDOW — RF BASELINE PATH
# ─────────────────────────────────────────────────────────────

class SlidingWindowExtractor:
    """
    Phase 1 / RF baseline path.
    Extracts (n_windows, 22) numpy matrix from a session event list.
    """
    def __init__(self, window_sec: float = 5.0, stride_sec: float = 1.0):
        self.window_sec = window_sec
        self.stride_sec = stride_sec

    def extract_stats(
        self, events: list[MouseEvent]
    ) -> tuple[np.ndarray, list[float]]:
        if not events:
            return np.empty((0, STATS_DIM)), []
        rows, starts = [], []
        t   = events[0].timestamp
        end = events[-1].timestamp
        while t + self.window_sec <= end:
            window = [e for e in events if t <= e.timestamp < t + self.window_sec]
            feat   = extract_window_stats(window)
            if feat is not None:
                rows.append(feat.to_array())
                starts.append(t)
            t += self.stride_sec
        if not rows:
            return np.empty((0, STATS_DIM)), []
        return np.vstack(rows), starts


# ─────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[MouseEvent]:
    """
    Load mouse.csv with expected columns:
      timestamp, event_type, x, y, delta_x, delta_y, speed, button
    """
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    events = []
    for _, row in df.iterrows():
        def _get(col: str, default: float = 0.0) -> float:
            v = row.get(col, default)
            return float(v) if pd.notna(v) else default
        events.append(MouseEvent(
            timestamp  = float(row["timestamp"]),
            x          = _get("x"),
            y          = _get("y"),
            dx         = _get("delta_x"),
            dy         = _get("delta_y"),
            speed      = _get("speed"),
            event_type = str(row.get("event_type", "mouse_move")),
            button     = str(row["button"]) if pd.notna(row.get("button")) else None,
        ))
    return events


# ─────────────────────────────────────────────────────────────
# LIVE LIBINPUT READER (Linux only)
# ─────────────────────────────────────────────────────────────

class LibinputLiveReader:
    """
    Background thread reading mouse events from libinput via udev.
    Requires membership in the 'input' group or root.
    """
    def __init__(self):
        self._queue:  queue.Queue             = queue.Queue()
        self._running: bool                  = False
        self._thread:  Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def flush(self) -> list[MouseEvent]:
        out = []
        while not self._queue.empty():
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def _read_loop(self):
        try:
            import libinput
        except ImportError:
            logger.error("python-libinput not installed. Run: pip install python-libinput")
            return

        li = libinput.LibInput(context_type=libinput.ContextType.UDEV)
        li.udev_assign_seat("seat0")
        prev_ts, prev_speed = None, 0.0

        while self._running:
            for event in li.get_events():
                if not self._running:
                    break
                et = event.type
                if et == libinput.EventType.POINTER_MOTION:
                    p  = event.get_pointer_event()
                    ts = event.time_usec / 1_000_000.0
                    dx, dy = p.get_dx(), p.get_dy()
                    dt = (ts - prev_ts) if prev_ts else 1e-6
                    speed = float(np.sqrt(dx**2 + dy**2) / max(dt, 1e-9))
                    prev_speed = speed; prev_ts = ts
                    self._queue.put(
                        MouseEvent(ts, 0.0, 0.0, dx, dy, speed, "mouse_move", None)
                    )
                elif et == libinput.EventType.POINTER_BUTTON:
                    p     = event.get_pointer_event()
                    ts    = event.time_usec / 1_000_000.0
                    state = "mouse_press" if p.get_button_state().value == 1 else "mouse_release"
                    btn   = {272: "left", 273: "right", 274: "middle"}.get(
                        p.get_button(), "unknown"
                    )
                    self._queue.put(
                        MouseEvent(ts, 0.0, 0.0, 0.0, 0.0, prev_speed, state, btn)
                    )
                elif et == libinput.EventType.POINTER_SCROLL_WHEEL:
                    p  = event.get_pointer_event()
                    ts = event.time_usec / 1_000_000.0
                    dy = p.get_scroll_value(libinput.PointerAxis.SCROLL_VERTICAL)
                    self._queue.put(
                        MouseEvent(ts, 0.0, 0.0, 0.0, dy, 0.0, "scroll", None)
                    )


# ─────────────────────────────────────────────────────────────
# FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

class Phase2Pipeline:
    """
    End-to-end session processor.
    Returns list of m_t tensors (1, 64), one per valid window.
    """
    def __init__(
        self,
        window_sec:  float = 5.0,
        stride_sec:  float = 1.0,
        max_seq_len: int   = 512,
        device:      str   = "cpu",
    ):
        self.window_sec  = window_sec
        self.stride_sec  = stride_sec
        self.max_seq_len = max_seq_len
        self.device      = torch.device(device)
        self.encoder     = MouseEncoderP2().to(self.device)
        self.normalizer  = PerUserNormalizer()
        self.encoder.eval()

    def process_session(
        self,
        events:         list[MouseEvent],
        user_id:        str,
        fit_normalizer: bool = False,
    ) -> list[torch.Tensor]:
        if not events:
            return []
        start_ts = events[0].timestamp
        end_ts   = events[-1].timestamp

        # first pass: fit normalizer on this session's stats
        if fit_normalizer:
            all_stats = []
            t = start_ts
            while t + self.window_sec <= end_ts:
                win  = [e for e in events if t <= e.timestamp < t + self.window_sec]
                feat = extract_window_stats(win)
                if feat is not None:
                    all_stats.append(feat.to_array())
                t += self.stride_sec
            if all_stats:
                self.normalizer.fit(user_id, np.vstack(all_stats))

        # second pass: encode
        outputs = []
        t = start_ts
        while t + self.window_sec <= end_ts:
            win = [e for e in events if t <= e.timestamp < t + self.window_sec]
            if len(win) < MIN_EVENTS_FOR_WINDOW:
                t += self.stride_sec; continue

            d    = compute_per_event_derivatives(win)
            seq  = build_tcn_sequence(win, d, self.max_seq_len).to(self.device)
            feat = extract_window_stats(win)
            if feat is None:
                t += self.stride_sec; continue

            arr = feat.to_array()[np.newaxis, :]
            if user_id in self.normalizer._stats:
                arr = self.normalizer.transform(user_id, arr)
            stats = torch.tensor(arr, dtype=torch.float32).to(self.device)

            pre_click = build_pre_click_subsequence(win, d)
            if pre_click is not None:
                pre_click = pre_click.to(self.device)

            with torch.no_grad():
                outputs.append(self.encoder(seq, stats, pre_click))
            t += self.stride_sec

        return outputs


# ─────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("── Architecture sanity check ──")
    enc    = MouseEncoderP2()
    seq    = torch.randn(1, TCN_INPUT_DIM, 512)
    stats  = torch.randn(1, STATS_DIM)
    clicks = torch.randn(3, 1, 20)
    out    = enc(seq, stats, clicks)
    assert out.shape == (1, FUSION_DIM), f"Bad output shape: {out.shape}"
    print(f"  Output shape  : {out.shape}   ✓  (expected (1, 64))")

    total   = sum(p.numel() for p in enc.parameters())
    frozen  = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    trained = total - frozen
    print(f"  Total params  : {total:,}")
    print(f"  Frozen params : {frozen:,}  ({100 * frozen / total:.1f}%)")
    print(f"  Trained params: {trained:,}  ({100 * trained / total:.1f}%)")
    print(f"\n  Feature vector ({STATS_DIM} features):")
    for i, name in enumerate(MouseWindowStats.feature_names()):
        print(f"    {i:2d}  {name}")

    if len(sys.argv) > 1:
        print(f"\n── CSV run: {sys.argv[1]} ──")
        events   = load_csv(sys.argv[1])
        print(f"  Events loaded : {len(events)}")
        pipeline = Phase2Pipeline()
        outputs  = pipeline.process_session(events, "test_user", fit_normalizer=True)
        print(f"  Windows encoded: {len(outputs)}")
        if outputs:
            print(f"  m_t shape      : {outputs[0].shape}")