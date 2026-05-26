import numpy as np

# =============================================================
# KEY ENCODING
# =============================================================

SPECIAL_KEY_MAP = {
    "space":     32,
    "enter":     13,
    "backspace":  8,
    "shift":     16,
    "ctrl":      17,
    "alt":       18,
    "tab":        9,
    "escape":    27,
    "delete":    46,
    "caps_lock": 20,
    "up":        38,
    "down":      40,
    "left":      37,
    "right":     39,
}


def encode_key(key: str) -> int:
    """
    Map a key string to an integer code 0–255.

    Priority:
      1. Special key map (space, enter, backspace, etc.)
      2. Single printable character → ord(char) % 256
      3. Unknown multi-char special key → 0 (fallback)
    """
    key = key.strip().lower()

    if key in SPECIAL_KEY_MAP:
        return SPECIAL_KEY_MAP[key]

    if len(key) == 1:
        return ord(key) % 256

    return 0


# =============================================================
# SEQUENCE NORMALISATION
# =============================================================

def normalize_sequence(seq: list[dict]) -> np.ndarray:
    """
    Convert a list of raw keystroke dicts to a normalised (N, 3) float32 array.

    Each dict must have:
        code  : int   key code 0–255
        hold  : float hold duration in ms
        ikl   : float inter-keystroke latency in ms

    Normalisation:
        hold  → log1p → z-score (zero-mean, unit-std)
        ikl   → log1p → z-score
        code  → divide by 255  (range [0, 1])

    Returns
    -------
    np.ndarray shape (N, 3) dtype float32
        columns: [hold_norm, ikl_norm, code_norm]
    """
    holds = np.array([e["hold"] for e in seq], dtype=np.float32)
    ikls  = np.array([e["ikl"]  for e in seq], dtype=np.float32)
    codes = np.array([e["code"] for e in seq], dtype=np.float32)

    holds = np.nan_to_num(holds)
    ikls  = np.nan_to_num(ikls)

    # log-compress to reduce extreme outlier impact
    holds = np.log1p(np.clip(holds, 0, 10_000))
    ikls  = np.log1p(np.clip(ikls,  0, 20_000))

    def z_score(x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-6:
            return np.zeros_like(x)
        return (x - x.mean()) / std

    holds = z_score(holds)
    ikls  = z_score(ikls)
    codes = codes / 255.0

    return np.stack([holds, ikls, codes], axis=1).astype(np.float32)


# =============================================================
# CSV ROW PARSER
# =============================================================

def parse_csv_events(rows: list[dict]) -> list[dict]:
    """
    Convert raw CSV rows (as dicts) into the internal keystroke event format
    expected by normalize_sequence().

    CSV schema (per row):
        timestamp   : float   unix timestamp in seconds
        event_type  : str     "key_press" | "key_release"
        key         : str     key name string
        interval_ms : float   IKL (already computed by Observer, 0.0 on releases)

    Logic:
        - Pair each key_press with its next matching key_release by key name
        - hold = (release.timestamp − press.timestamp) × 1000  (ms)
        - ikl  = interval_ms from the key_press row directly

    Returns
    -------
    list of dicts with keys: code, hold, ikl
    Sorted by press timestamp. Skips presses with no matching release.
    """
    press_rows    = [r for r in rows if r["event_type"] == "key_press"]
    release_rows  = [r for r in rows if r["event_type"] == "key_release"]

    # index releases by key name for fast lookup
    release_index: dict[str, list[float]] = {}
    for r in release_rows:
        key = r["key"].strip().lower()
        release_index.setdefault(key, []).append(float(r["timestamp"]))

    # sort release timestamps ascending so we always grab the first one after press
    for key in release_index:
        release_index[key].sort()

    events = []

    for pr in press_rows:
        key        = pr["key"].strip().lower()
        press_ts   = float(pr["timestamp"])
        ikl        = float(pr.get("interval_ms", 0.0))
        code       = encode_key(key)

        # find the first release timestamp that comes after this press
        candidates = release_index.get(key, [])
        release_ts = None
        for ts in candidates:
            if ts >= press_ts:
                release_ts = ts
                candidates.remove(ts)   # consume so it won't be reused
                break

        if release_ts is None:
            # no matching release found — skip this keystroke
            continue

        hold = (release_ts - press_ts) * 1000.0   # seconds → ms

        if hold < 0:
            continue   # malformed event

        events.append({
            "code": code,
            "hold": hold,
            "ikl":  ikl,
        })

    events.sort(key=lambda e: e["ikl"])   # preserve original press order (ikl is cumulative)

    return events