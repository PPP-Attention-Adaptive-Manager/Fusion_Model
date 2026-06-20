"""
train_keyboard_encoder.py
==========================
Trains the keyboard pre-embedder (LSTM / BiLSTM) via contrastive learning.

Two modes:
  --mode observer   Uses your Observer CSV recordings
                     (timestamp, session_id, device_id, event_type, key, interval_ms, context)

  --mode 136m       Uses the 136M Keystrokes dataset (Dhakal et al., CHI 2018)
                     (PARTICIPANT_ID, TEST_SECTION_ID, PRESS_TIME, RELEASE_TIME, KEYCODE, ...)

Usage:
    cd ~/fusion_model

    # Observer mode (your own recorded data)
    python train_keyboard_encoder.py \
        --mode observer \
        --data_root "/home/hefouzinho/aam_data_6_1/Data/After update" \
        --epochs 10

    # 136M dataset mode
    python train_keyboard_encoder.py \
        --mode 136m \
        --data_root "/home/hefouzinho/Downloads/Keystrokes/files" \
        --max_users 5000 \
        --epochs 10

Output:
    pre_embedders/keyboard/weights/kb_encoder_lstm.pt    (or _bilstm.pt)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FUSION_DIR = Path(__file__).resolve().parent
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from pre_embedders.keyboard import KeystrokeEncoder, parse_csv_events, KeystrokeWindowDataset
from pre_embedders.keyboard.dataset import collate_windows


# ═════════════════════════════════════════════════════════════════════════════
# MODE: observer  —  your Record tool's CSV format
# ═════════════════════════════════════════════════════════════════════════════

def find_sessions(user_dir: Path) -> list[Path]:
    """Recursively find session_* dirs, same logic as prepare_tucker_inputs.py."""
    results = []
    for p in sorted(user_dir.rglob("session_*")):
        if not p.is_dir():
            continue
        rel_parts = p.relative_to(user_dir).parts
        if any(part.startswith("session_") for part in rel_parts[:-1]):
            continue
        results.append(p)
    return results


def load_observer_sessions(data_root: Path, window_size: int, stride: int,
                            users: list[str] | None = None):
    """
    Walk the Observer data folder, parse each session's keyboard.csv,
    return a list of KeystrokeWindowDataset (one per valid session).
    """
    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if users:
        user_dirs = [d for d in user_dirs if d.name in users]

    log.info("users found: %d", len(user_dirs))

    session_datasets = []
    total_windows = 0

    for user_dir in user_dirs:
        for session_path in find_sessions(user_dir):
            kb_csv = session_path / "raw" / "keyboard.csv"
            if not kb_csv.exists():
                continue
            try:
                rows   = pd.read_csv(kb_csv).to_dict("records")
                events = parse_csv_events(rows)
                if len(events) < window_size:
                    continue
                ds = KeystrokeWindowDataset(events, window_size, stride)
                if len(ds) == 0:
                    continue
                session_datasets.append(ds)
                total_windows += len(ds)
                log.info("  %s / %s  →  %d events  %d windows",
                          user_dir.name, session_path.name, len(events), len(ds))
            except Exception as e:
                log.debug("  skip %s: %s", session_path, e)

    log.info("total sessions: %d  total windows: %d",
              len(session_datasets), total_windows)
    return session_datasets


# ═════════════════════════════════════════════════════════════════════════════
# MODE: 136m  —  136M Keystrokes dataset (Dhakal et al., CHI 2018)
# ═════════════════════════════════════════════════════════════════════════════

def load_136m_file(filepath: Path) -> list[list[dict]]:
    """
    Load one user's <id>_keystrokes.txt file.

    Schema (tab-separated):
        PARTICIPANT_ID  TEST_SECTION_ID  SENTENCE  USER_INPUT
        KEYSTROKE_ID    PRESS_TIME       RELEASE_TIME
        LETTER          KEYCODE

    Each TEST_SECTION_ID is one typed sentence == one session boundary.
    IKL is computed manually (not precomputed in this dataset, unlike Observer CSV).

    Returns
    -------
    list of sessions, each session is a list of {code, hold, ikl} dicts.
    """
    try:
        df = pd.read_csv(
            filepath, sep="\t", encoding="utf-8",
            on_bad_lines="skip", engine="python",
        )
    except Exception as e:
        log.debug("  failed to read %s: %s", filepath, e)
        return []

    required = {"TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"}
    if not required.issubset(df.columns):
        return []

    sessions = []

    for _, group in df.groupby("TEST_SECTION_ID"):
        group = group.sort_values("PRESS_TIME")

        events = []
        prev_release = None

        for _, row in group.iterrows():
            try:
                press   = float(row["PRESS_TIME"])
                release = float(row["RELEASE_TIME"])
                code    = int(row["KEYCODE"]) % 256
            except (ValueError, TypeError):
                continue

            hold = release - press
            if hold < 0 or hold > 15000:      # malformed event, same threshold as Phase 1
                continue

            ikl = 0.0 if prev_release is None else max(0.0, press - prev_release)

            events.append({"code": code, "hold": hold, "ikl": ikl})
            prev_release = release

        if len(events) >= 5:        # keep even short sentences, window filter happens later
            sessions.append(events)

    return sessions


def load_136m_sessions(data_root: Path, window_size: int, stride: int,
                        max_users: int | None = None, seed: int = 42):
    """
    Walk the 136M dataset folder, parse each user's file, return a list of
    KeystrokeWindowDataset (one per valid sentence-session).
    """
    files = sorted(data_root.glob("*_keystrokes.txt"))
    log.info("found %d user files", len(files))

    if max_users is not None and max_users < len(files):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=max_users, replace=False)
        files = [files[i] for i in sorted(idx)]
        log.info("sampling %d users (seed=%d)", max_users, seed)

    session_datasets = []
    total_windows = 0

    for i, filepath in enumerate(files):
        sessions = load_136m_file(filepath)

        for events in sessions:
            if len(events) < window_size:
                continue
            ds = KeystrokeWindowDataset(events, window_size, stride)
            if len(ds) == 0:
                continue
            session_datasets.append(ds)
            total_windows += len(ds)

        if (i + 1) % 500 == 0:
            log.info("  processed %d/%d files  |  sessions so far: %d  windows: %d",
                      i + 1, len(files), len(session_datasets), total_windows)

    log.info("total sessions: %d  total windows: %d",
              len(session_datasets), total_windows)
    return session_datasets


# ═════════════════════════════════════════════════════════════════════════════
# Contrastive loss
# ═════════════════════════════════════════════════════════════════════════════

def contrastive_loss(z: torch.Tensor) -> torch.Tensor:
    z      = F.normalize(z, dim=1)
    sim    = torch.matmul(z, z.T)
    labels = torch.arange(len(z), device=z.device)
    return F.cross_entropy(sim, labels)


def contrastive_accuracy(z: torch.Tensor) -> float:
    z      = F.normalize(z, dim=1)
    sim    = torch.matmul(z, z.T)
    preds  = sim.argmax(dim=1)
    labels = torch.arange(len(z), device=z.device)
    return (preds == labels).float().mean().item()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
        choices=["observer", "136m"],
        help="observer = your Record tool CSVs | 136m = 136M Keystrokes dataset")
    parser.add_argument("--data_root", type=str, required=True,
        help="observer: user folders root | 136m: path to the 'files' folder")
    parser.add_argument("--output_dir", type=str,
        default=str(FUSION_DIR / "pre_embedders" / "keyboard" / "weights"))
    parser.add_argument("--users", nargs="*", default=None,
        help="[observer mode] restrict to specific user folder names")
    parser.add_argument("--max_users", type=int, default=None,
        help="[136m mode] randomly sample N users instead of using all 168k")
    parser.add_argument("--bidirectional", type=lambda x: x.lower() == "true",
        default=False, help="False = LSTM, True = BiLSTM")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride",      type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=3e-4)
    args = parser.parse_args()

    data_root  = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("mode      : %s", args.mode)
    log.info("data_root : %s", data_root)
    log.info("variant   : %s", "BiLSTM" if args.bidirectional else "LSTM")
    log.info("device    : %s", device)

    if not data_root.exists():
        log.error("data_root does not exist: %s", data_root)
        sys.exit(1)

    # ── load sessions based on mode ──────────────────────────────────────────
    if args.mode == "observer":
        session_datasets = load_observer_sessions(
            data_root, args.window_size, args.stride, users=args.users,
        )
    else:  # 136m
        session_datasets = load_136m_sessions(
            data_root, args.window_size, args.stride, max_users=args.max_users,
        )

    if not session_datasets:
        log.error("No valid sessions found. Check --data_root and --mode.")
        sys.exit(1)

    total_windows = sum(len(ds) for ds in session_datasets)

    batch_size = args.batch_size
    if total_windows < batch_size:
        batch_size = max(2, total_windows // 2)
        log.warning("Total windows (%d) < batch_size — reduced to %d",
                    total_windows, batch_size)

    # ── build dataset + loader ───────────────────────────────────────────────
    full_dataset = ConcatDataset(session_datasets)
    loader = DataLoader(
        full_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_windows,
        num_workers = 0,
    )

    # ── build model ───────────────────────────────────────────────────────────
    model = KeystrokeEncoder(
        input_size    = 3,
        hidden_size   = args.hidden_size,
        num_layers    = args.num_layers,
        bidirectional = args.bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── train ─────────────────────────────────────────────────────────────────
    log.info("Starting training  |  sessions=%d  windows=%d  batch_size=%d",
              len(session_datasets), total_windows, batch_size)

    for epoch in range(args.epochs):
        model.train()
        t_loss = t_acc = 0.0

        for batch in loader:
            batch = batch.to(device)
            z     = model(batch)
            loss  = contrastive_loss(z)
            acc   = contrastive_accuracy(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_acc  += acc

        log.info("Epoch %2d/%d  loss: %.4f  acc: %.4f",
                  epoch + 1, args.epochs, t_loss / len(loader), t_acc / len(loader))

    # ── save ──────────────────────────────────────────────────────────────────
    variant   = "bilstm" if args.bidirectional else "lstm"
    save_path = output_dir / f"kb_encoder_{variant}.pt"

    model.eval().cpu()
    torch.save(model.state_dict(), save_path)

    log.info("=" * 55)
    log.info("Weights saved → %s", save_path)
    log.info("Mode          : %s", args.mode)
    log.info("Sessions used : %d", len(session_datasets))
    log.info("Windows used  : %d", total_windows)
    log.info("=" * 55)


if __name__ == "__main__":
    main()