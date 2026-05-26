"""
keyboard pre-embedder
---------------------
Drop this folder into:
    fusion_model/pre_embedders/keyboard/

Public API
----------
    KeystrokeEncoder          — LSTM / BiLSTM encoder (flag-switched)
    build_lstm_encoder()      — convenience constructor, LSTM variant
    build_bilstm_encoder()    — convenience constructor, BiLSTM variant
    StreamingWindowBuffer     — online inference buffer (push one event at a time)
    KeystrokeWindowDataset    — offline sliding-window dataset
    parse_csv_events()        — parse raw CSV rows into {code, hold, ikl} dicts
    normalize_sequence()      — normalise a list of events to (N, 3) float32 array
    train()                   — train encoder with contrastive loss
    extract_embeddings()      — batch-extract embeddings from a trained encoder

Output contract (fusion-facing)
--------------------------------
    shape  : (64,)  np.float32
    push   : fusion_input.keyboard_queue.put(embedding)
    cadence: one vector per S=10 completed keystrokes (~2–3 s at average typing speed)
"""

from .encoder  import KeystrokeEncoder, build_lstm_encoder, build_bilstm_encoder
from .dataset  import KeystrokeWindowDataset, StreamingWindowBuffer, collate_windows
from .preprocess import encode_key, normalize_sequence, parse_csv_events
from .train    import train, extract_embeddings

__all__ = [
    "KeystrokeEncoder",
    "build_lstm_encoder",
    "build_bilstm_encoder",
    "KeystrokeWindowDataset",
    "StreamingWindowBuffer",
    "collate_windows",
    "encode_key",
    "normalize_sequence",
    "parse_csv_events",
    "train",
    "extract_embeddings",
]