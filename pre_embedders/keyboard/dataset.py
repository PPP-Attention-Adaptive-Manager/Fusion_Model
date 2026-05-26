import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import normalize_sequence


# =============================================================
# SLIDING WINDOW DATASET
# =============================================================

class KeystrokeWindowDataset(Dataset):
    """
    Splits a session's keystroke event list into overlapping windows
    and returns normalised (W, 3) tensors ready for the encoder.

    Parameters
    ----------
    events      : list of dicts with keys: code, hold, ikl
    window_size : W — number of keystrokes per window
    stride      : S — step between consecutive windows  (S < W gives overlap)

    Item shape  : torch.Tensor (W, 3)  dtype float32
    """

    def __init__(
        self,
        events:      list[dict],
        window_size: int = 20,
        stride:      int = 10,
    ):
        self.window_size = window_size
        self.stride      = stride
        self.windows     = self._build_windows(events)

    # ----------------------------------------------------------
    def _build_windows(self, events: list[dict]) -> list[np.ndarray]:
        """
        Slide over the event list and materialise each window as a
        normalised numpy array of shape (W, 3).
        """
        windows = []

        n = len(events)

        for start in range(0, n - self.window_size + 1, self.stride):
            chunk = events[start : start + self.window_size]
            arr   = normalize_sequence(chunk)   # (W, 3) float32
            windows.append(arr)

        return windows

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# =============================================================
# COLLATE FUNCTION
# (all windows in a batch are the same size W — no padding needed)
# =============================================================

def collate_windows(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of (W, 3) tensors into a single (B, W, 3) batch tensor.
    Since every window has the same fixed size W, no padding is required.
    """
    return torch.stack(batch, dim=0)   # (B, W, 3)


# =============================================================
# STREAMING WINDOW BUFFER
# (for real-time / online inference — no Dataset needed)
# =============================================================

class StreamingWindowBuffer:
    """
    Accumulates incoming keystroke events one by one and yields a
    normalised window tensor whenever enough events have been collected.

    Use this at inference time instead of KeystrokeWindowDataset.

    Parameters
    ----------
    window_size : W  — keystrokes per emission
    stride      : S  — keystrokes between consecutive emissions

    Usage
    -----
        buf = StreamingWindowBuffer(window_size=20, stride=10)

        for event in live_event_stream:
            result = buf.push(event)
            if result is not None:
                embedding = encoder(result.unsqueeze(0))   # (1, 64)
                queue.put(embedding.squeeze(0).detach().cpu().numpy())
    """

    def __init__(self, window_size: int = 20, stride: int = 10):
        self.window_size = window_size
        self.stride      = stride
        self._buffer:    list[dict] = []
        self._since_last_emit: int  = 0

    # ----------------------------------------------------------
    def push(self, event: dict) -> torch.Tensor | None:
        """
        Add one keystroke event to the internal buffer.

        Returns a (W, 3) float32 tensor when a new window is ready,
        None otherwise.

        event must have keys: code (int), hold (float ms), ikl (float ms)
        """
        self._buffer.append(event)
        self._since_last_emit += 1

        if len(self._buffer) < self.window_size:
            return None

        if self._since_last_emit < self.stride:
            return None

        # take the most recent W events as the window
        window_events = self._buffer[-self.window_size :]
        arr           = normalize_sequence(window_events)   # (W, 3)
        self._since_last_emit = 0

        return torch.tensor(arr, dtype=torch.float32)       # (W, 3)

    # ----------------------------------------------------------
    def reset(self):
        """Clear buffer — call at session boundaries."""
        self._buffer          = []
        self._since_last_emit = 0