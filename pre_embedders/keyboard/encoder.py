import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# KEYSTROKE ENCODER  (LSTM and BiLSTM — single class, flag-switched)
# =============================================================

class KeystrokeEncoder(nn.Module):
    """
    Sliding-window LSTM / BiLSTM encoder for keyboard cognitive load.

    Accepts a batch of normalised keystroke windows of shape (B, W, 3)
    and returns one embedding vector per window of shape (B, embed_dim).

    Parameters
    ----------
    input_size    : features per keystroke event    — fixed at 3 (hold, ikl, code)
    hidden_size   : LSTM hidden units = embed_dim   — default 64
    num_layers    : stacked LSTM depth              — default 2
    bidirectional : False → LSTM  |  True → BiLSTM  — default False
    dropout       : dropout between LSTM layers     — default 0.2

    Output contract
    ---------------
    shape  : (B, hidden_size)  — same for LSTM and BiLSTM (projection aligns them)
    dtype  : torch.float32
    range  : unbounded (no activation at the end — raw embedding for downstream use)

    Integration note
    ----------------
    At inference time, squeeze batch dim and convert to numpy before pushing
    to the fusion queue:

        emb = encoder(window.unsqueeze(0))          # (1, 64)
        emb = emb.squeeze(0).detach().cpu().numpy() # (64,)  float32
        fusion_input.keyboard_queue.put(emb)
    """

    def __init__(
        self,
        input_size:    int  = 3,
        hidden_size:   int  = 64,
        num_layers:    int  = 2,
        bidirectional: bool = False,
        dropout:       float = 0.2,
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # project back to hidden_size so output shape is the same
        # regardless of bidirectionality
        self.projection = nn.Linear(
            hidden_size * self.num_directions,
            hidden_size,
        )

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, W, 3)
            Batch of normalised keystroke windows.

        Returns
        -------
        torch.Tensor  shape (B, hidden_size)
        """
        # x: (B, W, 3)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (B, W, hidden_size * num_directions)

        if self.bidirectional:
            # forward  final step : lstm_out[:, -1, :hidden_size]
            # backward final step : lstm_out[:,  0,  hidden_size:]
            fwd = lstm_out[:, -1, : self.hidden_size]
            bwd = lstm_out[:,  0,   self.hidden_size :]
            h   = torch.cat([fwd, bwd], dim=1)   # (B, 2 * hidden_size)
        else:
            h = lstm_out[:, -1, :]               # (B, hidden_size)

        out = self.projection(h)                 # (B, hidden_size)

        return out                               # (B, 64)


# =============================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================

def build_lstm_encoder(hidden_size: int = 64, num_layers: int = 2) -> KeystrokeEncoder:
    """Return an LSTM variant encoder."""
    return KeystrokeEncoder(
        input_size    = 3,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        bidirectional = False,
    )


def build_bilstm_encoder(hidden_size: int = 64, num_layers: int = 2) -> KeystrokeEncoder:
    """Return a BiLSTM variant encoder."""
    return KeystrokeEncoder(
        input_size    = 3,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        bidirectional = True,
    )