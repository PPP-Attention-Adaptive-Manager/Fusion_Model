"""
TCNEncoderConfig — configuration dataclass + named presets.

One preset = one filter bank experiment. To ablate:
    1. Add a new preset here.
    2. Point the target modality's __init__.py at it.
    3. Re-run training, compare LOSO accuracy.

Receptive field formula (for reference):
    RF = 1 + (kernel_size - 1) × (2^n_layers - 1)

Preset summary:
    multiscale  k=3, L=8  → RF=511   mouse (fast jerk + slow idle rhythm)
    narrow      k=3, L=4  → RF=31    keyboard (IKI bursts at 100–500ms)
    wide        k=7, L=4  → RF=211   alternative mouse experiment
    shallow     k=3, L=2  → RF=7     notif / switching (sparse signals)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TCNEncoderConfig:
    n_channels  : int             # channel width of every TemporalBlock
    kernel_size : int             # conv kernel size (same across all blocks)
    n_layers    : int             # number of TemporalBlocks (dilation doubles each layer)
    dilation_base: int            # base for exponential dilation schedule (almost always 2)
    dropout     : float           # dropout inside each TemporalBlock
    pool        : str             # "last" | "mean" | "max"  — how to collapse T → 1
    filter_init  : str = "random" # valid values: "random" | "temporal_bank_full" | "temporal_bank_half" | "identity"
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (self.dilation_base ** self.n_layers - 1)


# ---------------------------------------------------------------------------
# Named presets — import these in per-modality __init__.py files
# ---------------------------------------------------------------------------

multiscale = TCNEncoderConfig(
    n_channels   = 64,
    kernel_size  = 3,
    n_layers     = 8,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    # RF = 511 timesteps — covers fast jerk (~50ms) and slow idle drift (~5s at 100Hz)
)

narrow = TCNEncoderConfig(
    n_channels   = 64,
    kernel_size  = 3,
    n_layers     = 4,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    # RF = 31 timesteps — keyboard IKI patterns at 100–500ms scale
)

wide = TCNEncoderConfig(
    n_channels   = 64,
    kernel_size  = 7,
    n_layers     = 4,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    # RF = 211 timesteps — wider kernel experiment, alternative for mouse
)

shallow = TCNEncoderConfig(
    n_channels   = 32,
    kernel_size  = 3,
    n_layers     = 2,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    # RF = 7 timesteps — sparse event signals (notif, switching)
)


shallow_notif = TCNEncoderConfig(
    n_channels   = 32,
    kernel_size  = 3,
    n_layers     = 2,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    filter_init  = "temporal_bank_half",
    # notif: EMA (half) + identity (half)
    # difference excluded — fires on padding boundary artifact, not real signal
    # updates every 5s so half the buffer is zeros between updates
)

shallow_switching = TCNEncoderConfig(
    n_channels   = 32,
    kernel_size  = 3,
    n_layers     = 2,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    filter_init  = "identity",
    # switching: pure causal identity
    # buffer is sparse (real data every 120s, rest is zero-padded)
    # difference fires on zero-pad boundary = artifact not signal
    # EMA at last position mixes old/new — valid but identity is cleaner baseline
    # staleness/freshness mechanism in BufferedEncoder handles temporal weighting
)

# ALSO update narrow preset to add filter_init:
narrow = TCNEncoderConfig(
    n_channels   = 64,
    kernel_size  = 3,
    n_layers     = 4,
    dilation_base= 2,
    dropout      = 0.1,
    pool         = "last",
    filter_init  = "temporal_bank_full",
    # keyboard: EMA (1/3) + difference (1/3) + identity (1/3)
    # LSTM embeddings arrive every ~2-3s, rich temporal dynamics
    # block assignment: dims 0..21 EMA, 22..42 difference, 43..63 identity
    # assignment is arbitrary (LSTM dims are non-interpretable) — cyclic equiv
)