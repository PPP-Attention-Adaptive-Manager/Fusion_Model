from .fixed_tcn         import FixedTCNEncoder
from .configs           import TCNEncoderConfig, multiscale, narrow, shallow, wide
from .mouse.encoder     import MouseBufferedEncoder
from .keyboard.encoder  import KeyboardBufferedEncoder
from .notif.encoder     import NotifBufferedEncoder
from .switching.encoder import SwitchingBufferedEncoder

MODALITY_ENCODERS = [
    MouseBufferedEncoder,
    KeyboardBufferedEncoder,
    NotifBufferedEncoder,
    SwitchingBufferedEncoder,
]

__all__ = [
    "FixedTCNEncoder", "TCNEncoderConfig",
    "multiscale", "narrow", "shallow", "wide",
    "MODALITY_ENCODERS",
]