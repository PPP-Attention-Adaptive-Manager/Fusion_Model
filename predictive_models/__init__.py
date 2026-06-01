"""Active predictive-model registry for InferrerFusion.

Order must match `InferrerFusion.MODALITY_NAMES`:
mouse, keyboard, notif, switching.
"""

from predictive_models.keyboard import ActiveModel as KeyboardModel
from predictive_models.mouse import ActiveModel as MouseModel
from predictive_models.notif import ActiveModel as NotifModel
from predictive_models.switching import ActiveModel as SwitchingModel


MODALITY_MODELS = [
    MouseModel,
    KeyboardModel,
    NotifModel,
    SwitchingModel,
]

__all__ = ["MODALITY_MODELS"]
