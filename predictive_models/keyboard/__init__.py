# predictive_models/keyboard/__init__.py

# GRU baseline
# from .v1_gru import KeyboardGRU as ActiveModel

# TCN
# from .v2_tcn import KeyboardTCN as ActiveModel

# Transformer
# from .v3_transformer import KeyboardTransformer as ActiveModel

# Hybrid (recommended final model)
from .v4_hybrid import KeyboardHybrid as ActiveModel