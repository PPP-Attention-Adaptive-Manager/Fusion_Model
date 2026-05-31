# Switching Identity Integration Report

## 1. Files changed

- `fusion_model.py`
  - Added `DEFAULT_D_DIMS = [64, 64, 32, 64]`.
  - Updated `InferrerFusion` default dimensions to use that contract.

- `TCN_encoders/switching/encoder.py`
  - Added `SwitchingIdentityEncoder` as the default 64D path.
  - Kept the old random-frozen TCN path as `SwitchingRandomFrozenTCNEncoder`.
  - Added `SwitchingBufferedEncoder` compatibility factory.
  - Added mode config via `SWITCHING_ENCODER_MODE`.

- `TCN_encoders/switching/__init__.py`
  - Exported identity, random-frozen, mode constants, and builder helpers.

- `pre_embedders/switching/encoder.py`
  - Provides the GraphSAGE encoder-only wrapper.
  - Loads the exported encoder once.
  - Preserves metadata and validates the 64D float32 output contract.

- `pre_embedders/switching/__init__.py`
  - Exposes `load_model`, `get_output`, and switching validation helpers.

- `tests/test_switching_encoder_integration.py`
  - Added tests for identity shape/value preservation, cold start, fusion dims,
    default mode, and random-frozen ablation availability.

- `README.md`
  - Updated switching documentation to identity-only 64D.
  - Updated default `d_dims` to `[64, 64, 32, 64]`.

- `Architecture.md`
  - Updated diagrams and contracts to show the new 64D identity switching path.

- `AAM_TEAMMATE_INTEGRATION_GUIDE2.md`
  - Updated default dimension references to `[64, 64, 32, 64]`.

## 2. Old path

```text
GraphSAGE-GAE encoder output
-> np.ndarray shape (64,), float32
-> SwitchingBufferedEncoder
-> shallow random-frozen TCN
-> torch.Tensor shape (1,32)
-> fusion switching modality
```

This path was kept only because the old fusion contract used:

```python
d_dims = [64, 64, 32, 32]
```

## 3. New default path

```text
graph_json_or_path
-> switching_encoder/output.py
-> GraphSAGE encoder only
-> global_mean_pool
-> L2 normalization
-> np.ndarray shape (64,), float32
-> SwitchingIdentityEncoder
-> torch.Tensor shape (1,64)
-> fusion switching modality
```

The decoder is not loaded or called by the fusion integration.

## 4. Updated d_dims

The default fusion dimensions are now:

```python
d_dims = [64, 64, 32, 64]
#        mouse keyboard notif switching
```

With the default Tucker rank `R=8`, each predictive model still receives
`R ** 3 = 512` features from the Tucker slice.

## 5. Why identity-only is used

The GraphSAGE-GAE encoder already learns a behavioral graph representation over
a 120-second window. Since the switching signal is slow and already summarized
by the trained encoder, a random-frozen TCN can distort the learned embedding
without a clear scientific reason.

The default switching encoder now preserves the learned 64D embedding. It only
manages:

- shape conversion to `(1,64)`
- last-valid embedding buffering
- cold start zeros
- freshness/staleness decay
- metadata preservation

## 6. How to switch back to random-frozen TCN ablation

Direct instantiation:

```python
from TCN_encoders.switching.encoder import SwitchingRandomFrozenTCNEncoder

switching_enc = SwitchingRandomFrozenTCNEncoder()
```

Compatibility factory:

```python
from TCN_encoders.switching.encoder import SwitchingBufferedEncoder

switching_enc = SwitchingBufferedEncoder(mode="random_frozen_tcn")
```

Environment variable:

```text
SWITCHING_ENCODER_MODE=random_frozen_tcn
```

Default mode:

```text
SWITCHING_ENCODER_MODE=identity
```

## 7. Test results

Passed:

```text
python -m unittest tests.test_switching_encoder_integration
Ran 8 tests
OK
```

Passed:

```text
python -m unittest discover -s tests
Ran 8 tests
OK
```

Checked:

```text
python -c "from TCN_encoders.switching.encoder import SwitchingBufferedEncoder; e=SwitchingBufferedEncoder(); print(type(e).__name__, e.output_dim)"
SwitchingIdentityEncoder 64
```

Not run:

```text
python test_fusion.py
```

Reason: `test_fusion.py` is not present in the project root.

Runtime note: PyTorch Geometric emitted a deprecation warning for
`torch_geometric.distributed`; this did not affect the tests.
