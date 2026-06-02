# Switching Encoder Export

This package contains the encoder-only export of the trained switching / behavior graph module.

The original model was trained as a Graph AutoEncoder:

- GraphSAGE encoder
- Decoder for adjacency reconstruction during training

For fusion-time inference, only the GraphSAGE encoder is loaded. The decoder is not exported in `encoder.pt` and is not called by `output.py`.

## Files

- `encoder.pt`: GraphSAGE encoder `state_dict` only
- `feature_spec.json`: exact preprocessing schema from training
- `model_config.json`: architecture and training metadata needed to rebuild the encoder
- `embedding_contract.json`: output contract for the fusion model
- `output.py`: encoder-only runtime API
- `test_export.py`: smoke test

## Input

The input is one behavioral graph JSON for a 120 second window. It may be passed as a path or as a parsed dictionary.

```python
from output import load_model, get_output

session = load_model(
    export_dir="gnn_project/exports/switching_encoder",
    device="auto",
)

result = get_output(session, "gnn_project/data/.../data_graph_120s/graph_002.json")
```

## Output

`get_output(...)` returns:

```python
{
    "embedding": np.ndarray,  # shape (64,), dtype float32, L2-normalized
    "metadata": {
        "module": "switching",
        "encoder": "GraphSAGE_GAE",
        "embedding_dim": 64,
        "window_size_s": 120,
        "normalized": "l2",
        "graph_id": str | None,
        "session_id": str | None,
        "window_id": str | None,
    },
}
```

## Cold Start

If the graph is empty or invalid, the module returns a zero vector:

```python
{
    "embedding": np.zeros(64, dtype=np.float32),
    "metadata": {
        "module": "switching",
        "cold_start": True,
        "reason": "empty_or_invalid_graph",
    },
}
```

## Smoke Test

From the project root:

```powershell
python gnn_project/exports/switching_encoder/test_export.py --device auto
```
