# Switching Decoder Debug Export

This folder contains the decoder weights extracted from the trained Graph AutoEncoder checkpoint.

This is a QA/debug artifact only.

The fusion model must not use this decoder. Fusion inference uses only:

- `gnn_project/exports/switching_encoder/encoder.pt`
- `gnn_project/exports/switching_encoder/output.py`

The decoder is kept here so we can recombine:

```text
encoder.pt + decoder.pt
```

and verify that the separated files reproduce the full `best.pt` checkpoint.
