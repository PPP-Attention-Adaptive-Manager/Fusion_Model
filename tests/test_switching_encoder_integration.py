"""Smoke tests for the switching GraphSAGE export integration."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from pre_embedders.switching import load_model, validate_switching_output
from TFN import ActiveTFN
from TCN_encoders.switching.encoder import (
    SwitchingBufferedEncoder,
    SwitchingIdentityEncoder,
    SwitchingRandomFrozenTCNEncoder,
)


class SwitchingEncoderIntegrationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = load_model(device="cpu")

    def test_export_output_contract_on_cold_start(self) -> None:
        result = self.session.get_output(
            {
                "graph_id": "cold_start_smoke",
                "session_id": "test_session",
                "window": {"window_id": "window_000"},
                "nodes": [],
                "edges": [],
            }
        )
        result = validate_switching_output(result)

        embedding = result["embedding"]
        metadata = result["metadata"]

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (64,))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertFalse(np.isnan(embedding).any())
        self.assertTrue(np.isfinite(embedding).all())
        self.assertTrue(metadata.get("cold_start", False))
        self.assertEqual(float(np.linalg.norm(embedding)), 0.0)

    def test_validator_accepts_l2_normalized_non_cold_start(self) -> None:
        embedding = np.ones(64, dtype=np.float32)
        embedding = (embedding / np.linalg.norm(embedding)).astype(np.float32)

        result = validate_switching_output(
            {
                "embedding": embedding,
                "metadata": {
                    "module": "switching",
                    "encoder": "GraphSAGE_GAE",
                    "embedding_dim": 64,
                    "cold_start": False,
                    "decoder_used_at_inference": False,
                },
            }
        )

        self.assertEqual(result["embedding"].shape, (64,))
        self.assertEqual(result["embedding"].dtype, np.float32)
        self.assertFalse(np.isnan(result["embedding"]).any())
        self.assertAlmostEqual(float(np.linalg.norm(result["embedding"])), 1.0, places=5)

    def test_switching_identity_encoder_shape(self) -> None:
        raw_embedding = np.random.randn(64).astype(np.float32)
        encoder = SwitchingIdentityEncoder()

        embedding, freshness = encoder.step({"embedding": raw_embedding})

        self.assertEqual(tuple(embedding.shape), (1, 64))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(torch.allclose(embedding, torch.from_numpy(raw_embedding).unsqueeze(0)))
        self.assertEqual(freshness, 1.0)

    def test_switching_identity_cold_start(self) -> None:
        encoder = SwitchingIdentityEncoder()

        embedding, freshness = encoder.step(None)

        self.assertEqual(tuple(embedding.shape), (1, 64))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(torch.equal(embedding, torch.zeros(1, 64)))
        self.assertEqual(freshness, 0.0)

    def test_fusion_dims(self) -> None:
        fusion_model_text = (Path(__file__).resolve().parents[1] / "fusion_model.py").read_text()
        self.assertIn("DEFAULT_D_DIMS = [64, 64, 32, 64]", fusion_model_text)

    def test_tucker_accepts_updated_switching_dim(self) -> None:
        tfn = ActiveTFN(d_dims=[64, 64, 32, 64], rank=8)
        embeddings = [
            torch.randn(1, 64),
            torch.randn(1, 64),
            torch.randn(1, 32),
            torch.randn(1, 64),
        ]

        tensor = tfn(embeddings)

        self.assertEqual(tuple(tensor.shape), (1, 8, 8, 8, 8))
        for idx in range(4):
            self.assertEqual(tuple(tfn.get_slice(tensor, idx).shape), (1, 512))

    def test_buffered_encoder_defaults_to_identity_mode(self) -> None:
        payload = self.session.get_fusion_input(
            {
                "graph_id": "cold_start_smoke",
                "session_id": "test_session",
                "window": {"window_id": "window_000"},
                "nodes": [],
                "edges": [],
            }
        )

        encoder = SwitchingBufferedEncoder()
        embedding, freshness = encoder.step(payload)
        debug = encoder.debug_state()

        self.assertEqual(tuple(embedding.shape), (1, 64))
        self.assertEqual(freshness, 1.0)
        self.assertTrue(debug["cold_start"])
        self.assertEqual(debug["mode"], "identity")
        self.assertEqual(debug["metadata"].get("module"), "switching")

    def test_random_frozen_ablation_still_available(self) -> None:
        raw_embedding = np.random.randn(64).astype(np.float32)
        encoder = SwitchingBufferedEncoder(mode="random_frozen_tcn")

        embedding, freshness = encoder.step({"embedding": raw_embedding})

        self.assertEqual(tuple(embedding.shape), (1, 32))
        self.assertGreater(freshness, 0.0)
        self.assertEqual(encoder.debug_state()["mode"], "random_frozen_tcn")
        self.assertIsInstance(encoder, SwitchingRandomFrozenTCNEncoder)


if __name__ == "__main__":
    unittest.main()
