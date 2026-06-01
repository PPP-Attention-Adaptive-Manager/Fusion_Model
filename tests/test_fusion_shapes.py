"""Shape contract tests for the fusion training path."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from fusion_model import InferrerFusion
from scripts.fusion.train_fusion import compute_fusion_loss


class FusionShapeContractTest(unittest.TestCase):
    def test_fusion_input_and_output_shapes(self) -> None:
        batch_size = 3
        embeddings = [
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 32),
            torch.randn(batch_size, 64),
        ]
        self.assertEqual(tuple(embeddings[0].shape), (batch_size, 64))
        self.assertEqual(tuple(embeddings[1].shape), (batch_size, 64))
        self.assertEqual(tuple(embeddings[2].shape), (batch_size, 32))
        self.assertEqual(tuple(embeddings[3].shape), (batch_size, 64))

        model = InferrerFusion()
        output = model(embeddings)

        self.assertEqual(tuple(output["global"].shape), (batch_size, 11))
        self.assertEqual(len(output["per_model"]), 4)
        for tensor in output["per_model"]:
            self.assertEqual(tuple(tensor.shape), (batch_size, 12))
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(torch.isfinite(output["global"]).all())

    def test_tucker_slice_shape_for_all_modalities(self) -> None:
        model = InferrerFusion()
        embeddings = [
            torch.randn(2, 64),
            torch.randn(2, 64),
            torch.randn(2, 32),
            torch.randn(2, 64),
        ]
        tensor = model.tfn(embeddings)
        for idx in range(4):
            self.assertEqual(tuple(model.tfn.get_slice(tensor, idx).shape), (2, 512))

    def test_loss_uses_state_scores_without_pre_softmax(self) -> None:
        factors = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
        state_label = torch.tensor([2], dtype=torch.long)
        state_scores = torch.tensor([[0.0, -1.0, 4.0, -2.0, 1.0]], dtype=torch.float32)
        output = {
            "global": torch.cat([factors + 0.1, state_scores, torch.zeros(1, 1)], dim=-1),
            "per_model": [],
        }

        losses = compute_fusion_loss(output, factors, state_label)
        expected = 0.4 * F.huber_loss(output["global"][:, :5], factors) + 0.6 * F.cross_entropy(
            state_scores,
            state_label,
        )

        self.assertTrue(torch.allclose(losses["loss"], expected))


if __name__ == "__main__":
    unittest.main()
