"""End-to-end smoke tests for InferrerFusion wiring."""

from __future__ import annotations

import unittest

import torch

from fusion_model import DEFAULT_D_DIMS, InferrerFusion
from predictive_models import MODALITY_MODELS
from predictive_models.switching import ActiveModel as SwitchingActiveModel
from predictive_models.switching.v1_switching_gru import SwitchingGRU


class FusionSmokeTest(unittest.TestCase):
    def _dummy_embeddings(self, batch_size: int = 1):
        return [
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 32),
            torch.randn(batch_size, 64),
        ]

    def test_default_dims_are_identity_switching_dims(self) -> None:
        self.assertEqual(DEFAULT_D_DIMS, [64, 64, 32, 64])

    def test_registry_has_four_model_slots(self) -> None:
        self.assertEqual(len(MODALITY_MODELS), 4)
        for model_cls in MODALITY_MODELS:
            self.assertTrue(callable(model_cls))

    def test_switching_active_model_is_gru(self) -> None:
        self.assertIs(SwitchingActiveModel, SwitchingGRU)

    def test_fusion_forward_shapes(self) -> None:
        model = InferrerFusion()
        output = model(self._dummy_embeddings())

        self.assertEqual(tuple(output["global"].shape), (1, 11))
        self.assertEqual(len(output["per_model"]), 4)
        for per_model in output["per_model"]:
            self.assertEqual(tuple(per_model.shape), (1, 12))

    def test_fusion_accepts_batch_two(self) -> None:
        model = InferrerFusion()
        output = model(self._dummy_embeddings(batch_size=2))

        self.assertEqual(tuple(output["global"].shape), (2, 11))
        self.assertEqual(len(output["per_model"]), 4)
        for per_model in output["per_model"]:
            self.assertEqual(tuple(per_model.shape), (2, 12))

    def test_reset_subject_is_callable(self) -> None:
        model = InferrerFusion()
        model(self._dummy_embeddings())
        model.reset_subject()
        output = model(self._dummy_embeddings())

        self.assertEqual(tuple(output["global"].shape), (1, 11))

    def test_switching_gru_contract_and_reset(self) -> None:
        model = SwitchingGRU(input_flat_dim=512)
        output = model(torch.randn(1, 512))

        self.assertEqual(tuple(output.shape), (1, 12))
        self.assertIn("h", model.microstate)

        model.reset_microstate()
        self.assertEqual(model.microstate, {})


if __name__ == "__main__":
    unittest.main()
