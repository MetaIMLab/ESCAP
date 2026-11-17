"""Small checks for the released ESCAP architecture and source protocol map."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nnUNet")]

from nnunetv2.training.nnUNetTrainer.MAPSBalancedThreeEvidenceHeadsTrainer import (
    MAPSBalancedThreeEvidenceHeadsTrainer,
    SOURCE_CLASS_MASKS,
    balanced_evidential_loss,
    source_id,
)


class ESCAPModelTest(unittest.TestCase):
    def test_source_protocols(self):
        self.assertEqual(source_id("CHD68_case"), 0)
        self.assertEqual(source_id("ImageALCAPA_case"), 1)
        self.assertEqual(source_id("MM_WHS_case"), 2)
        self.assertFalse(SOURCE_CLASS_MASKS[1, 3])
        self.assertTrue(SOURCE_CLASS_MASKS[1, 8])
        with self.assertRaises(ValueError):
            source_id("unknown_case")

    def test_forward_and_balanced_evidence(self):
        model = MAPSBalancedThreeEvidenceHeadsTrainer.build_network_architecture().eval()
        with torch.no_grad():
            logits, auxiliary = model(
                torch.randn(1, 1, 32, 32, 32), domain_ids=torch.tensor([1]),
                return_aux=True,
            )
            target = torch.ones(1, 1, 32, 32, 32, dtype=torch.long)
            edl = balanced_evidential_loss(auxiliary["alpha"], target, torch.tensor([1]))
        self.assertEqual(tuple(logits.shape), (1, 10, 32, 32, 32))
        self.assertEqual(tuple(auxiliary["selected_queries"].shape), (1, 256))
        self.assertTrue(torch.isfinite(edl))


if __name__ == "__main__":
    unittest.main()
