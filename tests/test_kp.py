import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knit.kp import audit_run, image_stats, kp_mask
from knit.metrics import nearest_train_details


class TestKp(unittest.TestCase):
    def test_kp_mask(self) -> None:
        labels = np.array(
            [
                [[0, 1], [1, 0]],
                [[0, 0], [0, 0]],
                [[0, 1], [2, 0]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(kp_mask(labels), [True, False, False])

    def test_image_stats_single_p_cell(self) -> None:
        labels = np.zeros((4, 4), dtype=np.uint8)
        labels[1, 2] = 1
        stats = image_stats(labels)
        self.assertTrue(stats["kp"])
        self.assertAlmostEqual(stats["p_ratio"], 1 / 16)
        self.assertEqual(stats["shift_score"], 0.0)
        self.assertEqual(stats["largest_p_component"], 1)

    def test_image_stats_block(self) -> None:
        labels = np.zeros((4, 4), dtype=np.uint8)
        labels[:2, :2] = 1
        stats = image_stats(labels)
        self.assertAlmostEqual(stats["shift_h"], 0.5)
        self.assertAlmostEqual(stats["shift_v"], 0.5)
        self.assertEqual(stats["largest_p_component"], 4)

    def test_nearest_train_details(self) -> None:
        generated = np.array([[[0, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.uint8)
        train = np.array([[[0, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=np.uint8)
        details = nearest_train_details(generated, train)
        self.assertEqual(details["distances"], [0, 1])
        self.assertEqual(details["indices"], [0, 0])
        self.assertEqual(details["exact_copy_count"], 1)

    def test_audit_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = np.array(
                [[[0, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.uint8
            )
            train = np.array([[[0, 0], [0, 1]]], dtype=np.uint8)
            labels_path = root / "labels.npy"
            train_path = root / "train_labels.npy"
            out_path = root / "kp_summary.json"
            np.save(labels_path, generated)
            np.save(train_path, train)
            result = audit_run(
                labels_path, out_path, train_all=train_path, train_kp=train_path
            )
            self.assertEqual(result["num_samples"], 2)
            self.assertEqual(result["kp_count"], 1)
            self.assertAlmostEqual(result["kp_hit_rate"], 0.5)
            self.assertEqual(result["records"][0]["nearest_all"], 0)
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["labels_path"], str(labels_path))


if __name__ == "__main__":
    unittest.main()
