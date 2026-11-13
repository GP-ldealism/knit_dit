import unittest

import numpy as np

from knit.metrics import evaluate_labels


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.train = np.array(
            [
                [[0, 0], [0, 0]],
                [[16, 16], [16, 16]],
            ],
            dtype=np.uint8,
        )
        self.generated = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[12, 13], [14, 15]],
            ],
            dtype=np.uint8,
        )

    def test_duplicate_copy_structure_and_novelty(self) -> None:
        result = evaluate_labels(self.generated, train_labels=self.train)
        self.assertEqual(result["num_samples"], 3)
        self.assertEqual(result["duplicates"]["unique_samples"], 2)
        self.assertEqual(result["duplicates"]["duplicate_groups"], 1)
        self.assertEqual(result["duplicates"]["duplicate_samples"], 2)
        self.assertEqual(result["train_copy_count"], 2)
        self.assertAlmostEqual(result["train_copy_rate"], 2 / 3)
        self.assertEqual(result["novelty"]["exact_copy_count"], 2)
        self.assertEqual(result["structure"]["cross_samples"], 1)
        self.assertEqual(result["complex_proxy"]["samples"], 1)
        self.assertEqual(result["check"]["pass_rate"], 1.0)

    def test_invalid_labels_are_reported(self) -> None:
        invalid = np.array([[[0, 17], [0, 0]]], dtype=np.int64)
        result = evaluate_labels(invalid)
        self.assertEqual(result["label_validity"]["invalid_samples"], 1)
        self.assertEqual(result["label_validity"]["valid_samples"], 0)
        self.assertEqual(result["check"]["passed"], 0)
        self.assertFalse(result["label_validity"]["valid_rate"])

    def test_distribution_against_train(self) -> None:
        result = evaluate_labels(self.generated, train_labels=self.train)
        distribution = result["distribution"]
        self.assertEqual(len(distribution["frequencies"]), 17)
        self.assertAlmostEqual(distribution["tv"], 0.5)
        self.assertAlmostEqual(distribution["l1"], 1.0)

    def test_cost_and_method(self) -> None:
        summary = {
            "method": "a2",
            "seed": 7,
            "num_samples": 3,
            "elapsed_sec": 1.5,
            "feedback_stats": {
                "checks": 9,
                "candidate_evaluations": 4,
                "corrections": 2,
                "corrected_samples": 2,
            },
        }
        result = evaluate_labels(self.generated, run_summary=summary)
        self.assertEqual(result["method"], "a2")
        self.assertEqual(result["seed"], 7)
        self.assertAlmostEqual(result["cost"]["per_sample_sec"], 0.5)
        self.assertEqual(result["cost"]["candidate_evaluations"], 4)
        self.assertEqual(result["cost"]["feedback_corrections"], 2)


if __name__ == "__main__":
    unittest.main()
