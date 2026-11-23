import tempfile
import unittest
from pathlib import Path

import numpy as np

from knit.io import save_json
from knit.step2_report import build_rows, paired_comparisons, render_markdown, resolve_run


class TestStep2Report(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = np.zeros((2, 20, 20), dtype=np.uint8)
        self.train[0, 19, 19] = 1
        self.train_path = self.root / "train_labels.npy"
        np.save(self.train_path, self.train)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_run(self, rel: str, kp_ids: list[int], exact_id: int | None = None) -> Path:
        run = self.root / rel
        run.mkdir(parents=True, exist_ok=True)
        labels = np.zeros((4, 20, 20), dtype=np.uint8)
        for index in kp_ids:
            labels[index, 0, 0] = 1
        if exact_id is not None:
            labels[exact_id] = self.train[0]
        np.save(run / "labels.npy", labels)
        save_json(
            run / "eval.json",
            {
                "num_samples": 4,
                "train_copy_count": 1 if exact_id is not None else 0,
                "duplicate_samples": 0,
                "novelty": {"median": 10},
            },
        )
        save_json(
            run / "kp_summary.json",
            {
                "num_samples": 4,
                "kp_count": len(kp_ids),
                "p_ratio_median": 0.2,
                "shift_median": 0.7,
                "records": [
                    {"index": index, "nearest_all": 10 + index} for index in kp_ids
                ],
            },
        )
        return run

    def test_resolve_run_falls_back_to_synced_layout(self) -> None:
        expected = self.make_run("train/sample_step1000", [0])
        found = resolve_run(
            self.root, ("ft_full/sample_step1000", "train/sample_step1000")
        )
        self.assertEqual(found, expected)

    def test_build_rows_uses_judgments_and_exact_copies(self) -> None:
        self.make_run("comparison/a0_s42/sample", [0, 1])
        self.make_run("ft_full/sample_step1000", [1, 2])
        self.make_run("ft_kp/sample_step1000", [2, 3], exact_id=3)
        judgments = {
            "groups": {
                "A_step0": {
                    "recognizable_ids": [0, 1],
                    "styles": {"S1_THIN_V": 0, "S4_STAIR": 0},
                },
                "B_step1000": {
                    "recognizable_ids": [1, 2],
                    "styles": {"S1_THIN_V": 0, "S4_STAIR": 0},
                },
                "C_step1000": {
                    "recognizable_ids": [2, 3],
                    "styles": {"S1_THIN_V": 1, "S4_STAIR": 0},
                },
            }
        }
        rows = build_rows(self.root, judgments, self.train_path)
        by_key = {row["key"]: row for row in rows}
        self.assertEqual(by_key["A_step0"]["recognizable"], 2)
        self.assertEqual(by_key["A_step0"]["effective"], 2)
        self.assertEqual(by_key["C_step1000"]["recognizable"], 2)
        self.assertEqual(by_key["C_step1000"]["effective"], 1)
        self.assertEqual(by_key["C_step1000"]["s1"], 1)
        self.assertEqual(by_key["C_step500"]["status"], "未产出")

        comparisons = paired_comparisons(rows)
        c_vs_a = next(
            item
            for item in comparisons
            if item["group"] == "C_step1000" and item["base"] == "A_step0"
        )
        self.assertEqual(c_vs_a["gained"], [2, 3])
        self.assertEqual(c_vs_a["lost"], [0, 1])
        markdown = render_markdown(rows, comparisons)
        self.assertIn("| C K/P +1k | 已完成", markdown)


if __name__ == "__main__":
    unittest.main()
