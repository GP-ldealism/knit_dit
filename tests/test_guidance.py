import tempfile
import unittest
from pathlib import Path

import numpy as np

from knit.guidance import (
    compare_runs,
    make_templates,
    parse_steps,
    select_candidates,
    thin_v_score_labels,
    thin_v_score_logits,
)


class TestGuidance(unittest.TestCase):
    def test_parse_steps(self) -> None:
        self.assertEqual(parse_steps("1,3-5,8:10", 12), [1, 3, 4, 5, 8, 9, 10])
        self.assertEqual(parse_steps("80-220:70", 250), [80, 150, 220])

    def test_templates_cover_shift_and_spacing(self) -> None:
        templates = make_templates(20, 20)
        self.assertEqual(templates.shape[0], len(templates))
        self.assertTrue(np.any(templates[:, 0, 0]))
        self.assertTrue(
            any(np.any(template[:, 1:] != template[:, :-1]) for template in templates)
        )

    def test_thin_vertical_score_prefers_template(self) -> None:
        perfect = np.zeros((20, 20), dtype=np.uint8)
        perfect[:, ::2] = 1
        blank = np.zeros_like(perfect)
        labels = np.stack([perfect, blank])
        scores = thin_v_score_labels(labels)
        self.assertGreater(scores[0], 0.85)
        self.assertLess(scores[1], 0.60)

    def test_thin_vertical_score_logits_input(self) -> None:
        logits = np.full((1, 17, 20, 20), -1.0, dtype=np.float32)
        logits[0, 0] = 1.0
        blank_score = thin_v_score_logits(logits)[0]
        logits[0, 0] = -1.0
        logits[0, 0, :, 1::2] = 1.0
        logits[0, 1, :, ::2] = 1.0
        pattern_score = thin_v_score_logits(logits)[0]
        self.assertGreater(pattern_score, blank_score)

    def test_select_candidates_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blank = np.zeros((2, 20, 20), dtype=np.uint8)
            pattern = blank.copy()
            pattern[:, :, ::2] = 1
            for name, labels in (("a", blank), ("b", pattern), ("c", pattern)):
                folder = root / name
                folder.mkdir()
                np.save(folder / "labels.npy", labels)
            selected = select_candidates(
                [root / "a", root / "b"], root / "selected"
            )
            self.assertEqual(selected["num_samples"], 2)
            np.testing.assert_array_equal(
                np.load(root / "selected" / "labels.npy"), pattern
            )
            payload = compare_runs(
                {
                    "a": root / "a",
                    "b": root / "selected",
                    "c_weak": root / "c",
                    "c_strong": root / "c",
                },
                threshold=0.7,
                out=root / "compare.json",
            )
            self.assertIn("a_to_b", payload["pairs"])
            self.assertTrue((root / "compare.json").is_file())


if __name__ == "__main__":
    unittest.main()
