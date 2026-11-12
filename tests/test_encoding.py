import unittest

import numpy as np

from knit.encoding import CROSS_CLASSES, LabelSpec


class TestLabelSpec(unittest.TestCase):
    def test_roundtrip_single_and_batch(self) -> None:
        spec = LabelSpec()
        single = np.array([[0, 16], [12, 13]], dtype=np.uint8)
        encoded = spec.encode(single)
        self.assertEqual(encoded.shape, (17, 2, 2))
        self.assertEqual(float(encoded.min()), -1.0)
        self.assertEqual(float(encoded.max()), 1.0)
        self.assertTrue(np.array_equal(spec.decode(encoded), single))
        self.assertTrue(spec.roundtrip(single))

        batch = np.stack([single, single])
        encoded_batch = spec.encode(batch)
        self.assertEqual(encoded_batch.shape, (2, 17, 2, 2))
        self.assertTrue(np.array_equal(spec.decode(encoded_batch), batch))

    def test_validation_and_histogram(self) -> None:
        spec = LabelSpec()
        self.assertEqual(spec.validate(np.zeros((2, 2), dtype=np.uint8)), [])
        self.assertTrue(spec.validate(np.zeros((2, 2), dtype=np.float32)))
        self.assertTrue(spec.validate(np.array([[0, 17]], dtype=np.int64)))
        histogram = spec.histogram(np.array([[0, 1], [1, 16]], dtype=np.uint8))
        self.assertEqual(int(histogram[0]), 1)
        self.assertEqual(int(histogram[1]), 2)
        self.assertEqual(int(histogram[16]), 1)
        self.assertEqual(set(spec.presence(np.array([[0, 1], [1, 16]]))), {0, 1, 16})

    def test_class_groups(self) -> None:
        self.assertEqual(LabelSpec.group_of(0), "knit")
        self.assertEqual(LabelSpec.group_of(4), "move")
        self.assertEqual(LabelSpec.group_of(12), "cross")
        self.assertEqual(LabelSpec.group_of(16), "stack")
        self.assertEqual(LabelSpec.group_of(99), "unknown")
        self.assertEqual(CROSS_CLASSES, frozenset({12, 13, 14, 15}))


if __name__ == "__main__":
    unittest.main()
