import unittest

from knit.model import (
    ModelSpec,
    build_dit,
    check_forward,
    count_parameters,
    describe_model,
)


CONFIG = {
    "image_size": 20,
    "patch_size": 2,
    "in_channels": 17,
    "hidden_size": 192,
    "depth": 6,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "num_classes": 1,
    "class_dropout_prob": 0.0,
    "learn_sigma": False,
}


class TestModel(unittest.TestCase):
    def test_spec_and_parameter_count(self) -> None:
        spec = ModelSpec.from_config(CONFIG)
        self.assertEqual(spec.validate(), [])
        self.assertEqual(spec.num_patches, 100)
        model = build_dit(CONFIG)
        self.assertEqual(count_parameters(model), 4_185_668)
        self.assertEqual(count_parameters(model, trainable_only=False), 4_204_868)

    def test_describe_and_forward(self) -> None:
        model = build_dit(CONFIG)
        description = describe_model(model)
        self.assertEqual(description["image_size"], 20)
        self.assertEqual(description["num_patches"], 100)
        report = check_forward(model, batch_size=2)
        self.assertEqual(report["input_shape"], [2, 17, 20, 20])
        self.assertEqual(report["output_shape"], [2, 17, 20, 20])
        self.assertTrue(report["finite"])


if __name__ == "__main__":
    unittest.main()
