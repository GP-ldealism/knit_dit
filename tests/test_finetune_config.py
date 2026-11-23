import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from knit.finetune_config import build_finetune_config


class FinetuneConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "image_size": 20,
            "in_channels": 17,
            "noise_schedule": "squaredcos_cap_v2",
            "predict_xstart": True,
            "predict_v": False,
            "learn_sigma": False,
            "updates": 100000,
            "lr": 0.0001,
            "ema_decay": 0.9999,
            "ema_start_step": 10000,
        }

    def test_build_kp_config(self) -> None:
        config = build_finetune_config(self.source, "/data/kp", "/data/full")
        self.assertEqual(config["data"]["train_labels"], "/data/kp/train_labels.npy")
        self.assertEqual(config["data"]["train_meta"], "/data/kp/train_meta.json")
        self.assertEqual(config["data"]["val_labels"], "/data/kp/val_labels.npy")
        self.assertEqual(config["data"]["val_meta"], "/data/kp/val_meta.json")
        self.assertEqual(config["data"]["test_labels"], "/data/full/test_labels.npy")
        self.assertEqual(config["data"]["palette"], "/data/full/palette.json")
        self.assertEqual(config["updates"], 1000)
        self.assertEqual(config["lr"], 1e-5)
        self.assertEqual(config["warmup_updates"], 0)
        self.assertEqual(config["lr_schedule"], "constant")
        self.assertEqual(config["ema_decay"], 0.99)
        self.assertEqual(config["ema_start_step"], 0)
        self.assertEqual(config["ckpt_every"], 500)
        self.assertEqual(config["val_every"], 500)
        self.assertEqual(config["noise_schedule"], "squaredcos_cap_v2")
        self.assertTrue(config["predict_xstart"])
        self.assertEqual(self.source["lr"], 0.0001)
        self.assertEqual(self.source["ema_decay"], 0.9999)

    def test_reject_wrong_diffusion_source(self) -> None:
        bad = dict(self.source, noise_schedule="linear")
        with self.assertRaises(ValueError):
            build_finetune_config(bad, "/data/kp", "/data/full")
        bad = dict(self.source, predict_xstart=False)
        with self.assertRaises(ValueError):
            build_finetune_config(bad, "/data/kp", "/data/full")

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            out = root / "nested" / "config.json"
            source.write_text(
                json.dumps(self.source), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "knit.finetune_config",
                    "--source",
                    str(source),
                    "--out",
                    str(out),
                    "--data-dir",
                    "/data/kp",
                    "--base-data-dir",
                    "/data/full",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(config["data"]["train_labels"], "/data/kp/train_labels.npy")
            self.assertEqual(config["updates"], 1000)


if __name__ == "__main__":
    unittest.main()
