import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> tuple[Path, Path]:
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((8, 20, 20), dtype=np.uint8)
    labels[:, ::2, :] = 1
    np.save(processed / "train_labels.npy", labels)
    np.save(processed / "val_labels.npy", labels[:2])
    (processed / "train_meta.json").write_text("[]", encoding="utf-8")
    (processed / "val_meta.json").write_text("[]", encoding="utf-8")
    config = {
        "image_size": 20,
        "in_channels": 17,
        "patch_size": 2,
        "hidden_size": 192,
        "depth": 2,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "num_classes": 1,
        "class_dropout_prob": 0.0,
        "learn_sigma": False,
        "noise_schedule": "squaredcos_cap_v2",
        "predict_xstart": True,
        "predict_v": False,
        "t_sample": "uniform",
        "min_snr_gamma": None,
        "batch_size": 2,
        "lr": 1e-5,
        "weight_decay": 0.0,
        "updates": 3,
        "val_every": 2,
        "ckpt_every": 2,
        "log_every": 10,
        "num_workers": 0,
        "amp": "none",
        "grad_clip": 1.0,
        "warmup_updates": 0,
        "min_lr": 0.0,
        "lr_schedule": "constant",
        "ema_decay": 0.0,
        "ema_start_step": 0,
        "grad_accum_steps": 1,
        "data": {
            "train_labels": str(processed / "train_labels.npy"),
            "train_meta": str(processed / "train_meta.json"),
            "val_labels": str(processed / "val_labels.npy"),
            "val_meta": str(processed / "val_meta.json"),
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, processed


@unittest.skipUnless(torch is not None, "torch required for train smoke test")
class InitFromTrainTest(unittest.TestCase):
    def test_init_from_resets_step_and_records_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, processed = _write_fixture(root)
            base_args = [
                sys.executable,
                "-m",
                "knit.train",
                "--config",
                str(config_path),
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--val-every",
                "2",
                "--ckpt-every",
                "2",
            ]
            src_out = root / "src"
            first = subprocess.run(
                base_args + ["--out", str(src_out), "--updates", "3"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            src_summary = json.loads(
                (src_out / "train_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(src_summary["last_step"], 3)

            ft_out = root / "ft"
            second = subprocess.run(
                base_args
                + [
                    "--out",
                    str(ft_out),
                    "--updates",
                    "2",
                    "--init-from",
                    str(src_out / "model.pt"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            summary = json.loads(
                (ft_out / "train_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["start_step"], 0)
            self.assertEqual(summary["last_step"], 2)
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["init_from"], str(src_out / "model.pt"))
            self.assertEqual(summary["init_from_step"], 3)
            self.assertEqual(summary["init_from_weights"], "model")
            self.assertFalse(summary["init_from_has_ema"])
            self.assertTrue(summary["init_from_has_optimizer"])
            self.assertAlmostEqual(summary["first_lr"], 1e-5)
            self.assertEqual(
                summary["data"]["train_labels"], str(processed / "train_labels.npy")
            )
            self.assertEqual(summary["init_from_config_check"]["noise_schedule"]["match"], True)

    def test_init_from_and_resume_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = _write_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "knit.train",
                    "--config",
                    str(config_path),
                    "--out",
                    str(root / "out"),
                    "--device",
                    "cpu",
                    "--resume",
                    "a.pt",
                    "--init-from",
                    "b.pt",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mutually exclusive", result.stderr)


@unittest.skipUnless(torch is not None, "torch required for checkpoint test")
class CheckpointDescribeTest(unittest.TestCase):
    def test_describe_and_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            state = {"weight": torch.zeros(1)}
            torch.save(
                {
                    "model": state,
                    "ema": state,
                    "opt": {},
                    "step": 100000,
                    "config": {
                        "noise_schedule": "squaredcos_cap_v2",
                        "predict_xstart": True,
                        "predict_v": False,
                        "learn_sigma": False,
                    },
                },
                path,
            )
            from knit.checkpoint import describe_checkpoint

            info = describe_checkpoint(
                path,
                expect_step=100000,
                expect_ema=True,
                expect_noise_schedule="squaredcos_cap_v2",
                expect_predict_xstart=True,
            )
            self.assertTrue(info["expect_ok"])
            self.assertTrue(info["has_ema"])
            self.assertEqual(info["step"], 100000)
            bad = describe_checkpoint(path, expect_step=1)
            self.assertFalse(bad["expect_ok"])
            self.assertTrue(any("expected step" in item for item in bad["expect_errors"]))


if __name__ == "__main__":
    unittest.main()
