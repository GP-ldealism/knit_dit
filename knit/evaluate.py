import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from knit import checker
from knit.io import save_json, sha256_array


def evaluate(
    labels: np.ndarray,
    num_classes: int = 17,
    train_labels: np.ndarray | None = None,
) -> dict:
    labels = np.asarray(labels)
    total = int(labels.shape[0])
    hashes = [sha256_array(item) for item in labels]
    counts = Counter(hashes)
    unique = len(counts)
    duplicates = {
        digest: count for digest, count in counts.items() if count > 1
    }
    train_hashes: set[str] = set()
    if train_labels is not None:
        train_hashes = {sha256_array(item) for item in np.asarray(train_labels)}
    copied = sum(1 for digest in hashes if digest in train_hashes)
    values, value_counts = np.unique(labels, return_counts=True)
    pixel_counts = {
        str(int(value)): int(count)
        for value, count in zip(values, value_counts)
    }
    reports = checker.check_batch(labels, num_classes=num_classes)
    return {
        "num_samples": total,
        "unique_samples": unique,
        "duplicate_rate": 1.0 - unique / total if total else 0.0,
        "duplicate_groups": len(duplicates),
        "duplicate_samples": sum(duplicates.values()),
        "train_copy_count": copied,
        "train_copy_rate": copied / total if total else 0.0,
        "class_presence": {str(k): v for k, v in sorted(pixel_counts.items())},
        "check": checker.summarize(reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--train-labels", default=None)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    labels_path = args.labels
    out = args.out
    if args.run_dir:
        run_dir = Path(args.run_dir)
        labels_path = labels_path or str(run_dir / "labels.npy")
        out = out or str(run_dir / "eval.json")
    if not labels_path:
        raise ValueError("provide --labels or --run-dir")

    labels = np.load(labels_path)
    train_labels = np.load(args.train_labels) if args.train_labels else None
    summary = evaluate(labels, num_classes=args.num_classes, train_labels=train_labels)
    if out:
        save_json(out, summary)
    print(summary)


if __name__ == "__main__":
    main()
