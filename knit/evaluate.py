import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from knit import metrics
from knit.encoding import LabelSpec
from knit.io import load_json, read_jsonl, save_json


def evaluate(
    labels: np.ndarray,
    num_classes: int = 17,
    train_labels: np.ndarray | None = None,
    meta_records: list[dict[str, Any]] | None = None,
    run_summary: dict[str, Any] | None = None,
    compute_novelty: bool = True,
) -> dict[str, Any]:
    return metrics.evaluate_labels(
        labels,
        spec=LabelSpec(num_classes=num_classes),
        train_labels=train_labels,
        meta_records=meta_records,
        run_summary=run_summary,
        compute_novelty=compute_novelty,
    )


def evaluate_run(
    run_dir: str | Path,
    train_labels: np.ndarray | None = None,
    num_classes: int = 17,
    compute_novelty: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    labels = np.load(run_dir / "labels.npy")
    meta_records = read_jsonl(run_dir / "meta.jsonl")
    summary_path = run_dir / "sample_summary.json"
    run_summary = load_json(summary_path) if summary_path.is_file() else {}
    result = evaluate(
        labels,
        num_classes=num_classes,
        train_labels=train_labels,
        meta_records=meta_records,
        run_summary=run_summary,
        compute_novelty=compute_novelty,
    )
    result["run_dir"] = str(run_dir)
    result["elapsed_sec"] = run_summary.get("elapsed_sec")
    if result.get("seed") is None:
        result["seed"] = run_summary.get("seed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--train-labels", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs="+", default=None)
    parser.add_argument("--no-novelty", action="store_true")
    args = parser.parse_args()

    train_labels = np.load(args.train_labels) if args.train_labels else None
    compute_novelty = not args.no_novelty

    if args.compare:
        results = {
            run_dir: evaluate_run(
                run_dir,
                train_labels=train_labels,
                num_classes=args.num_classes,
                compute_novelty=compute_novelty,
            )
            for run_dir in args.compare
        }
        out = args.out or "runs/eval_compare.json"
        save_json(out, results)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    labels_path = args.labels
    out = args.out
    meta_records: list[dict[str, Any]] = []
    run_summary: dict[str, Any] = {}
    run_dir: Path | None = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        labels_path = labels_path or str(run_dir / "labels.npy")
        meta_path = Path(args.meta) if args.meta else run_dir / "meta.jsonl"
        summary_path = (
            Path(args.summary) if args.summary else run_dir / "sample_summary.json"
        )
        meta_records = read_jsonl(meta_path)
        run_summary = load_json(summary_path) if summary_path.is_file() else {}
        out = out or str(run_dir / "eval.json")
    if not labels_path:
        raise ValueError("provide --labels, --run-dir, or --compare")

    labels = np.load(labels_path)
    if args.meta:
        meta_records = read_jsonl(args.meta)
    if args.summary:
        run_summary = load_json(args.summary)
    summary = evaluate(
        labels,
        num_classes=args.num_classes,
        train_labels=train_labels,
        meta_records=meta_records,
        run_summary=run_summary,
        compute_novelty=compute_novelty,
    )
    if run_dir is not None:
        summary["run_dir"] = str(run_dir)
    if out:
        save_json(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
