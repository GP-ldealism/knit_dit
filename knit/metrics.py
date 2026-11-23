from collections import Counter
from typing import Any, Iterable

import numpy as np

from knit import checker
from knit.encoding import (
    COMPLEX_CLASSES,
    CROSS_CLASSES,
    MOVE_CLASSES,
    STACK_CLASSES,
    LabelSpec,
)
from knit.io import sha256_array


def _as_batch(labels: Any) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected [H,W] or [N,H,W] labels, got {arr.shape}")
    return arr


def _mode(values: Iterable[Any]) -> Any:
    values = list(values)
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def label_validity(labels: np.ndarray, spec: LabelSpec) -> dict[str, Any]:
    total = int(labels.shape[0])
    invalid_samples = 0
    issue_counts: dict[str, int] = {}
    for item in labels:
        issues = spec.validate(item)
        if issues:
            invalid_samples += 1
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return {
        "valid_samples": total - invalid_samples,
        "invalid_samples": invalid_samples,
        "valid_rate": (total - invalid_samples) / total if total else 0.0,
        "issue_counts": issue_counts,
    }


def duplicate_metrics(labels: np.ndarray) -> dict[str, Any]:
    hashes = [sha256_array(item) for item in labels]
    counts = Counter(hashes)
    unique = len(counts)
    duplicate_groups = {digest: count for digest, count in counts.items() if count > 1}
    total = int(labels.shape[0])
    return {
        "unique_samples": unique,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_samples": int(sum(duplicate_groups.values())),
        "duplicate_rate": 1.0 - unique / total if total else 0.0,
    }


def class_metrics(labels: np.ndarray, spec: LabelSpec) -> dict[str, Any]:
    counts = spec.histogram(labels)
    total = int(counts.sum())
    present = [index for index, count in enumerate(counts) if count > 0]
    frequencies = {
        str(index): float(count / total) if total else 0.0
        for index, count in enumerate(counts)
    }
    probabilities = counts[counts > 0].astype(np.float64) / total if total else np.array([])
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if probabilities.size else 0.0
    max_entropy = float(np.log(spec.num_classes)) if spec.num_classes > 1 else 0.0
    return {
        "counts": {str(index): int(count) for index, count in enumerate(counts)},
        "frequencies": frequencies,
        "names": {str(index): spec.name_of(index) for index in range(spec.num_classes)},
        "present_classes": present,
        "coverage_rate": len(present) / spec.num_classes if spec.num_classes else 0.0,
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
    }


def distribution_metrics(
    labels: np.ndarray,
    reference: np.ndarray,
    spec: LabelSpec,
) -> dict[str, Any]:
    counts = spec.histogram(labels).astype(np.float64)
    reference_counts = spec.histogram(reference).astype(np.float64)
    total = float(counts.sum())
    reference_total = float(reference_counts.sum())
    if total == 0 or reference_total == 0:
        return {"tv": None, "l1": None}
    frequencies = counts / total
    reference_frequencies = reference_counts / reference_total
    difference = np.abs(frequencies - reference_frequencies)
    return {
        "tv": float(0.5 * difference.sum()),
        "l1": float(difference.sum()),
        "frequencies": [float(value) for value in frequencies],
        "reference_frequencies": [float(value) for value in reference_frequencies],
    }


def structure_metrics(labels: np.ndarray) -> dict[str, Any]:
    total = int(labels.shape[0])
    flat = labels.reshape(total, -1)
    cross_mask = np.isin(flat, list(CROSS_CLASSES))
    move_mask = np.isin(flat, list(MOVE_CLASSES))
    stack_mask = np.isin(flat, list(STACK_CLASSES))
    complex_mask = np.isin(flat, list(COMPLEX_CLASSES))
    cross_samples = int(cross_mask.any(axis=1).sum()) if total else 0
    move_samples = int(move_mask.any(axis=1).sum()) if total else 0
    stack_samples = int(stack_mask.any(axis=1).sum()) if total else 0
    complex_samples = int(complex_mask.any(axis=1).sum()) if total else 0
    knit_only_samples = total - complex_samples
    cells = int(flat.size)
    return {
        "cross_cells": int(cross_mask.sum()),
        "cross_cell_rate": int(cross_mask.sum()) / cells if cells else 0.0,
        "cross_samples": cross_samples,
        "cross_sample_rate": cross_samples / total if total else 0.0,
        "move_cells": int(move_mask.sum()),
        "move_cell_rate": int(move_mask.sum()) / cells if cells else 0.0,
        "move_samples": move_samples,
        "move_sample_rate": move_samples / total if total else 0.0,
        "stack_cells": int(stack_mask.sum()),
        "stack_cell_rate": int(stack_mask.sum()) / cells if cells else 0.0,
        "stack_samples": stack_samples,
        "stack_sample_rate": stack_samples / total if total else 0.0,
        "complex_samples": complex_samples,
        "complex_rate": complex_samples / total if total else 0.0,
        "knit_only_samples": knit_only_samples,
        "knit_only_rate": knit_only_samples / total if total else 0.0,
    }


def nearest_train_hamming(
    labels: np.ndarray,
    train_labels: np.ndarray,
    max_chunk_elements: int = 20_000_000,
) -> dict[str, Any]:
    generated = labels.reshape(labels.shape[0], -1)
    train = train_labels.reshape(train_labels.shape[0], -1)
    if generated.shape[1] != train.shape[1]:
        raise ValueError(
            f"label spatial size mismatch: {generated.shape[1]} vs {train.shape[1]}"
        )
    total = int(generated.shape[0])
    if total == 0 or train.shape[0] == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "exact_copy_count": 0,
            "exact_copy_rate": 0.0,
        }
    chunk_size = max(1, min(64, max_chunk_elements // max(train.shape[0] * train.shape[1], 1)))
    nearest = np.empty(total, dtype=np.int32)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        distances = np.count_nonzero(
            generated[start:end, None, :] != train[None, :, :], axis=2
        )
        nearest[start:end] = distances.min(axis=1)
    exact = int(np.count_nonzero(nearest == 0))
    return {
        "mean": float(nearest.mean()),
        "median": float(np.median(nearest)),
        "min": int(nearest.min()),
        "max": int(nearest.max()),
        "exact_copy_count": exact,
        "exact_copy_rate": exact / total,
    }


def nearest_train_details(
    labels: np.ndarray,
    train_labels: np.ndarray,
    max_chunk_elements: int = 20_000_000,
) -> dict[str, Any]:
    generated = labels.reshape(labels.shape[0], -1)
    train = train_labels.reshape(train_labels.shape[0], -1)
    if generated.shape[1] != train.shape[1]:
        raise ValueError(
            f"label spatial size mismatch: {generated.shape[1]} vs {train.shape[1]}"
        )
    total = int(generated.shape[0])
    if total == 0 or train.shape[0] == 0:
        return {
            "distances": [],
            "indices": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "exact_copy_count": 0,
            "exact_copy_rate": 0.0,
        }
    chunk_size = max(1, min(64, max_chunk_elements // max(train.shape[0] * train.shape[1], 1)))
    distances = np.empty(total, dtype=np.int32)
    indices = np.empty(total, dtype=np.int32)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        block = np.count_nonzero(
            generated[start:end, None, :] != train[None, :, :], axis=2
        )
        distances[start:end] = block.min(axis=1)
        indices[start:end] = block.argmin(axis=1)
    exact = int(np.count_nonzero(distances == 0))
    return {
        "distances": [int(value) for value in distances],
        "indices": [int(value) for value in indices],
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "min": int(distances.min()),
        "max": int(distances.max()),
        "exact_copy_count": exact,
        "exact_copy_rate": exact / total,
    }


def check_metrics(labels: np.ndarray, spec: LabelSpec) -> dict[str, Any]:
    reports = checker.check_batch(labels, num_classes=spec.num_classes)
    summary = checker.summarize(reports)
    total = int(summary["total"])
    known_reports = [report for report in reports if not report.unknown]
    known_passed = sum(1 for report in known_reports if report.ok)
    unknown_samples = sum(1 for report in reports if report.unknown)
    summary["known_rule_samples"] = len(known_reports)
    summary["known_rule_pass_rate"] = (
        known_passed / len(known_reports) if known_reports else None
    )
    summary["unknown_samples"] = unknown_samples
    summary["unknown_rate"] = unknown_samples / total if total else 0.0
    return summary


def cost_metrics(
    meta_records: Iterable[dict[str, Any]] | None = None,
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    records = list(meta_records or [])
    summary = run_summary or {}
    feedback_stats = summary.get("feedback_stats") or {}
    elapsed = summary.get("elapsed_sec")
    if elapsed is None and records:
        values = [float(record.get("elapsed_sec", 0.0)) for record in records]
        elapsed = sum(values) if any(values) else None
    candidate_evaluations = int(feedback_stats.get("candidate_evaluations", 0))
    if not candidate_evaluations and records:
        candidate_evaluations = sum(
            int(record.get("candidate_evaluations", 0)) for record in records
        )
    feedback_events = int(feedback_stats.get("checks", 0))
    if not feedback_events and records:
        feedback_events = sum(int(record.get("feedback_events", 0)) for record in records)
    if elapsed is None and not candidate_evaluations and not feedback_events:
        return None
    num_samples = int(summary.get("num_samples") or len(records))
    return {
        "total_elapsed_sec": float(elapsed) if elapsed is not None else None,
        "per_sample_sec": (
            float(elapsed) / num_samples if elapsed is not None and num_samples else None
        ),
        "candidate_evaluations": candidate_evaluations,
        "feedback_events": feedback_events,
        "feedback_corrections": int(feedback_stats.get("corrections", 0)),
        "feedback_corrected_samples": int(
            feedback_stats.get("corrected_samples", 0)
        ),
    }


def evaluate_labels(
    labels: Any,
    spec: LabelSpec | None = None,
    train_labels: Any | None = None,
    meta_records: Iterable[dict[str, Any]] | None = None,
    run_summary: dict[str, Any] | None = None,
    compute_novelty: bool = True,
) -> dict[str, Any]:
    spec = spec or LabelSpec()
    labels = _as_batch(labels)
    records = list(meta_records or [])
    total = int(labels.shape[0])
    hashes = [sha256_array(item) for item in labels]
    duplicates = duplicate_metrics(labels)
    structure = structure_metrics(labels)
    result: dict[str, Any] = {
        "num_samples": total,
        "num_classes": spec.num_classes,
        "label_validity": label_validity(labels, spec),
        "duplicates": duplicates,
        "unique_samples": duplicates["unique_samples"],
        "duplicate_rate": duplicates["duplicate_rate"],
        "duplicate_groups": duplicates["duplicate_groups"],
        "duplicate_samples": duplicates["duplicate_samples"],
        "class_stats": class_metrics(labels, spec),
        "structure": structure,
        "check": check_metrics(labels, spec),
        "cost": cost_metrics(records, run_summary),
    }
    if train_labels is not None:
        train = _as_batch(train_labels)
        train_hashes = {sha256_array(item) for item in train}
        copy_mask = np.array([digest in train_hashes for digest in hashes], dtype=bool)
        copied = int(copy_mask.sum())
        result["train_copy_count"] = copied
        result["train_copy_rate"] = copied / total if total else 0.0
        result["train_copy"] = {
            "count": copied,
            "rate": copied / total if total else 0.0,
        }
        result["distribution"] = distribution_metrics(labels, train, spec)
        result["novelty"] = (
            nearest_train_hamming(labels, train) if compute_novelty else None
        )
        complex_mask = np.isin(labels.reshape(total, -1), list(COMPLEX_CLASSES)).any(
            axis=1
        )
        complex_not_copy = int(np.count_nonzero(complex_mask & ~copy_mask))
        result["complex_proxy"] = {
            "samples": complex_not_copy,
            "rate": complex_not_copy / total if total else 0.0,
            "definition": "contains MOVE/CROSS/STACK classes and is not an exact train copy",
        }
    else:
        result["train_copy_count"] = 0
        result["train_copy_rate"] = 0.0
        result["train_copy"] = {"count": 0, "rate": 0.0, "available": False}
        result["distribution"] = None
        result["novelty"] = None
        result["complex_proxy"] = None
    result["method"] = _mode(record.get("method") for record in records)
    if run_summary:
        if result["method"] is None:
            result["method"] = run_summary.get("method")
        result["seed"] = run_summary.get("seed")
        result["ddim_steps"] = run_summary.get("ddim_steps")
        result["checkpoint"] = run_summary.get("checkpoint")
    return result
