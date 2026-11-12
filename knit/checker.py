from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class CheckReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues, "unknown": self.unknown}


def check(labels: np.ndarray, num_classes: int = 17) -> CheckReport:
    arr = np.asarray(labels)
    issues: list[str] = []
    unknown: list[str] = []
    if arr.ndim != 2:
        issues.append(f"expected 2D labels, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        issues.append(f"expected integer labels, got dtype {arr.dtype}")
    if arr.size:
        low = int(arr.min())
        high = int(arr.max())
        if low < 0 or high >= num_classes:
            issues.append(f"label range {low}..{high} outside 0..{num_classes - 1}")
    unknown.append("CROSS pairing/priority rules not implemented")
    return CheckReport(ok=not issues, issues=issues, unknown=unknown)


def check_batch(labels: Iterable[np.ndarray], num_classes: int = 17) -> list[CheckReport]:
    return [check(item, num_classes=num_classes) for item in labels]


def summarize(reports: Iterable[CheckReport]) -> dict[str, Any]:
    reports = list(reports)
    total = len(reports)
    passed = sum(1 for report in reports if report.ok)
    issue_counts: dict[str, int] = {}
    unknown_counts: dict[str, int] = {}
    for report in reports:
        for issue in report.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for item in report.unknown:
            unknown_counts[item] = unknown_counts.get(item, 0) + 1
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "issue_counts": issue_counts,
        "unknown_counts": unknown_counts,
    }
