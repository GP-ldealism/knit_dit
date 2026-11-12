from dataclasses import dataclass
from typing import Any

import numpy as np

CLASS_NAMES = (
    "K", "P", "T", "M",
    "FR1", "FR2", "FL1", "FL2",
    "BR1", "BR2", "BL1", "BL2",
    "XR+", "XR-", "XL+", "XL-",
    "S",
)

KNIT_CLASSES = frozenset({0, 1, 2, 3})
MOVE_CLASSES = frozenset({4, 5, 6, 7, 8, 9, 10, 11})
CROSS_CLASSES = frozenset({12, 13, 14, 15})
STACK_CLASSES = frozenset({16})
COMPLEX_CLASSES = MOVE_CLASSES | CROSS_CLASSES | STACK_CLASSES


@dataclass(frozen=True)
class LabelSpec:
    num_classes: int = 17
    low: float = -1.0
    high: float = 1.0

    def validate(self, labels: Any) -> list[str]:
        arr = np.asarray(labels)
        issues: list[str] = []
        if arr.ndim not in (2, 3):
            issues.append(f"expected labels with shape [H,W] or [N,H,W], got {arr.shape}")
            return issues
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            issues.append(f"expected integer labels, got dtype {arr.dtype}")
        if arr.size:
            low = int(arr.min())
            high = int(arr.max())
            if low < 0 or high >= self.num_classes:
                issues.append(
                    f"label range {low}..{high} outside 0..{self.num_classes - 1}"
                )
        return issues

    @property
    def names(self) -> tuple[str, ...]:
        if self.num_classes <= len(CLASS_NAMES):
            return CLASS_NAMES[: self.num_classes]
        return CLASS_NAMES + tuple(
            f"class_{index}" for index in range(len(CLASS_NAMES), self.num_classes)
        )

    def name_of(self, label: int) -> str:
        index = int(label)
        if 0 <= index < len(self.names):
            return self.names[index]
        return f"class_{index}"

    @staticmethod
    def group_of(label: int) -> str:
        index = int(label)
        if index in KNIT_CLASSES:
            return "knit"
        if index in MOVE_CLASSES:
            return "move"
        if index in CROSS_CLASSES:
            return "cross"
        if index in STACK_CLASSES:
            return "stack"
        return "unknown"

    def encode(self, labels: Any) -> np.ndarray:
        issues = self.validate(labels)
        if issues:
            raise ValueError("; ".join(issues))
        arr = np.asarray(labels, dtype=np.int64)
        onehot = np.eye(self.num_classes, dtype=np.float32)[arr]
        if arr.ndim == 2:
            onehot = np.moveaxis(onehot, -1, 0)
        else:
            onehot = np.moveaxis(onehot, -1, 1)
        return onehot * (self.high - self.low) + self.low

    def decode(self, x: Any) -> np.ndarray:
        if hasattr(x, "dim") and hasattr(x, "argmax"):
            dim = int(x.dim())
            if dim == 4:
                return x.argmax(dim=1).detach().cpu().numpy()
            if dim == 3:
                return x.argmax(dim=0).detach().cpu().numpy()
            raise ValueError(f"expected [C,H,W] or [N,C,H,W], got {tuple(x.shape)}")
        arr = np.asarray(x)
        if arr.ndim == 4:
            return arr.argmax(axis=1)
        if arr.ndim == 3:
            return arr.argmax(axis=0)
        raise ValueError(f"expected [C,H,W] or [N,C,H,W], got {arr.shape}")

    def histogram(self, labels: Any) -> np.ndarray:
        arr = np.asarray(labels).ravel()
        counts = np.zeros(self.num_classes, dtype=np.int64)
        if arr.size == 0:
            return counts
        values, value_counts = np.unique(arr, return_counts=True)
        for value, count in zip(values, value_counts):
            index = int(value)
            if 0 <= index < self.num_classes:
                counts[index] += int(count)
        return counts

    def presence(self, labels: Any) -> list[int]:
        counts = self.histogram(labels)
        return [index for index, count in enumerate(counts) if count > 0]

    def roundtrip(self, labels: Any) -> bool:
        arr = np.asarray(labels)
        decoded = self.decode(self.encode(arr))
        return bool(np.array_equal(arr, decoded))
