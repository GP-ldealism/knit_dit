from typing import Iterable

import numpy as np

from knit.checker import CheckReport
from knit.io import labels_to_onehot


def propose(
    labels: np.ndarray,
    report: CheckReport,
    budget: int = 24,
    num_classes: int = 17,
) -> list[np.ndarray]:
    if budget < 1:
        return []
    return [np.asarray(labels, dtype=np.uint8).copy()]


def rank(candidates: Iterable[np.ndarray], report: CheckReport) -> np.ndarray:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no repair candidates")
    return candidates[0]


def encode(labels: np.ndarray, num_classes: int = 17) -> np.ndarray:
    return labels_to_onehot(labels, num_classes)
