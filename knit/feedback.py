from typing import Callable, Iterable

import numpy as np
import torch

from knit import checker, repair


def make_feedback_fn(
    steps: Iterable[int],
    budget: int = 24,
    num_classes: int = 17,
) -> Callable[[torch.Tensor], torch.Tensor]:
    target_steps = set(int(step) for step in steps)
    state = {
        "step": 0,
        "budget_left": int(budget),
        "checks": 0,
        "candidate_evaluations": 0,
        "corrections": 0,
        "corrected_samples": 0,
    }

    def feedback_fn(pred_xstart: torch.Tensor) -> torch.Tensor:
        step = state["step"]
        state["step"] += 1
        if step not in target_steps or state["budget_left"] <= 0:
            return pred_xstart

        labels = pred_xstart.argmax(dim=1)
        state["checks"] += int(labels.shape[0])
        output = pred_xstart.clone()
        for index in range(labels.shape[0]):
            original = labels[index].detach().cpu().numpy()
            report = checker.check(original, num_classes=num_classes)
            if report.ok:
                continue
            candidates = repair.propose(
                original,
                report,
                budget=state["budget_left"],
                num_classes=num_classes,
            )
            if not candidates:
                continue
            state["budget_left"] -= len(candidates)
            state["candidate_evaluations"] += len(candidates)
            best = repair.rank(candidates, report)
            if not np.array_equal(best, original):
                state["corrections"] += 1
                state["corrected_samples"] += 1
            encoded = repair.encode(best, num_classes=num_classes)
            output[index] = torch.from_numpy(encoded).to(
                dtype=pred_xstart.dtype, device=pred_xstart.device
            )
        return output

    feedback_fn.stats = state
    return feedback_fn
