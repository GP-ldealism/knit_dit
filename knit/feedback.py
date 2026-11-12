from typing import Callable, Iterable

import torch

from knit import checker, repair


def make_feedback_fn(
    steps: Iterable[int],
    budget: int = 24,
    num_classes: int = 17,
) -> Callable[[torch.Tensor], torch.Tensor]:
    target_steps = set(int(step) for step in steps)
    state = {"step": 0, "budget_left": int(budget)}

    def feedback_fn(pred_xstart: torch.Tensor) -> torch.Tensor:
        step = state["step"]
        state["step"] += 1
        if step not in target_steps or state["budget_left"] <= 0:
            return pred_xstart

        labels = pred_xstart.argmax(dim=1)
        output = pred_xstart.clone()
        for index in range(labels.shape[0]):
            report = checker.check(
                labels[index].detach().cpu().numpy(), num_classes=num_classes
            )
            if report.ok:
                continue
            candidates = repair.propose(
                labels[index].detach().cpu().numpy(),
                report,
                budget=state["budget_left"],
                num_classes=num_classes,
            )
            if not candidates:
                continue
            state["budget_left"] -= len(candidates)
            best = repair.rank(candidates, report)
            encoded = repair.encode(best, num_classes=num_classes)
            output[index] = torch.from_numpy(encoded).to(
                dtype=pred_xstart.dtype, device=pred_xstart.device
            )
        return output

    return feedback_fn
