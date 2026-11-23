import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from knit.io import append_jsonl, load_json, save_json, save_labels_png, sha256_array
from knit.kp import kp_mask

K_ID = 0
P_ID = 1


def parse_steps(spec: str, total_steps: int) -> list[int]:
    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bounds = part.split(":")
            if len(bounds) == 2 and "-" not in bounds[0]:
                start, end = (int(value) for value in bounds)
                steps.update(range(start, end + 1))
            elif len(bounds) == 2 and "-" in bounds[0]:
                start, end = (int(value) for value in bounds[0].split("-"))
                increment = int(bounds[1])
                if increment <= 0:
                    raise ValueError(f"step increment must be positive: {part}")
                steps.update(range(start, end + 1, increment))
            else:
                raise ValueError(f"bad step range: {part}")
        elif "-" in part:
            bounds = part.split("-")
            if len(bounds) not in (2, 3):
                raise ValueError(f"bad step range: {part}")
            start, end = (int(value) for value in bounds[:2])
            increment = int(bounds[2]) if len(bounds) == 3 else 1
            if increment <= 0:
                raise ValueError(f"step increment must be positive: {part}")
            steps.update(range(start, end + 1, increment))
        else:
            steps.add(int(part))
    result = sorted(step for step in steps if 0 <= step < total_steps)
    if not result:
        raise ValueError(f"no valid steps in 0..{total_steps - 1}: {spec}")
    return result


def make_templates(height: int = 20, width: int = 20) -> np.ndarray:
    unique: dict[bytes, np.ndarray] = {}
    for line_width in (1, 2):
        for gap in range(1, 5):
            period = line_width + gap
            for offset in range(period):
                mask = np.zeros((height, width), dtype=bool)
                columns = [
                    column
                    for column in range(offset, width, period)
                    if column + line_width <= width
                ]
                if len(columns) < 2:
                    continue
                for column in columns:
                    mask[:, column : column + line_width] = True
                if mask.mean() > 0.5:
                    continue
                unique[mask.tobytes()] = mask
    if not unique:
        raise ValueError("no thin vertical templates")
    return np.stack(list(unique.values()))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) * float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def thin_v_score_logits(
    logits: np.ndarray,
    templates: np.ndarray | None = None,
    temperature: float = 4.0,
) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[1] < 2:
        raise ValueError(f"expected [N,C,H,W] logits, got {arr.shape}")
    if templates is None:
        templates = make_templates(arr.shape[2], arr.shape[3])
    probs = _softmax(arr, temperature)
    p_prob = probs[:, P_ID]
    k_prob = probs[:, K_ID]
    scores: list[np.ndarray] = []
    for mask in templates:
        foreground = p_prob[:, mask].mean(axis=1)
        background = k_prob[:, ~mask].mean(axis=1)
        scores.append(0.5 * foreground + 0.5 * background)
    return np.max(np.stack(scores, axis=0), axis=0)


def thin_v_score_labels(
    labels: np.ndarray,
    num_classes: int = 17,
    templates: np.ndarray | None = None,
    temperature: float = 4.0,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected [N,H,W] labels, got {arr.shape}")
    logits = np.full(
        (arr.shape[0], num_classes, arr.shape[1], arr.shape[2]),
        -1.0,
        dtype=np.float32,
    )
    batch_index = np.arange(arr.shape[0])[:, None, None]
    row_index = np.arange(arr.shape[1])[None, :, None]
    column_index = np.arange(arr.shape[2])[None, None, :]
    logits[batch_index, arr, row_index, column_index] = 1.0
    return thin_v_score_logits(logits, templates=templates, temperature=temperature)


def make_thin_v_cond_fn(
    model: Any,
    diffusion: Any,
    steps: Iterable[int],
    strength: float,
    temperature: float = 4.0,
) -> Any:
    if strength <= 0:
        return None
    import torch

    target = set(int(step) for step in steps)
    templates_np = make_templates()
    templates = torch.from_numpy(templates_np).to(device=next(model.parameters()).device)
    timestep_map = getattr(diffusion, "timestep_map", None)
    reverse_map = (
        {int(original): index for index, original in enumerate(timestep_map)}
        if timestep_map is not None
        else None
    )
    state = {
        "step": 0,
        "applied_steps": 0,
        "applied_samples": 0,
        "score_sum": 0.0,
    }

    def cond_fn(x: Any, t: Any, **model_kwargs: Any) -> Any:
        index = state["step"]
        state["step"] += 1
        if index not in target:
            return torch.zeros_like(x)
        with torch.enable_grad():
            x_work = x.detach().requires_grad_(True)
            if reverse_map is None:
                loop_t = t
            else:
                mapped = [reverse_map[int(value)] for value in t.detach().cpu().tolist()]
                loop_t = torch.tensor(mapped, dtype=t.dtype, device=t.device)
            out = diffusion.p_mean_variance(
                model,
                x_work,
                loop_t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            logits = out["pred_xstart"]
            logits = logits * float(temperature)
            probs = torch.softmax(logits, dim=1)
            p_prob = probs[:, P_ID]
            k_prob = probs[:, K_ID]
            scores: list[Any] = []
            for mask in templates:
                foreground = (p_prob * mask[None]).flatten(1).mean(dim=1)
                background = (k_prob * (~mask)[None]).flatten(1).mean(dim=1)
                scores.append(0.5 * foreground + 0.5 * background)
            score = torch.max(torch.stack(scores, dim=0), dim=0).values
            gradient = torch.autograd.grad(score.sum(), x_work)[0]
        norm = gradient.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        gradient = gradient / (norm + 1e-8) * float(strength)
        state["applied_steps"] += 1
        state["applied_samples"] += int(x.shape[0])
        state["score_sum"] += float(score.detach().sum().item())
        return gradient.detach()

    cond_fn.stats = state
    return cond_fn


def score_run(run_dir: str | Path, threshold: float = 0.7) -> dict[str, Any]:
    path = Path(run_dir)
    labels = np.load(path / "labels.npy")
    scores = thin_v_score_labels(labels)
    hits = scores >= float(threshold)
    return {
        "run_dir": str(path),
        "num_samples": int(len(labels)),
        "threshold": float(threshold),
        "rule_hits": int(hits.sum()),
        "rule_hit_rate": float(hits.mean()) if len(hits) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "scores": scores.tolist(),
        "kp_hits": int(kp_mask(labels).sum()),
    }


def select_candidates(inputs: list[str | Path], out: str | Path, palette: Any = None) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("select_candidates needs at least two candidate runs")
    arrays: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    for item in inputs:
        path = Path(item)
        labels_path = path / "labels.npy" if path.is_dir() else path
        labels = np.load(labels_path)
        arrays.append(labels)
        score_rows.append(thin_v_score_labels(labels))
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"candidate shapes differ: {sorted(shapes)}")
    stack = np.stack(score_rows, axis=0)
    chosen = np.argmax(stack, axis=0)
    count = arrays[0].shape[0]
    labels = np.stack(
        [arrays[int(candidate)][index] for index, candidate in enumerate(chosen)]
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    for offset, item in enumerate(labels):
        save_labels_png(item, out_dir / f"{offset:06d}.png", palette=palette)
    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()
    for offset, item in enumerate(labels):
        candidate = int(chosen[offset])
        append_jsonl(
            meta_path,
            {
                "id": offset,
                "candidate": candidate,
                "candidate_source": str(inputs[candidate]),
                "rule_score": float(stack[candidate, offset]),
                "sha256": sha256_array(item),
            },
        )
    summary = {
        "method": "b_select",
        "num_samples": int(count),
        "candidates": [str(item) for item in inputs],
        "chosen_counts": {str(index): int((chosen == index).sum()) for index in range(len(inputs))},
        "score_mean": float(np.max(stack, axis=0).mean()),
    }
    save_json(out_dir / "selection.json", summary)
    save_json(
        out_dir / "sample_summary.json",
        {
            "method": "b_select",
            "num_samples": int(count),
            "candidates": [str(item) for item in inputs],
            "selection": summary,
            "elapsed_sec": None,
        },
    )
    return summary


def compare_runs(
    runs: dict[str, str | Path],
    threshold: float,
    out: str | Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"threshold": float(threshold), "runs": {}, "pairs": {}}
    score_maps: dict[str, np.ndarray] = {}
    hit_maps: dict[str, np.ndarray] = {}
    for name, path in runs.items():
        result = score_run(path, threshold=threshold)
        result.pop("scores")
        payload["runs"][name] = result
        labels = np.load(Path(path) / "labels.npy")
        score_maps[name] = thin_v_score_labels(labels)
        hit_maps[name] = score_maps[name] >= float(threshold)
        flat = labels.reshape(len(labels), -1)
        keys = {row.tobytes() for row in flat}
        payload["runs"][name]["duplicate_samples"] = int(len(flat) - len(keys))
    base = "a"
    if base in hit_maps:
        for name in hit_maps:
            if name == base:
                continue
            gained = np.flatnonzero(~hit_maps[base] & hit_maps[name]).tolist()
            lost = np.flatnonzero(hit_maps[base] & ~hit_maps[name]).tolist()
            payload["pairs"][f"{base}_to_{name}"] = {
                "gained": gained,
                "lost": lost,
                "gained_count": len(gained),
                "lost_count": len(lost),
                "score_delta_mean": float((score_maps[name] - score_maps[base]).mean()),
            }
    if out is not None:
        save_json(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score")
    score_parser.add_argument("--run-dir", required=True)
    score_parser.add_argument("--threshold", type=float, default=0.7)
    score_parser.add_argument("--out")

    select_parser = sub.add_parser("select")
    select_parser.add_argument("--inputs", nargs="+", required=True)
    select_parser.add_argument("--out", required=True)
    select_parser.add_argument("--palette")

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--run", action="append", default=[])
    compare_parser.add_argument("--a")
    compare_parser.add_argument("--b")
    compare_parser.add_argument("--c-weak")
    compare_parser.add_argument("--c-strong")
    compare_parser.add_argument("--threshold", type=float, default=0.7)
    compare_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "score":
        result = score_run(args.run_dir, threshold=args.threshold)
        save_json(args.out, result) if args.out else None
        print(
            {
                key: result[key]
                for key in ("rule_hits", "rule_hit_rate", "kp_hits", "score_mean")
            }
        )
    elif args.command == "select":
        palette = load_json(args.palette) if args.palette else None
        if isinstance(palette, dict):
            palette = palette.get("palette")
        print(select_candidates(args.inputs, args.out, palette=palette))
    elif args.command == "compare":
        if args.run:
            runs = {}
            for item in args.run:
                if "=" not in item:
                    parser.error(f"bad --run value: {item}")
                name, path = item.split("=", 1)
                runs[name] = path
        else:
            if not all((args.a, args.b, args.c_weak, args.c_strong)):
                parser.error("compare needs --run or --a/--b/--c-weak/--c-strong")
            runs = {
                "a": args.a,
                "b": args.b,
                "c_weak": args.c_weak,
                "c_strong": args.c_strong,
            }
        print(compare_runs(runs, args.threshold, args.out)["pairs"])


if __name__ == "__main__":
    main()
