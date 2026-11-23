import argparse
from pathlib import Path
from typing import Any

import numpy as np

from knit.io import load_json, save_json

RUNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "A_step0": ("A 历史 100k", ("comparison/a0_s42/sample",)),
    "B_step500": ("B 全量 +500", ("ft_full/sample_step500", "train/sample_step500")),
    "B_step1000": ("B 全量 +1k", ("ft_full/sample_step1000", "train/sample_step1000")),
    "C_step500": ("C K/P +500", ("ft_kp/sample_step500", "train_kp/sample_step500")),
    "C_step1000": ("C K/P +1k", ("ft_kp/sample_step1000", "train_kp/sample_step1000")),
}

JUDGMENT_KEYS = ("recognizable_ids", "borderline_ids")


def resolve_run(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return None


def load_judgments(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    data = load_json(path)
    groups = data.get("groups") if isinstance(data, dict) else None
    return groups if isinstance(groups, dict) else data if isinstance(data, dict) else {}


def exact_copy_ids(labels_path: Path | None, train_labels_path: Path | None) -> set[int] | None:
    if labels_path is None or train_labels_path is None:
        return None
    if not labels_path.is_file() or not train_labels_path.is_file():
        return None
    generated = np.load(labels_path).reshape(-1, 400).tolist()
    train = np.load(train_labels_path).reshape(-1, 400).tolist()
    train_keys = {tuple(row) for row in train}
    return {index for index, row in enumerate(generated) if tuple(row) in train_keys}


def style_count(judgment: dict[str, Any], style: str) -> int | None:
    styles = judgment.get("styles")
    if isinstance(styles, dict):
        return int(styles.get(style, 0))
    details = judgment.get("judgments")
    if not isinstance(details, dict):
        return None
    return sum(1 for value in details.values() if value == style)


def build_rows(
    root: Path,
    judgments: dict[str, Any],
    train_labels_path: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = judgments.get("groups") if isinstance(judgments.get("groups"), dict) else judgments
    for key, (label, candidates) in RUNS.items():
        run = resolve_run(root, candidates)
        row: dict[str, Any] = {
            "key": key,
            "label": label,
            "status": "未产出" if run is None else "缺文件",
            "run_dir": str(run) if run is not None else None,
            "num_samples": None,
            "kp_count": None,
            "recognizable": None,
            "effective": None,
            "kp_recog_rate": None,
            "p_ratio": None,
            "shift": None,
            "train_copy": None,
            "duplicate": None,
            "nearest": None,
            "s1": None,
            "s4": None,
            "recognizable_ids": None,
        }
        if run is not None:
            eval_path = run / "eval.json"
            kp_path = run / "kp_summary.json"
            if eval_path.is_file() and kp_path.is_file():
                row["status"] = "已完成"
                report = load_json(eval_path)
                kp_report = load_json(kp_path)
                row["num_samples"] = int(report.get("num_samples", kp_report.get("num_samples", 0)))
                row["kp_count"] = int(kp_report.get("kp_count", 0))
                row["p_ratio"] = kp_report.get("p_ratio_median")
                row["shift"] = kp_report.get("shift_median")
                row["train_copy"] = int(report.get("train_copy_count", 0))
                row["duplicate"] = int(report.get("duplicate_samples", 0))
                near = [
                    int(item["nearest_all"])
                    for item in kp_report.get("records", [])
                    if item.get("nearest_all") is not None
                ]
                row["nearest"] = float(np.median(near)) if near else None
                judgment = groups.get(key, {})
                raw_ids = judgment.get("recognizable_ids") if isinstance(judgment, dict) else None
                if raw_ids is not None:
                    recog_ids = {int(value) for value in raw_ids}
                    row["recognizable"] = len(recog_ids)
                    row["recognizable_ids"] = sorted(recog_ids)
                    row["s1"] = style_count(judgment, "S1_THIN_V")
                    row["s4"] = style_count(judgment, "S4_STAIR")
                    exact = exact_copy_ids(run / "labels.npy", train_labels_path)
                    if exact is not None:
                        row["effective"] = len(recog_ids - exact)
                    elif row["train_copy"] == 0:
                        row["effective"] = len(recog_ids)
                    if row["kp_count"]:
                        row["kp_recog_rate"] = row["recognizable"] / row["kp_count"]
        rows.append(row)
    return rows


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recog_map = {
        row["key"]: set(row["recognizable_ids"] or [])
        for row in rows
        if row["recognizable_ids"] is not None
    }
    comparisons: list[dict[str, Any]] = []
    for key in ("C_step500", "C_step1000"):
        if key not in recog_map:
            continue
        for base in ("A_step0", "B_step1000"):
            if base not in recog_map:
                continue
            comparisons.append(
                {
                    "group": key,
                    "base": base,
                    "gained": sorted(recog_map[key] - recog_map[base]),
                    "lost": sorted(recog_map[base] - recog_map[key]),
                }
            )
    return comparisons


def fmt_int(value: Any) -> str:
    return "-" if value is None else str(value)


def fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1%}"


def fmt_float(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def render_markdown(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# 11.23 第二步 A/B/C 结果汇总",
        "",
        "- 口径：method=a0，seed=42，DDIM 250，EMA，固定请求前 100 条。",
        "- 可辨样式按 `docs/11.23_kp_step1.md`；数值来自各组 `eval.json`、`kp_summary.json` 与 `step2_style_judgments.json`。",
        "- 未产出或未填写人工判读的项显示“-”或“待测”。",
        "",
        "| 组别 | 状态 | 总请求 | 混合 K/P | 可辨 K/P | 有效花型 | K/P 内可辨率 | P 占比中位数 | 平移得分中位数 | 精确复制/组内重复 | K/P 最近邻 | S1_THIN_V | S4_STAIR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        recog = fmt_int(row["recognizable"])
        effective = fmt_int(row["effective"])
        lines.append(
            "| {label} | {status} | {n} | {kp} | {recog} | {eff} | {rate} | {p} | {shift} | {copy}/{dup} | {near} | {s1} | {s4} |".format(
                label=row["label"],
                status=row["status"],
                n=fmt_int(row["num_samples"]),
                kp=fmt_int(row["kp_count"]),
                recog=recog,
                eff=effective,
                rate=fmt_pct(row["kp_recog_rate"]),
                p=fmt_pct(row["p_ratio"]),
                shift=fmt_float(row["shift"]),
                copy=fmt_int(row["train_copy"]),
                dup=fmt_int(row["duplicate"]),
                near="-" if row["nearest"] is None else f"{row['nearest']:.1f}/400",
                s1=fmt_int(row["s1"]),
                s4=fmt_int(row["s4"]),
            )
        )
    lines.extend(["", "## 配对比较", ""])
    if not comparisons:
        lines.append("- C 组未产出或尚无人工判读，暂无法做配对比较。")
    else:
        for item in comparisons:
            lines.append(
                f"- {item['group']} vs {item['base']}：新增 {item['gained']}，丢失 {item['lost']}。"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/knit_dit/11.23")
    parser.add_argument("--base-root", default="outputs/knit_dit/11.12")
    parser.add_argument("--judgments")
    parser.add_argument("--train-labels")
    parser.add_argument("--out")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    root = Path(args.root)
    base_root = Path(args.base_root)
    judgments_path = Path(args.judgments) if args.judgments else root / "step2_style_judgments.json"
    train_labels_path = (
        Path(args.train_labels)
        if args.train_labels
        else base_root / "processed/train_labels.npy"
    )
    judgments = load_judgments(judgments_path)
    rows = build_rows(root, judgments, train_labels_path)
    comparisons = paired_comparisons(rows)
    markdown = render_markdown(rows, comparisons)

    out_path = Path(args.out) if args.out else root / "step2_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    if args.json_out:
        save_json(
            args.json_out,
            {"rows": rows, "comparisons": comparisons},
        )
    print(markdown, end="")


if __name__ == "__main__":
    main()
