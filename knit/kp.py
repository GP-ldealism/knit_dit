import argparse
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from knit.io import load_json, read_jsonl, save_json, sha256_array
from knit.metrics import nearest_train_details
from knit.visualize import palette_array, readable_palette, render_labels


PACK_K = 0
PACK_P = 1
PACK_PAIR = {PACK_K, PACK_P}
HISTORY_SAMPLE = Path("outputs/knit_dit/11.12/b2_100k/x0/sample")


def kp_mask(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError(f"expected [N,H,W] labels, got {arr.shape}")
    return np.array(
        [set(np.unique(item).tolist()) == PACK_PAIR for item in arr], dtype=bool
    )


def _max_component(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    best = 0
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            best = max(best, size)
    return best


def image_stats(item: np.ndarray, max_shift: int = 8) -> dict[str, Any]:
    arr = np.asarray(item)
    values = set(np.unique(arr).tolist())
    p_mask = arr == PACK_P
    shifts: dict[str, Any] = {}
    for name, axis in (("h", 1), ("v", 0)):
        best_score = 0.0
        best_period = 0
        for period in range(1, max_shift + 1):
            if axis == 1:
                left, right = p_mask[:, :-period], p_mask[:, period:]
            else:
                left, right = p_mask[:-period, :], p_mask[period:, :]
            union = int(np.count_nonzero(left | right))
            score = int(np.count_nonzero(left & right)) / union if union else 0.0
            if score > best_score:
                best_score, best_period = score, period
        shifts[f"shift_{name}"] = best_score
        shifts[f"period_{name}"] = best_period
    edges = int(np.count_nonzero(arr[:, 1:] != arr[:, :-1])) + int(
        np.count_nonzero(arr[1:, :] != arr[:-1, :])
    )
    edge_pairs = arr[:, 1:].size + arr[1:, :].size
    record: dict[str, Any] = {
        "kp": values == PACK_PAIR,
        "pure_k": values == {PACK_K},
        "pure_p": values == {PACK_P},
        "other_labels": sorted(values - PACK_PAIR),
        "p_ratio": float(p_mask.mean()),
        "edge_density": edges / edge_pairs if edge_pairs else 0.0,
        "largest_p_component": _max_component(p_mask),
    }
    record.update(shifts)
    record["shift_score"] = max(record["shift_h"], record["shift_v"])
    return record


def audit_batch(
    labels: np.ndarray,
    meta: list[dict[str, Any]] | None = None,
    train_all: np.ndarray | None = None,
    train_kp: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    records = [image_stats(item) for item in labels]
    for index, record in enumerate(records):
        record["index"] = index
        if meta is not None and index < len(meta):
            source = meta[index]
            record["source_id"] = source.get("id", index)
            record["source_path"] = source.get("path")
            record["mother"] = source.get("mother")
            record["sha256"] = source.get("sha256")
    if train_all is not None:
        details = nearest_train_details(labels, train_all)
        for index, record in enumerate(records):
            record["nearest_all"] = details["distances"][index]
            record["nearest_all_index"] = details["indices"][index]
    if train_kp is not None:
        details = nearest_train_details(labels, train_kp)
        for index, record in enumerate(records):
            record["nearest_kp"] = details["distances"][index]
            record["nearest_kp_index"] = details["indices"][index]
    return summarize(records)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    kp_records = [record for record in records if record["kp"]]
    values = lambda key: [record[key] for record in kp_records]
    median = lambda key: float(np.median(values(key))) if kp_records else None
    quartiles = lambda key: (
        [float(v) for v in np.percentile(values(key), [25, 75])] if kp_records else None
    )
    return {
        "num_samples": total,
        "kp_count": len(kp_records),
        "kp_hit_rate": len(kp_records) / total if total else 0.0,
        "pure_k_count": sum(1 for record in records if record["pure_k"]),
        "pure_p_count": sum(1 for record in records if record["pure_p"]),
        "other_label_count": sum(1 for record in records if record["other_labels"]),
        "kp_mothers": sorted(
            {record["mother"] for record in kp_records if record.get("mother")}
        ),
        "p_ratio_median": median("p_ratio"),
        "p_ratio_quartiles": quartiles("p_ratio"),
        "shift_median": median("shift_score"),
        "shift_quartiles": quartiles("shift_score"),
        "shift_h_median": median("shift_h"),
        "shift_v_median": median("shift_v"),
        "largest_p_component_median": median("largest_p_component"),
        "edge_density_median": median("edge_density"),
        "records": records,
    }


def prepare_kp(processed: str | Path, out: str | Path) -> dict[str, Any]:
    processed = Path(processed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"source": str(processed), "splits": {}}
    for split in ("train", "val"):
        labels = np.load(processed / f"{split}_labels.npy")
        meta = load_json(processed / f"{split}_meta.json")
        if len(labels) != len(meta):
            raise ValueError(f"{split}: {len(labels)} labels vs {len(meta)} meta")
        for index, record in enumerate(meta):
            digest = sha256_array(labels[index])
            if record.get("sha256") != digest:
                raise ValueError(f"{split}[{index}]: meta sha256 mismatch")
        mask = kp_mask(labels)
        rows = np.flatnonzero(mask).tolist()
        subset = labels[mask]
        subset_meta = [meta[index] for index in rows]
        np.save(out / f"{split}_labels.npy", subset.astype(np.uint8))
        save_json(out / f"{split}_meta.json", subset_meta)
        save_json(out / f"{split}_source_rows.json", rows)
        mothers = Counter(record.get("mother", "") for record in subset_meta)
        summary["splits"][split] = {
            "source_count": int(len(labels)),
            "kp_count": int(len(subset)),
            "source_rows": rows,
            "mother_count": len(mothers),
            "per_mother": dict(sorted(mothers.items())),
        }
    save_json(out / "summary.json", summary)
    return summary


def choose_representatives(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    labels = np.load(out / "train_labels.npy")
    meta = load_json(out / "train_meta.json")
    stats = [image_stats(item) for item in labels]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(meta):
        groups[record.get("mother", "")].append(index)
    chosen: list[dict[str, Any]] = []
    for mother in sorted(groups):
        indices = groups[mother]
        scores = [stats[index]["shift_score"] for index in indices]
        median = float(np.median(scores))
        representative = min(
            indices, key=lambda index: (abs(stats[index]["shift_score"] - median), index)
        )
        chosen.append(
            {
                "mother": mother,
                "index": representative,
                "id": len(chosen),
                "path": meta[representative].get("path"),
                "shift_score": stats[representative]["shift_score"],
                "p_ratio": stats[representative]["p_ratio"],
                "crop_count": len(indices),
                "crop_indices": indices,
            }
        )
    payload = {"rule": "closest shift_score to mother median, ties by source order", "representatives": chosen}
    save_json(out.parent / "audit" / "representatives.json", payload)
    return payload


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tile_sheet(
    labels: np.ndarray,
    ids: list[str],
    out: str | Path,
    cols: int = 9,
    scale: int = 8,
    title: str | None = None,
) -> None:
    labels = np.asarray(labels)
    side = int(labels.shape[-1]) * scale
    pad = 4
    cap = 16
    band = 26 if title else 0
    rows = ceil(len(labels) / cols)
    canvas = Image.new(
        "RGB", (cols * side + (cols + 1) * pad, band + rows * (cap + side) + (rows + 1) * pad), (16, 16, 16)
    )
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad + 2, 5), title, fill=(240, 240, 240), font=_font(13))
    palette = palette_array(readable_palette())
    font = _font(11)
    for index, item in enumerate(labels):
        row, col = divmod(index, cols)
        x = pad + col * (side + pad)
        y = band + pad + row * (cap + side + pad)
        draw.text((x + 2, y + 1), str(ids[index]), fill=(220, 220, 220), font=font)
        canvas.paste(render_labels(item, palette, scale), (x, y + cap))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def ascii_sheet(
    labels: np.ndarray,
    ids: list[str],
    out: str | Path,
    stats: list[dict[str, Any]] | None = None,
) -> None:
    lines: list[str] = []
    for index, item in enumerate(np.asarray(labels)):
        lines.append(f"=== {ids[index]} ===")
        if stats is not None:
            record = stats[index]
            lines.append(
                " ".join(
                    [
                        f"shift={record['shift_score']:.3f}",
                        f"h={record['shift_h']:.3f}",
                        f"v={record['shift_v']:.3f}",
                        f"p={record['p_ratio']:.3f}",
                        f"comp={record['largest_p_component']}",
                        f"edge={record['edge_density']:.3f}",
                    ]
                )
            )
        for row in item:
            lines.append("".join("#" if int(value) == PACK_P else "." for value in row))
        lines.append("")
    Path(out).write_text("\n".join(lines), encoding="utf-8")


def build_blind_set(
    history: dict[str, Any],
    val: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_mother: dict[str, list[int]] = defaultdict(list)
    for record in val["records"]:
        if record["kp"]:
            by_mother[record.get("mother", "")].append(record["index"])
    target = sum(1 for record in history["records"] if record["kp"])
    picked: list[int] = []
    mothers = sorted(by_mother)
    while len(picked) < target:
        added = False
        for mother in mothers:
            if by_mother[mother] and len(picked) < target:
                picked.append(by_mother[mother].pop(0))
                added = True
        if not added:
            break
    sources = [("gen", record["index"]) for record in history["records"] if record["kp"]]
    sources += [("val", index) for index in picked]
    order = rng.permutation(len(sources))
    assignments = []
    for slot, source_index in enumerate(order):
        source, index = sources[int(source_index)]
        assignments.append(
            {
                "blind_id": f"B{slot + 1:02d}",
                "source": source,
                "index": int(index),
            }
        )
    return {"seed": seed, "assignments": assignments}


def _audit_history(audit_dir: Path, kp_out: Path) -> dict[str, Any]:
    history_labels = np.load(HISTORY_SAMPLE / "labels.npy")
    history_meta = read_jsonl(HISTORY_SAMPLE / "meta.jsonl")
    train_all = np.load(audit_dir.parent.parent / "11.12" / "processed" / "train_labels.npy")
    train_kp = np.load(kp_out / "train_labels.npy")
    result = audit_batch(history_labels, history_meta, train_all, train_kp)
    save_json(audit_dir / "history.json", result)
    return result


def run_audit(processed: str | Path, kp_out: str | Path, audit_out: str | Path) -> dict[str, Any]:
    processed = Path(processed)
    kp_out = Path(kp_out)
    audit_dir = Path(audit_out)
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_all = np.load(processed / "train_labels.npy")
    train_kp = np.load(kp_out / "train_labels.npy")
    val_kp = np.load(kp_out / "val_labels.npy")
    val_kp_meta = load_json(kp_out / "val_meta.json")
    history_labels = np.load(HISTORY_SAMPLE / "labels.npy")
    history = _audit_history(audit_dir, kp_out)
    train = audit_batch(train_kp, load_json(kp_out / "train_meta.json"), train_all, train_kp)
    val = audit_batch(val_kp, val_kp_meta, train_all, train_kp)
    save_json(audit_dir / "train_kp.json", train)
    save_json(audit_dir / "val_kp.json", val)
    representatives = choose_representatives(kp_out)
    rep_labels = np.stack(
        [train_kp[item["index"]] for item in representatives["representatives"]]
    )
    rep_ids = [f"M{item['id']:03d}" for item in representatives["representatives"]]
    for start in range(0, len(rep_labels), 54):
        end = min(start + 54, len(rep_labels))
        tile_sheet(
            rep_labels[start:end],
            rep_ids[start:end],
            audit_dir / f"representatives_{start // 54 + 1}.png",
            title=f"KP TRAIN MOTHER REPRESENTATIVES {start + 1}-{end}",
        )
    blind = build_blind_set(history, val)
    save_json(audit_dir / "blind_key.json", blind)
    blind_labels = []
    for assignment in blind["assignments"]:
        if assignment["source"] == "gen":
            blind_labels.append(history_labels[assignment["index"]])
        else:
            blind_labels.append(val_kp[assignment["index"]])
    tile_sheet(
        np.stack(blind_labels),
        [item["blind_id"] for item in blind["assignments"]],
        audit_dir / "blind_sheet.png",
        cols=9,
        scale=10,
        title="BLIND REVIEW: 18 GENERATED + 18 REAL K/P",
    )
    summary = {
        "history": {key: value for key, value in history.items() if key != "records"},
        "train_kp": {key: value for key, value in train.items() if key != "records"},
        "val_kp": {key: value for key, value in val.items() if key != "records"},
        "blind_set_size": len(blind["assignments"]),
    }
    save_json(audit_dir / "summary.json", summary)
    return summary


def run_ascii(kp_out: str | Path, audit_out: str | Path) -> None:
    kp_out = Path(kp_out)
    audit_dir = Path(audit_out)
    reps = load_json(audit_dir / "representatives.json")["representatives"]
    train_kp = np.load(kp_out / "train_labels.npy")
    items = [train_kp[item["index"]] for item in reps]
    ids = ["M%03d" % item["id"] for item in reps]
    stats = [image_stats(item) for item in items]
    ascii_sheet(items[:54], ids[:54], audit_dir / "ascii_reps_1.txt", stats[:54])
    ascii_sheet(items[54:], ids[54:], audit_dir / "ascii_reps_2.txt", stats[54:])

    blind = load_json(audit_dir / "blind_key.json")["assignments"]
    history_labels = np.load(HISTORY_SAMPLE / "labels.npy")
    val_kp = np.load(kp_out / "val_labels.npy")
    blind_items = []
    for assignment in blind:
        if assignment["source"] == "gen":
            blind_items.append(history_labels[assignment["index"]])
        else:
            blind_items.append(val_kp[assignment["index"]])
    ascii_sheet(
        blind_items,
        [item["blind_id"] for item in blind],
        audit_dir / "ascii_blind.txt",
    )

    history = load_json(audit_dir / "history.json")
    kp_records = [record for record in history["records"] if record["kp"]]
    gen_items = [history_labels[record["index"]] for record in kp_records]
    gen_ids = ["G%02d" % (index + 1) for index in range(len(kp_records))]
    ascii_sheet(gen_items, gen_ids, audit_dir / "ascii_history_kp.txt", kp_records)


def run_val_ascii(kp_out: str | Path, audit_out: str | Path) -> None:
    kp_out = Path(kp_out)
    audit_dir = Path(audit_out)
    labels = np.load(kp_out / "val_labels.npy")
    meta = load_json(kp_out / "val_meta.json")
    stats = [image_stats(item) for item in labels]
    ids = ["V%03d" % index for index in range(len(labels))]
    index = [
        {"id": ids[i], "source_index": i, "path": meta[i].get("path"), "mother": meta[i].get("mother")}
        for i in range(len(labels))
    ]
    save_json(audit_dir / "val_ascii_index.json", index)
    ascii_sheet(labels[:51], ids[:51], audit_dir / "ascii_val_1.txt", stats[:51])
    ascii_sheet(labels[51:], ids[51:], audit_dir / "ascii_val_2.txt", stats[51:])


def audit_run(
    labels_path: str | Path,
    out: str | Path,
    train_all: str | Path | None = None,
    train_kp: str | Path | None = None,
) -> dict[str, Any]:
    labels_path = Path(labels_path)
    labels = np.load(labels_path)
    meta_path = labels_path.parent / "meta.jsonl"
    meta = read_jsonl(meta_path) if meta_path.is_file() else None
    train_all_array = np.load(train_all) if train_all else None
    train_kp_array = np.load(train_kp) if train_kp else None
    result = audit_batch(
        labels,
        meta=meta,
        train_all=train_all_array,
        train_kp=train_kp_array,
    )
    result["labels_path"] = str(labels_path)
    result["train_all_path"] = str(train_all) if train_all else None
    result["train_kp_path"] = str(train_kp) if train_kp else None
    save_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument("--val-ascii", action="store_true")
    parser.add_argument("--audit-run", action="store_true")
    parser.add_argument("--processed", default=None)
    parser.add_argument("--kp-out", default="outputs/knit_dit/11.23_kp/processed_kp")
    parser.add_argument("--audit-out", default="outputs/knit_dit/11.23_kp/audit")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--train-all", default=None)
    parser.add_argument("--train-kp", default=None)
    args = parser.parse_args()
    if args.prepare or args.audit:
        if not args.processed:
            parser.error("--prepare/--audit requires --processed")
    if args.prepare:
        save = prepare_kp(args.processed, args.kp_out)
        print({key: value["kp_count"] for key, value in save["splits"].items()})
    if args.audit:
        print(run_audit(args.processed, args.kp_out, args.audit_out))
    if args.ascii:
        run_ascii(args.kp_out, args.audit_out)
    if args.val_ascii:
        run_val_ascii(args.kp_out, args.audit_out)
    if args.audit_run:
        if not args.labels or not args.out:
            parser.error("--audit-run requires --labels and --out")
        result = audit_run(args.labels, args.out, args.train_all, args.train_kp)
        print(
            {
                key: result[key]
                for key in (
                    "num_samples",
                    "kp_count",
                    "kp_hit_rate",
                    "p_ratio_median",
                    "shift_median",
                )
            }
        )


if __name__ == "__main__":
    main()
