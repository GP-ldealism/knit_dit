import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from knit.io import (
    labels_to_onehot,
    load_json,
    read_labels,
    read_palette,
    save_json,
    sha256_array,
)


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"
    )


def mother_id(path: str | Path, regex: str | None = None) -> str:
    stem = Path(path).stem
    if regex:
        match = re.match(regex, stem)
        if match:
            return match.group(1)
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def audit(
    root: str | Path,
    num_classes: int = 17,
    regex: str | None = None,
    out: str | Path | None = None,
) -> dict:
    root = Path(root)
    files = iter_images(root)
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    label_pixels: Counter[int] = Counter()
    label_files: Counter[int] = Counter()
    mothers: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    unreadable: list[dict[str, str]] = []
    examples: list[str] = []

    for path in files:
        try:
            with Image.open(path) as img:
                modes[img.mode] += 1
                sizes[f"{img.width}x{img.height}"] += 1
                if img.mode == "P":
                    labels = np.array(img, dtype=np.uint8)
                    present = np.unique(labels)
                    for value in present:
                        label_pixels[int(value)] += int((labels == value).sum())
                        label_files[int(value)] += 1
                    hashes[sha256_array(labels)].append(_relative(path, root))
                else:
                    labels = None
            mothers[mother_id(path, regex)] += 1
            if len(examples) < 20:
                examples.append(_relative(path, root))
        except Exception as exc:
            unreadable.append({"path": _relative(path, root), "error": repr(exc)})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    summary = {
        "root": str(root),
        "file_count": len(files),
        "num_classes": num_classes,
        "modes": dict(modes),
        "sizes": dict(sizes),
        "mother_count": len(mothers),
        "mother_examples": sorted(mothers)[:20],
        "mother_file_counts": dict(sorted(mothers.items())),
        "label_pixels": {str(k): v for k, v in sorted(label_pixels.items())},
        "label_files": {str(k): v for k, v in sorted(label_files.items())},
        "label_min": min(label_pixels) if label_pixels else None,
        "label_max": max(label_pixels) if label_pixels else None,
        "duplicate_groups": duplicate_groups,
        "duplicate_file_count": sum(len(g) for g in duplicate_groups),
        "unreadable": unreadable,
        "examples": examples,
    }
    if out is not None:
        save_json(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _split_mothers(
    mothers: list[str], ratios: tuple[float, float, float], seed: int
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(mothers)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], [], []
    if n == 1:
        return shuffled, [], []
    if n == 2:
        return shuffled[:1], shuffled[1:], []
    n_val = max(1, int(round(n * ratios[1])))
    n_test = max(1, int(round(n * ratios[2])))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_splits(
    root: str | Path,
    out: str | Path,
    num_classes: int = 17,
    regex: str | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    root = Path(root)
    files = iter_images(root)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[mother_id(path, regex)].append(path)

    train_m, val_m, test_m = _split_mothers(sorted(groups), ratios, seed)
    split_mothers = {"train": train_m, "val": val_m, "test": test_m}
    split_files = {
        name: [p for mother in names for p in groups[mother]]
        for name, names in split_mothers.items()
    }

    skipped: list[dict[str, str]] = []
    duplicates_within: dict[str, list[str]] = {}
    kept_hashes: dict[str, dict[str, str]] = {}
    kept_files: dict[str, list[str]] = {}
    file_mothers: dict[str, str] = {}

    for name in ("train", "val", "test"):
        seen: dict[str, str] = {}
        kept: list[str] = []
        duplicates: list[str] = []
        for path in split_files[name]:
            rel = _relative(path, root)
            try:
                labels = read_labels(path, num_classes)
            except Exception as exc:
                skipped.append({"path": rel, "error": repr(exc)})
                continue
            digest = sha256_array(labels)
            if digest in seen:
                duplicates.append(rel)
                continue
            seen[digest] = rel
            kept.append(rel)
            file_mothers[rel] = mother_id(path, regex)
        kept_hashes[name] = seen
        kept_files[name] = kept
        duplicates_within[name] = duplicates

    duplicates_cross: list[dict[str, str]] = []
    train_hashes = kept_hashes["train"]
    for name in ("val", "test"):
        filtered: list[str] = []
        for rel in kept_files[name]:
            digest = sha256_array(read_labels(root / rel, num_classes))
            if digest in train_hashes:
                duplicates_cross.append(
                    {"split": name, "path": rel, "conflict_with": train_hashes[digest]}
                )
                continue
            filtered.append(rel)
        kept_files[name] = filtered
        kept_hashes[name] = {
            sha256_array(read_labels(root / rel, num_classes)): rel for rel in filtered
        }

    val_hashes = kept_hashes["val"]
    filtered_test: list[str] = []
    for rel in kept_files["test"]:
        digest = sha256_array(read_labels(root / rel, num_classes))
        if digest in val_hashes:
            duplicates_cross.append(
                {"split": "test", "path": rel, "conflict_with": val_hashes[digest]}
            )
            continue
        filtered_test.append(rel)
    kept_files["test"] = filtered_test

    summary = {
        "root": str(root),
        "seed": seed,
        "ratios": list(ratios),
        "num_classes": num_classes,
        "mother_regex": regex,
        "mothers": split_mothers,
        "files": kept_files,
        "file_mothers": file_mothers,
        "stats": {
            "mother_counts": {k: len(v) for k, v in split_mothers.items()},
            "file_counts": {k: len(v) for k, v in kept_files.items()},
            "duplicates_within": duplicates_within,
            "duplicates_cross": duplicates_cross,
        },
        "skipped": skipped,
    }
    save_json(out, summary)
    print(json.dumps(summary["stats"], ensure_ascii=False, indent=2))
    return summary


def _center_crop(labels: np.ndarray, size: int) -> np.ndarray:
    h, w = labels.shape
    if h < size or w < size:
        raise ValueError(f"cannot crop {h}x{w} to {size}x{size}")
    top = (h - size) // 2
    left = (w - size) // 2
    return labels[top : top + size, left : left + size]


def encode(
    root: str | Path,
    splits_path: str | Path,
    out: str | Path,
    num_classes: int = 17,
    image_size: int = 20,
    center_crop: int | None = None,
) -> dict:
    root = Path(root)
    splits = load_json(splits_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    palette: list[int] | None = None
    stats: dict[str, int] = {}

    for name in ("train", "val", "test"):
        arrays: list[np.ndarray] = []
        meta: list[dict] = []
        for rel in splits["files"][name]:
            path = root / rel
            labels = read_labels(path, num_classes)
            if labels.shape != (image_size, image_size):
                if center_crop is None:
                    raise ValueError(
                        f"{path}: shape {labels.shape}, expected "
                        f"{(image_size, image_size)}; use --center-crop"
                    )
                labels = _center_crop(labels, center_crop)
            if labels.shape != (image_size, image_size):
                raise ValueError(f"{path}: shape {labels.shape} after crop")
            arrays.append(labels)
            meta.append(
                {
                    "path": rel,
                    "mother": splits["file_mothers"].get(rel, ""),
                    "shape": list(labels.shape),
                    "sha256": sha256_array(labels),
                }
            )
            if palette is None:
                palette = read_palette(path)
        labels_array = (
            np.stack(arrays).astype(np.uint8)
            if arrays
            else np.zeros((0, image_size, image_size), dtype=np.uint8)
        )
        np.save(out / f"{name}_labels.npy", labels_array)
        save_json(out / f"{name}_meta.json", meta)
        stats[name] = len(arrays)

    save_json(out / "palette.json", {"palette": palette})
    summary = {"out": str(out), "counts": stats, "image_size": image_size}
    save_json(out / "encode_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


class KnitDataset(Dataset):
    def __init__(
        self,
        labels_path: str | Path,
        meta_path: str | Path | None = None,
        num_classes: int = 17,
    ) -> None:
        self.labels = np.load(labels_path)
        self.meta = load_json(meta_path) if meta_path is not None else []
        self.num_classes = num_classes

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict:
        onehot = labels_to_onehot(self.labels[index], self.num_classes)
        return {
            "x0": torch.from_numpy(onehot),
            "y": torch.tensor(0, dtype=torch.long),
            "index": index,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--splits", default="data/nik/splits/splits.json")
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--center-crop", type=int, default=None)
    parser.add_argument("--mother-regex", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.audit:
        audit(
            args.root,
            num_classes=args.num_classes,
            regex=args.mother_regex,
            out=args.out or "data/nik/audit.json",
        )
    if args.split:
        build_splits(
            args.root,
            out=args.out or args.splits,
            num_classes=args.num_classes,
            regex=args.mother_regex,
            seed=args.seed,
        )
    if args.encode:
        encode(
            args.root,
            splits_path=args.splits,
            out=args.out or "data/nik/processed",
            num_classes=args.num_classes,
            image_size=args.image_size,
            center_crop=args.center_crop,
        )


if __name__ == "__main__":
    main()
