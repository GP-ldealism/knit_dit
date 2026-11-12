import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from knit.encoding import LabelSpec


def default_palette() -> list[int]:
    palette: list[int] = []
    for i in range(256):
        palette.extend(((i * 37) % 256, (i * 73) % 256, (i * 109) % 256))
    return palette


def read_palette(path: str | Path) -> list[int] | None:
    with Image.open(path) as img:
        if img.mode != "P":
            return None
        palette = img.getpalette()
    if palette is None:
        return None
    palette = list(palette)
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))
    return palette[:768]


def read_labels(path: str | Path, num_classes: int = 17) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode != "P":
            raise ValueError(f"{path}: expected P mode, got {img.mode}")
        labels = np.array(img, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError(f"{path}: expected 2D label image, got {labels.shape}")
    if labels.size and int(labels.max()) >= num_classes:
        raise ValueError(
            f"{path}: label {int(labels.max())} outside 0..{num_classes - 1}"
        )
    return labels


def save_labels_png(
    labels: np.ndarray,
    path: str | Path,
    palette: list[int] | None = None,
) -> None:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError(f"expected 2D labels, got {labels.shape}")
    img = Image.fromarray(labels, mode="P")
    img.putpalette(default_palette() if palette is None else palette)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def labels_to_onehot(labels: np.ndarray, num_classes: int = 17) -> np.ndarray:
    return LabelSpec(num_classes=num_classes).encode(labels)


def onehot_to_labels(x: Any) -> np.ndarray:
    return LabelSpec().decode(x)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
