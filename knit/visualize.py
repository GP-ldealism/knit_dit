import argparse
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from knit.encoding import CLASS_NAMES
from knit.io import default_palette, load_json


def _font(size: int = 12) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def palette_array(palette: Any) -> np.ndarray:
    if palette is None:
        palette = default_palette()
    arr = np.asarray(palette, dtype=np.uint8).reshape(-1)
    if arr.size < 768:
        arr = np.concatenate([arr, np.zeros(768 - arr.size, dtype=np.uint8)])
    return arr[:768].reshape(256, 3)


READABLE_COLORS = (
    (250, 250, 250),
    (70, 70, 70),
    (180, 180, 120),
    (255, 165, 0),
    (0, 90, 255),
    (0, 160, 255),
    (0, 200, 0),
    (0, 130, 0),
    (150, 0, 255),
    (200, 0, 255),
    (255, 0, 0),
    (200, 0, 0),
    (255, 0, 255),
    (255, 0, 128),
    (0, 255, 255),
    (0, 180, 180),
    (255, 255, 0),
)


def readable_palette(num_classes: int = 17) -> list[int]:
    palette = [0] * 768
    for index in range(min(num_classes, len(READABLE_COLORS))):
        palette[index * 3 : index * 3 + 3] = list(READABLE_COLORS[index])
    return palette


def render_labels(labels: np.ndarray, palette: np.ndarray, scale: int) -> Image.Image:
    rgb = palette[np.asarray(labels, dtype=np.uint8)]
    img = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def _montage(
    images: list[Image.Image], cols: int, cell: int, gap: int, bg: tuple[int, int, int]
) -> Image.Image:
    rows = ceil(len(images) / cols)
    width = cols * cell + (cols + 1) * gap
    height = rows * cell + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), bg)
    for index, img in enumerate(images):
        row, col = divmod(index, cols)
        x = gap + col * (cell + gap)
        y = gap + row * (cell + gap)
        canvas.paste(img, (x, y))
    return canvas


def _block(
    title: str,
    images: list[Image.Image],
    cols: int,
    cell: int,
    gap: int,
    bg: tuple[int, int, int],
    font: Any,
) -> Image.Image:
    grid = _montage(images, cols, cell, gap, bg)
    band_h = 22
    band = Image.new("RGB", (grid.width, band_h), (24, 24, 24))
    draw = ImageDraw.Draw(band)
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (grid.width, band_h + grid.height), bg)
    out.paste(band, (0, 0))
    out.paste(grid, (0, band_h))
    return out


def _legend(
    palette: np.ndarray,
    num_classes: int,
    font: Any,
    swatch: int = 16,
    pad: int = 6,
) -> Image.Image:
    names = list(CLASS_NAMES[:num_classes])
    cell_w = swatch + 6 + 70
    per_row = max(1, 8)
    rows = ceil(len(names) / per_row)
    width = per_row * cell_w
    height = rows * (swatch + pad) + pad
    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    for index, name in enumerate(names):
        row, col = divmod(index, per_row)
        x = col * cell_w + pad
        y = row * (swatch + pad) + pad
        color = tuple(int(v) for v in palette[index])
        draw.rectangle([x, y, x + swatch, y + swatch], fill=color, outline=(80, 80, 80))
        draw.text((x + swatch + 6, y + 2), f"{index}:{name}", fill=(230, 230, 230), font=font)
    return img


def _hcat(left: Image.Image, right: Image.Image, gap: int, bg: tuple[int, int, int]) -> Image.Image:
    height = max(left.height, right.height)
    width = left.width + gap + right.width
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def _vcat(top: Image.Image, bottom: Image.Image, gap: int, bg: tuple[int, int, int]) -> Image.Image:
    width = max(top.width, bottom.width)
    height = top.height + gap + bottom.height
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height + gap))
    return canvas


def build(
    train_labels: np.ndarray,
    gen_labels: np.ndarray,
    palette: np.ndarray,
    num: int,
    cols: int,
    scale: int,
    gap: int,
    seed: int,
    train_indices: list[int] | None,
    layout: str,
    legend: bool,
    num_classes: int,
) -> Image.Image:
    bg = (16, 16, 16)
    font = _font(12)
    cell = int(train_labels.shape[-1]) * scale

    if train_indices is None:
        rng = np.random.default_rng(seed)
        pool = train_labels.shape[0]
        count = min(num, pool)
        train_indices = rng.choice(pool, size=count, replace=False).tolist()
    else:
        train_indices = train_indices[:num]

    gen_count = min(num, gen_labels.shape[0])
    gen_slice = gen_labels[:gen_count]

    train_imgs = [render_labels(train_labels[i], palette, scale) for i in train_indices]
    gen_imgs = [render_labels(gen_slice[i], palette, scale) for i in range(gen_count)]

    train_block = _block(
        f"TRAIN  n={len(train_imgs)}", train_imgs, cols, cell, gap, bg, font
    )
    gen_block = _block(
        f"GENERATED  n={len(gen_imgs)}", gen_imgs, cols, cell, gap, bg, font
    )

    if layout == "side":
        canvas = _hcat(train_block, gen_block, gap * 3, bg)
    else:
        canvas = _vcat(train_block, gen_block, gap * 3, bg)

    if legend:
        strip = _legend(palette, num_classes, font)
        canvas = _vcat(canvas, strip, gap * 2, bg)
    return canvas


def _resolve_gen_labels(gen_labels: str | None, gen_dir: str | None) -> Path:
    if gen_labels:
        return Path(gen_labels)
    if gen_dir:
        base = Path(gen_dir)
        for candidate in (base / "labels.npy", base / "sample" / "labels.npy"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no labels.npy under {base}")
    raise ValueError("provide --gen-labels or --gen-dir")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--gen-labels", default=None)
    parser.add_argument("--gen-dir", default=None)
    parser.add_argument("--palette", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--train-indices", default=None)
    parser.add_argument("--layout", choices=["stack", "side"], default="stack")
    parser.add_argument(
        "--palette-mode", choices=["index", "readable"], default="index"
    )
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args()

    train_labels = np.load(args.train_labels)
    gen_labels = np.load(_resolve_gen_labels(args.gen_labels, args.gen_dir))
    palette = None
    if args.palette:
        payload = load_json(args.palette)
        palette = payload.get("palette") if isinstance(payload, dict) else payload
    if args.palette_mode == "readable":
        palette = readable_palette(args.num_classes)
    palette = palette_array(palette)

    train_indices = None
    if args.train_indices:
        train_indices = [int(item) for item in args.train_indices.split(",") if item != ""]

    canvas = build(
        train_labels=train_labels,
        gen_labels=gen_labels,
        palette=palette,
        num=args.num,
        cols=args.cols,
        scale=args.scale,
        gap=args.gap,
        seed=args.seed,
        train_indices=train_indices,
        layout=args.layout,
        legend=not args.no_legend,
        num_classes=args.num_classes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"wrote {out} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
