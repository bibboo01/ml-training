"""Small generated image datasets to try the vision backend offline (always the same, seed 42).

shapes      classify: circle / square / triangle (900 images, 64x64)
shapes-seg  segment:  images/ + masks/ with background / circle / square pixels (300 images)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ml_training.utils import ensure_dir, log


def _canvas(rng: np.random.Generator, size: int, noise: float = 25) -> np.ndarray:
    base = rng.integers(0, 255, 3)
    img = np.clip(base + rng.normal(0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0) if noise > 25 else cv2.GaussianBlur(img, (5, 5), 0)


def _draw(img: np.ndarray, kind: str, rng: np.random.Generator, value=None) -> None:
    s = img.shape[0]
    color = tuple(int(c) for c in rng.integers(0, 255, 3))  # always drawn: same shape for the mask
    if value is not None:
        color = value
    r = int(rng.integers(s // 6, s // 3))
    cx, cy = (int(v) for v in rng.integers(r + 2, s - r - 2, 2))
    if kind == "circle":
        cv2.circle(img, (cx, cy), r, color, -1)
    elif kind == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], color)


def _shapes(dst: Path, rng: np.random.Generator) -> None:
    for kind in ("circle", "square", "triangle"):
        folder = ensure_dir(dst / kind)
        for i in range(300):
            img = _canvas(rng, 64)
            _draw(img, kind, rng)
            cv2.imwrite(str(folder / f"{kind}_{i:03d}.png"), img)


def _shapes_seg(dst: Path, rng: np.random.Generator) -> None:
    idir, mdir = ensure_dir(dst / "images"), ensure_dir(dst / "masks")
    (dst / "classes.txt").write_text("background\ncircle\nsquare\n", encoding="utf-8")
    for i in range(300):
        img = _canvas(rng, 128, noise=45)  # textured background, flat shapes
        mask = np.zeros((128, 128), np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            kind = ("circle", "square")[int(rng.integers(0, 2))]
            seed = int(rng.integers(0, 1 << 30))
            _draw(img, kind, np.random.default_rng(seed))
            _draw(mask, kind, np.random.default_rng(seed), value=1 if kind == "circle" else 2)
        cv2.imwrite(str(idir / f"{i:03d}.png"), img)
        cv2.imwrite(str(mdir / f"{i:03d}.png"), mask)


SAMPLES = {"shapes": _shapes, "shapes-seg": _shapes_seg}


def make(name: str, dst: Path) -> Path:
    if name not in SAMPLES:
        raise KeyError(f"Unknown image sample {name!r}: {list(SAMPLES)}")
    SAMPLES[name](Path(dst), np.random.default_rng(42))
    log.info("Sample dataset %s written to %s", name, dst)
    return Path(dst)
