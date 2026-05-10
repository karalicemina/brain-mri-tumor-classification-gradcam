from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """
    Lightweight metadata for a single image file.

    We keep this intentionally small for analysis pipelines, so we can scan large
    datasets without loading pixel arrays into RAM.
    """

    path: Path
    label: str
    width: int
    height: int
    channels: int  # 1 (grayscale) or 3 (RGB) or other (rare)
    mode: str      # PIL mode string (e.g., "L", "RGB", "RGBA")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_class_folders(data_dir: Path) -> list[Path]:
    """
    Return class folders under `data_dir` (folder-per-class dataset layout).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    class_dirs.sort(key=lambda p: p.name.lower())
    return class_dirs


def iter_image_paths(data_dir: Path) -> Iterable[tuple[Path, str]]:
    """
    Yield (image_path, class_label) for a folder-per-class dataset layout.
    """
    for class_dir in list_class_folders(data_dir):
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if _is_image_file(p):
                yield p, label


def read_image_metadata(path: Path) -> tuple[int, int, int, str]:
    """
    Read image metadata (width, height, channels, PIL mode) without keeping pixels.
    """
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode

        # Map PIL mode to a channel count for our analysis.
        # - "L": grayscale (1 channel)
        # - "RGB": 3
        # - "RGBA": 4 (rare in MRI datasets)
        # - other modes exist; we keep a best-effort mapping.
        if mode == "L":
            channels = 1
        elif mode == "RGB":
            channels = 3
        elif mode == "RGBA":
            channels = 4
        else:
            # Fallback: attempt to infer from bands length.
            try:
                channels = len(img.getbands())
            except Exception:
                channels = -1

    return width, height, channels, mode


def scan_dataset(data_dir: Path, *, max_images: int | None = None) -> list[ImageRecord]:
    """
    Scan dataset and return a list of ImageRecord metadata objects.

    Parameters
    ----------
    data_dir:
        Dataset directory with folder-per-class layout.
    max_images:
        Optional cap to speed up analysis on very large datasets.
        If None, scan all images.
    """
    records: list[ImageRecord] = []
    for i, (path, label) in enumerate(iter_image_paths(data_dir)):
        if max_images is not None and i >= max_images:
            break
        w, h, c, mode = read_image_metadata(path)
        records.append(ImageRecord(path=path, label=label, width=w, height=h, channels=c, mode=mode))
    return records


def class_counts(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.label] = counts.get(r.label, 0) + 1
    return counts

