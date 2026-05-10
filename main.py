# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — modular CLI (dataset analysis)
# Author: [Your Name]
# University: [Your University]
# Purpose: Phase-1 dataset scanning and figure export for the thesis pipeline.
# -----------------------------------------------------------------------------
"""Brain MRI thesis pipeline CLI (dataset analysis phase)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data_loader import class_counts, scan_dataset


def ensure_output_dirs(outputs_dir: Path) -> tuple[Path, Path]:
    figures_dir = Path(outputs_dir) / "figures"
    metrics_dir = Path(outputs_dir) / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, metrics_dir


def plot_class_distribution(counts: dict[str, int], *, out_path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title("Class distribution (image count per class)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sample_images(
    records,
    *,
    samples_per_class: int,
    out_path: Path,
) -> None:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for r in records:
        if len(by_class[r.label]) < samples_per_class:
            by_class[r.label].append(r.path)

    class_names = sorted(by_class.keys())
    n_rows = len(class_names)
    n_cols = samples_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, cls in enumerate(class_names):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(by_class[cls]):
                continue

            img_path = by_class[cls][j]
            with Image.open(img_path) as img:
                vis = img.convert("RGB")
            ax.imshow(vis)
            if j == 0:
                ax.set_title(cls, loc="left", fontsize=12, fontweight="bold")

    fig.suptitle("Sample images per class", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_resolution_analysis(records, *, out_path_scatter: Path, out_path_hist: Path) -> None:
    widths = np.array([r.width for r in records], dtype=np.int32)
    heights = np.array([r.height for r in records], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(widths, heights, s=10, alpha=0.5)
    ax.set_title("Image resolution scatter (width vs height)")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path_scatter, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(widths, bins=30, alpha=0.6, label="width")
    ax.hist(heights, bins=30, alpha=0.6, label="height")
    ax.set_title("Image resolution histogram")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_hist, dpi=200)
    plt.close(fig)


def plot_grayscale_vs_rgb(records, *, out_path: Path) -> None:
    gray = sum(1 for r in records if r.channels == 1)
    rgb = sum(1 for r in records if r.channels == 3)
    other = len(records) - gray - rgb

    labels = ["Grayscale (1ch)", "RGB (3ch)", "Other"]
    values = [gray, rgb, other]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_title("Image color mode distribution")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def phase1_analyze_dataset(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir, _ = ensure_output_dirs(outputs_dir)

    records = scan_dataset(data_dir, max_images=args.max_images)
    if len(records) == 0:
        raise RuntimeError(
            f"No images found under {data_dir}. "
            "Expected folder-per-class structure: data/ClassName/image.png"
        )

    counts = class_counts(records)

    plot_class_distribution(counts, out_path=figures_dir / "class_distribution.png")
    plot_sample_images(
        records,
        samples_per_class=args.samples_per_class,
        out_path=figures_dir / "sample_images_per_class.png",
    )
    plot_resolution_analysis(
        records,
        out_path_scatter=figures_dir / "resolution_scatter.png",
        out_path_hist=figures_dir / "resolution_histogram.png",
    )
    plot_grayscale_vs_rgb(records, out_path=figures_dir / "grayscale_vs_rgb.png")

    summary_path = figures_dir / "dataset_summary.txt"
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    gray = sum(1 for r in records if r.channels == 1)
    rgb = sum(1 for r in records if r.channels == 3)
    other = len(records) - gray - rgb
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset directory: {data_dir}\n")
        f.write(f"Images scanned: {len(records)}\n")
        f.write(f"Classes: {len(counts)}\n\n")
        f.write("Class counts:\n")
        for k in sorted(counts.keys()):
            f.write(f"- {k}: {counts[k]}\n")
        f.write("\nResolution stats (pixels):\n")
        f.write(f"- width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}\n")
        f.write(f"- height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}\n")
        f.write("\nChannel distribution:\n")
        f.write(f"- grayscale(1ch): {gray}\n")
        f.write(f"- rgb(3ch):       {rgb}\n")
        f.write(f"- other:          {other}\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brain MRI Image Classification — modular thesis pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="PHASE 1: dataset analysis + figures")
    p_analyze.add_argument("--data_dir", type=str, default="data", help="Dataset root directory")
    p_analyze.add_argument("--outputs_dir", type=str, default="outputs", help="Outputs directory")
    p_analyze.add_argument(
        "--samples_per_class",
        type=int,
        default=4,
        help="Number of sample images to show per class",
    )
    p_analyze.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on scanned images (for quick exploration)",
    )
    p_analyze.set_defaults(func=phase1_analyze_dataset)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

