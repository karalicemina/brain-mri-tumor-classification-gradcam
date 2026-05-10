# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI dataset — exploratory analysis and figures
# Author: [Your Name]
# University: [Your University]
# Purpose: Counts, resolution stats, and sample plots for thesis methodology.
# -----------------------------------------------------------------------------
"""Collect statistics and figures from processed class folders."""

import os

from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.paths import PROJECT_ROOT, ensure_project_directories


def get_image_paths_per_class(
    dataset_root: str,
    class_names: List[str],
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> Dict[str, List[str]]:
    image_paths_per_class: Dict[str, List[str]] = {}

    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Class directory not found: {class_dir}")
            image_paths_per_class[class_name] = []
            continue

        class_image_paths: List[str] = []
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith(valid_extensions):
                    class_image_paths.append(os.path.join(root, fname))

        image_paths_per_class[class_name] = sorted(class_image_paths)
        print(f"[INFO] Found {len(class_image_paths)} images in class '{class_name}'.")

    return image_paths_per_class


def print_dataset_statistics(image_paths_per_class: Dict[str, List[str]]) -> None:
    print("\n========== DATASET STATISTICS ==========")
    total_images = sum(len(v) for v in image_paths_per_class.values())

    for class_name, paths in image_paths_per_class.items():
        count = len(paths)
        percentage = (count / total_images * 100.0) if total_images > 0 else 0.0
        print(f"Class '{class_name}': {count} images ({percentage:.2f}%)")

    print(f"\nTotal images in dataset: {total_images}")
    print("========================================\n")


def plot_class_distribution(
    image_paths_per_class: Dict[str, List[str]],
    output_dir: str,
    figure_name: str = "class_distribution.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    class_names = list(image_paths_per_class.keys())
    counts = [len(image_paths_per_class[c]) for c in class_names]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, counts, color=["#4C72B0", "#55A868", "#C44E52"])

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    output_path = os.path.join(output_dir, figure_name)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved class distribution plot to: {output_path}")

    plt.show()
    plt.close()


def show_sample_images(
    image_paths_per_class: Dict[str, List[str]],
    num_samples: int,
    output_dir: str,
    figure_name: str = "sample_images.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    class_names = list(image_paths_per_class.keys())
    num_classes = len(class_names)

    if num_classes == 0:
        print("[WARNING] No classes found when trying to display sample images.")
        return

    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=num_samples,
        figsize=(3 * num_samples, 3 * num_classes),
    )

    if num_classes == 1 and num_samples == 1:
        axes = np.array([[axes]])
    elif num_classes == 1:
        axes = np.array([axes])
    elif num_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, class_name in enumerate(class_names):
        paths = image_paths_per_class[class_name]
        if len(paths) == 0:
            print(f"[WARNING] No images for class '{class_name}' to display.")
            continue

        selected_paths = paths[:num_samples]

        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(selected_paths):
                continue

            img_path = selected_paths[col_idx]
            img_bgr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue

            ax.imshow(img_bgr, cmap="gray")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, figure_name)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved sample images figure to: {output_path}")

    plt.show()
    plt.close(fig)


def compute_resolution_statistics(
    image_paths_per_class: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    print("\n========== IMAGE RESOLUTION STATISTICS ==========")

    for class_name, paths in image_paths_per_class.items():
        widths: List[int] = []
        heights: List[int] = []

        for img_path in paths:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARNING] Could not read image for resolution stats: {img_path}")
                continue

            if img.ndim == 2:
                h, w = img.shape
            else:
                h, w = img.shape[:2]

            widths.append(w)
            heights.append(h)

        if len(widths) == 0 or len(heights) == 0:
            print(f"Class '{class_name}': no valid images for resolution statistics.")
            continue

        w_array = np.array(widths)
        h_array = np.array(heights)

        class_stats = {
            "min_width": float(w_array.min()),
            "max_width": float(w_array.max()),
            "mean_width": float(w_array.mean()),
            "min_height": float(h_array.min()),
            "max_height": float(h_array.max()),
            "mean_height": float(h_array.mean()),
        }

        stats[class_name] = class_stats

        print(
            f"Class '{class_name}': "
            f"width (min/mean/max) = {class_stats['min_width']:.0f}/"
            f"{class_stats['mean_width']:.1f}/"
            f"{class_stats['max_width']:.0f}, "
            f"height (min/mean/max) = {class_stats['min_height']:.0f}/"
            f"{class_stats['mean_height']:.1f}/"
            f"{class_stats['max_height']:.0f}"
        )

    print("===============================================\n")
    return stats


def main() -> None:
    ensure_project_directories()

    dataset_root = PROJECT_ROOT / "data" / "processed"
    output_figures_dir = PROJECT_ROOT / "outputs" / "figures"
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["glioma", "meningioma", "pituitary"]

    ds = str(dataset_root)
    out = str(output_figures_dir)

    image_paths_per_class = get_image_paths_per_class(ds, class_names)
    print_dataset_statistics(image_paths_per_class)
    plot_class_distribution(image_paths_per_class, out)
    show_sample_images(
        image_paths_per_class,
        num_samples=5,
        output_dir=out,
        figure_name="sample_mri_images.png",
    )
    compute_resolution_statistics(image_paths_per_class)


if __name__ == "__main__":
    main()
