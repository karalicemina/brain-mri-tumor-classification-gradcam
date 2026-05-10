# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — Grad-CAM localization
# Author: [Your Name]
# University: [Your University]
# Purpose: Thesis interpretability pipeline (CAM, masks, qualitative figures).
# -----------------------------------------------------------------------------
"""Grad-CAM tumor localization for trained Keras MRI classifiers (ResNet50 / CNN)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.constants import CLASS_NAMES_THESIS
from utils.gradcam import compute_gradcam_heatmap, find_last_conv_layer
from utils.model_tools import ensure_model_built, resolve_model_path
from utils.paths import PROJECT_ROOT, ensure_project_directories
from utils.preprocessing import (
    create_tta_versions,
    load_and_preprocess_custom_image,
    load_and_preprocess_image,
)

_BATCH_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_class_names(dataset_dir: Path | str) -> List[str]:
    root = Path(dataset_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def select_one_image_path(dataset_dir: Path | str) -> Tuple[str, str]:
    class_names = get_class_names(dataset_dir)
    if not class_names:
        raise ValueError(f"No class subdirectories found in: {dataset_dir}")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    root = Path(dataset_dir)

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for fname in sorted(class_dir.iterdir()):
            if fname.suffix.lower() in valid_ext:
                return str(fname.resolve()), class_name

    raise ValueError(f"No image files found under class folders in: {dataset_dir}")


def collect_test_image_paths(test_dir: Path | str, limit: int = 30) -> List[Tuple[str, str]]:
    paths_with_class: List[Tuple[str, str]] = []
    root = Path(test_dir)

    for class_name in get_class_names(root):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for fname in sorted(class_dir.iterdir()):
            if fname.suffix.lower() in _BATCH_IMAGE_EXTS:
                paths_with_class.append((str(fname.resolve()), class_name))

    paths_with_class.sort(key=lambda x: x[0])
    return paths_with_class[:limit]


def collect_balanced_test_paths(
    test_dir: Path | str, per_class: int = 10
) -> List[Tuple[str, str]]:
    expected_classes = ("glioma", "meningioma", "pituitary")
    out: List[Tuple[str, str]] = []
    root = Path(test_dir)

    for class_name in expected_classes:
        class_dir = root / class_name
        if not class_dir.is_dir():
            print(f"[WARNING] Missing class folder (skipped): {class_dir}")
            continue
        paths = [
            str(p.resolve())
            for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in _BATCH_IMAGE_EXTS
        ]
        for p in paths[:per_class]:
            out.append((p, class_name))

    return out


def build_brain_region_mask_from_rgb(
    rgb: np.ndarray,
    border_shrink_frac: float = 0.03,
    intensity_thresh: int = 8,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, bw = cv2.threshold(gray, intensity_thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_u8, [largest], -1, 255, thickness=-1)

    er_k = np.ones((7, 7), np.uint8)
    mask_u8 = cv2.erode(mask_u8, er_k, iterations=1)

    m = mask_u8.astype(np.float32) / 255.0

    bx = max(1, int(border_shrink_frac * w))
    by = max(1, int(border_shrink_frac * h))
    border_gate = np.ones((h, w), dtype=np.float32)
    border_gate[:by, :] = 0.0
    border_gate[-by:, :] = 0.0
    border_gate[:, :bx] = 0.0
    border_gate[:, -bx:] = 0.0

    return m * border_gate


def _prepare_smoothed_heatmap(
    heatmap: np.ndarray,
    height: int,
    width: int,
    brain_mask: np.ndarray,
    gaussian_ksize: int = 9,
) -> np.ndarray:
    hm = cv2.resize(heatmap.astype(np.float32), (width, height))
    hm_min, hm_max = float(hm.min()), float(hm.max())
    hm = (hm - hm_min) / (hm_max - hm_min + 1e-8)
    k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
    hm = cv2.GaussianBlur(hm, (k, k), 0)
    return hm * brain_mask


def _percentile_threshold_mask(
    heatmap_smoothed: np.ndarray,
    brain_mask: np.ndarray,
    percentile: float,
    brain_gate: float = 0.05,
) -> Tuple[np.ndarray, float]:
    valid = heatmap_smoothed[brain_mask > brain_gate]
    if valid.size < 16:
        return np.zeros_like(heatmap_smoothed, dtype=np.uint8), 1.0
    thr = float(np.percentile(valid, percentile))
    binary_u8 = (
        ((heatmap_smoothed >= thr) & (brain_mask > brain_gate)).astype(np.uint8) * 255
    )
    return binary_u8, thr


def _cleanup_mask_and_pick_component(
    binary_u8: np.ndarray,
    heatmap_smoothed: np.ndarray,
    min_area_pixels: int,
    open_kernel: int = 3,
) -> Tuple[np.ndarray, int]:
    ksz = max(3, open_kernel)
    k = np.ones((ksz, ksz), np.uint8)
    opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(binary_u8), -1

    best_label = -1
    best_score = -1.0
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        comp = labels == lab
        score = float(heatmap_smoothed[comp].sum())
        if score > best_score:
            best_score = score
            best_label = lab

    if best_label < 0:
        return np.zeros_like(binary_u8), -1

    picked = ((labels == best_label).astype(np.uint8)) * 255
    return picked, best_label


def localize_tumor_with_bounding_box(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    activation_threshold: float = 0.4,
    central_margin_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_gray, boxed, _, _ = localize_tumor_from_gradcam(
        original_image=original_image,
        heatmap=heatmap,
        percentile=82.0,
        border_shrink_frac=max(0.02, central_margin_fraction * 0.2),
    )
    mask_color = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
    return mask_color, boxed


def localize_tumor_from_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    *,
    percentile: float = 82.0,
    border_shrink_frac: float = 0.03,
    gaussian_ksize: int = 9,
    min_area_frac: float = 0.0015,
    morph_open_kernel: int = 3,
    overlay_alpha: float = 0.33,
    box_thickness: int = 5,
    contour_color: Tuple[int, int, int] = (0, 255, 100),
    contour_thickness: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    h, w, _ = original_image.shape
    empty_gray = np.zeros((h, w), dtype=np.uint8)
    out_rgb = original_image.copy().astype(np.uint8)

    brain_mask = build_brain_region_mask_from_rgb(
        original_image.astype(np.uint8),
        border_shrink_frac=border_shrink_frac,
    )

    heatmap_smooth = _prepare_smoothed_heatmap(
        heatmap, h, w, brain_mask, gaussian_ksize=gaussian_ksize
    )

    binary_u8, _thr_used = _percentile_threshold_mask(
        heatmap_smooth, brain_mask, percentile=percentile
    )

    if not np.any(binary_u8):
        binary_u8, _thr_used = _percentile_threshold_mask(
            heatmap_smooth,
            brain_mask,
            percentile=max(percentile - 12.0, 65.0),
        )

    if not np.any(binary_u8):
        tissue = brain_mask > 0.05
        if np.any(tissue):
            scaled = heatmap_smooth.copy()
            mx = float(scaled[tissue].max())
            if mx > 1e-8:
                u8 = np.zeros_like(scaled, dtype=np.uint8)
                u8[tissue] = np.clip(
                    (scaled[tissue] / mx * 255.0), 0, 255
                ).astype(np.uint8)
                adapt = cv2.adaptiveThreshold(
                    u8,
                    maxValue=255,
                    adaptiveMethod=cv2.ADAPTIVE_GAUSSIAN_C,
                    thresholdType=cv2.THRESH_BINARY,
                    blockSize=min(51, max(11, ((min(h, w) // 8) | 1))),
                    C=-5,
                )
                bm255 = np.clip(brain_mask * 255.0, 0, 255).astype(np.uint8)
                binary_u8 = cv2.bitwise_and(adapt, bm255)

    image_area = h * w
    min_area = max(40, int(min_area_frac * image_area))

    comp_mask_u8, comp_label = _cleanup_mask_and_pick_component(
        binary_u8,
        heatmap_smooth,
        min_area_pixels=min_area,
        open_kernel=morph_open_kernel,
    )

    viz = heatmap_smooth.copy()
    tissue = brain_mask > 0.05
    if np.any(tissue):
        vals = viz[tissue]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 99))
        viz = np.clip((viz - lo) / (hi - lo + 1e-8), 0.0, 1.0) * brain_mask
    heatmap_uint8 = np.uint8(255 * np.clip(viz, 0.0, 1.0))
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    if comp_label < 0 or not np.any(comp_mask_u8):
        return empty_gray, out_rgb, heatmap_rgb, False

    mask_gray = comp_mask_u8

    tint = np.zeros_like(out_rgb, dtype=np.float32)
    tint[:, :] = np.array(contour_color, dtype=np.float32)
    fg = (mask_gray > 0).astype(np.float32)[..., np.newaxis]
    base_f = out_rgb.astype(np.float32)
    blended = base_f * (1.0 - overlay_alpha * fg) + tint * (overlay_alpha * fg)
    out_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_gray, out_rgb, heatmap_rgb, False

    main_contour = max(contours, key=cv2.contourArea)
    smoothed_polys = []
    for c in contours:
        eps = max(2.0, 0.01 * cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, eps, closed=True)
        if len(approx) >= 3:
            smoothed_polys.append(approx)
    if smoothed_polys:
        cv2.polylines(
            out_rgb,
            smoothed_polys,
            isClosed=True,
            color=contour_color,
            thickness=contour_thickness,
            lineType=cv2.LINE_AA,
        )

    bx, by, bw_, bh_ = cv2.boundingRect(main_contour)
    box_color_rgb = (40, 220, 40)
    cv2.rectangle(
        out_rgb,
        (bx, by),
        (bx + bw_, by + bh_),
        color=box_color_rgb,
        thickness=box_thickness,
        lineType=cv2.LINE_AA,
    )

    return mask_gray, out_rgb, heatmap_rgb, True


def thesis_localization_four_panel(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    localized_rgb: np.ndarray,
    predicted_class_name: str,
    confidence: float,
):
    subtitle = (
        "Grad-CAM localization is approximate and intended for interpretability.\n"
        "It highlights regions consulted by the model — not guaranteed tumor boundaries."
    )

    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.9))
    title_fs = 15
    caption_fs = 12

    axes[0].imshow(original_rgb.astype(np.uint8))
    axes[0].set_title("Original MRI", fontsize=title_fs, pad=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb.astype(np.uint8))
    axes[1].set_title(
        "Grad-CAM heatmap\n(brain-masked)",
        fontsize=title_fs - 2,
        pad=12,
    )
    axes[1].axis("off")

    axes[2].imshow(mask_rgb.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        "Clean localization mask",
        fontsize=title_fs - 2,
        pad=12,
    )
    axes[2].axis("off")

    axes[3].imshow(localized_rgb.astype(np.uint8))
    axes[3].set_title(
        "Tumor localization (overlay + bbox)",
        fontsize=title_fs - 2,
        pad=12,
    )
    axes[3].axis("off")

    axes[3].annotate(
        f"Prediction: {predicted_class_name}\n{confidence * 100:.1f}% confidence",
        xy=(12, 16),
        xycoords="axes pixels",
        fontsize=caption_fs,
        fontweight="semibold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="black",
            edgecolor="#00cc44",
            linewidth=2,
            alpha=0.65,
        ),
    )

    fig.suptitle(
        f'Predicted: {predicted_class_name} — {confidence * 100:.1f}% confidence',
        fontsize=caption_fs + 3,
        y=1.03,
        fontweight="medium",
    )
    fig.text(
        0.5,
        0.015,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=caption_fs,
        style="italic",
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    return fig, axes


def run_batch_localization_pipeline(
    image_path: str,
    model: tf.keras.Model,
    last_conv_layer: layers.Layer,
    image_size: Tuple[int, int] = (224, 224),
    activation_threshold: float = 0.4,
    central_margin_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    original_image, input_tensor = load_and_preprocess_image(
        image_path, target_size=image_size
    )
    preds_arr = model.predict(input_tensor, verbose=0)[0]
    predicted_class_index = int(np.argmax(preds_arr))
    prediction_confidence = float(preds_arr[predicted_class_index])

    heatmap, _ = compute_gradcam_heatmap(
        model=model,
        img_tensor=input_tensor,
        last_conv_layer=last_conv_layer,
    )
    mask_gray, image_with_box, heatmap_rgb, has_valid_box = localize_tumor_from_gradcam(
        original_image=original_image,
        heatmap=heatmap,
        percentile=82.0,
        border_shrink_frac=max(0.02, central_margin_fraction * 0.2),
    )
    mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

    return (
        original_image,
        heatmap_rgb,
        mask_rgb,
        image_with_box,
        has_valid_box,
        predicted_class_index,
        prediction_confidence,
    )


def localize_from_path(
    image_path: str,
    model: tf.keras.Model,
    last_conv_layer: layers.Layer,
    image_size: Tuple[int, int] = (224, 224),
    activation_threshold: float = 0.4,
    central_margin_fraction: float = 0.15,
) -> np.ndarray:
    _, _, _, image_with_box, _, _, _ = run_batch_localization_pipeline(
        image_path=image_path,
        model=model,
        last_conv_layer=last_conv_layer,
        image_size=image_size,
        activation_threshold=activation_threshold,
        central_margin_fraction=central_margin_fraction,
    )
    return image_with_box


def run_batch_localization(user_model_path: Optional[str] = None) -> None:
    ensure_project_directories()

    model_path = resolve_model_path(PROJECT_ROOT, user_model_path)
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test"
    output_dir = PROJECT_ROOT / "outputs" / "final_results"

    image_size: Tuple[int, int] = (224, 224)
    batch_activation_threshold = 0.4
    batch_central_margin_fraction = 0.15

    print(f"[INFO] Project root: {PROJECT_ROOT.resolve()}")
    print(f"[INFO] Output directory: {output_dir.resolve()}")

    if not Path(model_path).is_file():
        print(f"[ERROR] Model file not found: {model_path}")
        return

    if not test_dir.is_dir():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_balanced_test_paths(test_dir, per_class=10)
    if not pairs:
        print(
            f"[ERROR] No image files ({', '.join(_BATCH_IMAGE_EXTS)}) "
            f"found under: {test_dir}"
        )
        return

    total = len(pairs)
    print(
        f"[INFO] Batch mode: {total} image(s) (10 per class: glioma, meningioma, pituitary)."
    )

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    ensure_model_built(model, image_size)

    last_conv_layer = find_last_conv_layer(model)
    print(f"[INFO] Using last Conv2D layer for Grad-CAM: {last_conv_layer.name}")

    plt.ioff()

    class_names_in_order = list(CLASS_NAMES_THESIS)
    per_class_index: dict = {}

    for i, (img_path, folder_class) in enumerate(pairs, start=1):
        print(f"[INFO] Processing image {i}/{total}: {img_path}")

        try:
            (
                original_image,
                heatmap_rgb,
                mask_rgb,
                image_with_box,
                has_valid_box,
                predicted_class_index,
                prediction_confidence,
            ) = run_batch_localization_pipeline(
                image_path=img_path,
                model=model,
                last_conv_layer=last_conv_layer,
                image_size=image_size,
                activation_threshold=batch_activation_threshold,
                central_margin_fraction=batch_central_margin_fraction,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process image: {exc}")
            continue

        if image_with_box is None or not isinstance(image_with_box, np.ndarray):
            print(f"[ERROR] Invalid localization output (not an array) for: {img_path}")
            continue
        if image_with_box.size == 0:
            print(f"[ERROR] Empty localization output for: {img_path}")
            continue

        if not has_valid_box:
            print("[INFO] No valid bounding box found for this image")

        pred_label = (
            class_names_in_order[predicted_class_index]
            if 0 <= predicted_class_index < len(class_names_in_order)
            else f"class_idx_{predicted_class_index}"
        )

        per_class_index[folder_class] = per_class_index.get(folder_class, 0) + 1
        class_idx = per_class_index[folder_class]
        stem = f"{folder_class}_{class_idx}"

        def _save_rgb(save_path: Path, rgb: np.ndarray) -> None:
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            print("Saving image:", save_path)
            if cv2.imwrite(str(save_path), bgr):
                print(f"[INFO] Saved result to: {save_path}")
            else:
                print(f"[ERROR] cv2.imwrite failed for: {save_path}")

        _save_rgb(output_dir / f"{stem}_original.png", original_image)
        _save_rgb(output_dir / f"{stem}_mask.png", mask_rgb)
        _save_rgb(output_dir / f"{stem}_boxed.png", image_with_box)

        panel_path = output_dir / f"{stem}_panel.png"
        print("Saving image:", panel_path)
        thesis_fig, _ = thesis_localization_four_panel(
            original_rgb=original_image,
            heatmap_rgb=heatmap_rgb,
            mask_rgb=mask_rgb,
            localized_rgb=image_with_box,
            predicted_class_name=pred_label,
            confidence=prediction_confidence,
        )
        thesis_fig.savefig(str(panel_path), dpi=300, bbox_inches="tight")
        plt.close(thesis_fig)
        print(f"[INFO] Saved result to: {panel_path}")


def main(
    image_path: Optional[str] = None,
    user_model_path: Optional[str] = None,
) -> None:
    ensure_project_directories()

    model_path = resolve_model_path(PROJECT_ROOT, user_model_path)
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test"
    final_results_dir = PROJECT_ROOT / "outputs" / "final_results"
    output_figure_path = final_results_dir / "tumor_localization_single.png"

    image_size: Tuple[int, int] = (224, 224)

    final_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    ensure_model_built(model, image_size)

    if image_path is not None:
        img_path = image_path
        print(f"[INFO] Selected image path: {img_path}")

        base_image, base_tensor = load_and_preprocess_custom_image(
            img_path, target_size=image_size
        )
        original_image = base_image

        tta_versions = create_tta_versions(base_tensor)

        probs_list = []
        for v in tta_versions:
            preds_v = model.predict(v, verbose=0)
            probs_list.append(preds_v[0])

        probs_avg = np.mean(np.stack(probs_list, axis=0), axis=0)
        predicted_class_index = int(np.argmax(probs_avg))
        prediction_confidence = float(probs_avg[predicted_class_index])

        predicted_class_name = (
            CLASS_NAMES_THESIS[predicted_class_index]
            if 0 <= predicted_class_index < len(CLASS_NAMES_THESIS)
            else f"index_{predicted_class_index}"
        )

        best_idx = 0
        best_score = -1.0
        for i, v in enumerate(tta_versions):
            preds_v = model.predict(v, verbose=0)
            score = float(preds_v[0][predicted_class_index])
            if score > best_score:
                best_score = score
                best_idx = i

        input_tensor = tta_versions[best_idx]

    else:
        ds_path = str(test_dir)
        img_path, true_class_name = select_one_image_path(ds_path)
        print(f"[INFO] Selected image path: {img_path}")
        print(f"[INFO] Folder class (ground truth label): {true_class_name}")

        original_image, input_tensor = load_and_preprocess_image(
            img_path, target_size=image_size
        )

        predictions = model.predict(input_tensor, verbose=0)
        predicted_class_index = int(np.argmax(predictions[0]))
        prediction_confidence = float(predictions[0][predicted_class_index])

        predicted_class_name = (
            CLASS_NAMES_THESIS[predicted_class_index]
            if 0 <= predicted_class_index < len(CLASS_NAMES_THESIS)
            else f"index_{predicted_class_index}"
        )

    print(f"[RESULT] Predicted class index: {predicted_class_index}")
    print(f"[RESULT] Predicted class name:  {predicted_class_name}")
    print(f"[RESULT] Prediction confidence: {prediction_confidence * 100:.2f}%")

    last_conv_layer = find_last_conv_layer(model)
    print(f"[INFO] Using last Conv2D layer for Grad-CAM: {last_conv_layer.name}")

    heatmap, _ = compute_gradcam_heatmap(
        model=model,
        img_tensor=input_tensor,
        last_conv_layer=last_conv_layer,
    )

    mask_gray, image_with_box, heatmap_rgb, mask_has_content = localize_tumor_from_gradcam(
        original_image=original_image,
        heatmap=heatmap,
        percentile=82.0,
    )

    localization_mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

    if not mask_has_content:
        print("[WARNING] No reliable localized tumour region survived post-processing.")

    if image_path is not None:
        if (prediction_confidence * 100.0) < 70.0 or not mask_has_content:
            print("[WARNING] External image result may be unreliable.")

    fig, _ = thesis_localization_four_panel(
        original_rgb=original_image.astype(np.uint8),
        heatmap_rgb=heatmap_rgb,
        mask_rgb=localization_mask_rgb,
        localized_rgb=image_with_box.astype(np.uint8),
        predicted_class_name=predicted_class_name,
        confidence=prediction_confidence,
    )
    fig.savefig(str(output_figure_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    abs_out = str(output_figure_path.resolve())
    print(f"[INFO] Saved single-image thesis figure to: {abs_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM-based tumour localization (ResNet50 thesis pipeline)."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=(
            "Run single-image 4-panel figure (one sample from test set if --image omitted)."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to one MRI image for the 4-panel demo (implies single-image mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to .keras classifier (default: models/resnet50_model.keras). "
            "Relative paths resolved from project root."
        ),
    )

    args = parser.parse_args()

    if args.image is not None or args.single:
        main(image_path=args.image, user_model_path=args.model)
    else:
        run_batch_localization(user_model_path=args.model)
