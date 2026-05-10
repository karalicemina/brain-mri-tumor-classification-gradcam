# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — Grad-CAM figure export
# Author: [Your Name]
# University: [Your University]
# Purpose: Single-image four-panel Grad-CAM visualization for thesis chapters.
# -----------------------------------------------------------------------------
"""Grad-CAM visualization (four-panel figure) for thesis documentation."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tumor_localization import (
    CLASS_NAMES_THESIS,
    compute_gradcam_heatmap,
    ensure_model_built,
    find_last_conv_layer,
    load_and_preprocess_image,
    localize_tumor_from_gradcam,
    resolve_model_path,
    select_one_image_path,
    thesis_localization_four_panel,
)
from utils.paths import PROJECT_ROOT, ensure_project_directories


def main() -> None:
    ensure_project_directories()

    model_path = resolve_model_path(PROJECT_ROOT, None)
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test"
    figures_output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_path = figures_output_dir / "gradcam_result.png"

    image_size = (224, 224)

    figures_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    ensure_model_built(model, image_size)

    print(f"[INFO] Class order (softmax): {list(CLASS_NAMES_THESIS)}")

    img_path, folder_label = select_one_image_path(test_dir)
    print(f"[INFO] Selected file: {img_path}")
    print(f"[INFO] Dataset folder label (ground truth): {folder_label}")

    original_image, input_tensor = load_and_preprocess_image(
        img_path, target_size=image_size
    )

    probs = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = (
        CLASS_NAMES_THESIS[pred_idx]
        if 0 <= pred_idx < len(CLASS_NAMES_THESIS)
        else f"class_idx_{pred_idx}"
    )
    confidence = float(probs[pred_idx])

    print(f"[RESULT] Prediction: {pred_name} ({confidence * 100:.2f}% confidence)")

    conv_layer = find_last_conv_layer(model)
    print(f"[INFO] Grad-CAM source layer: {conv_layer.name}")

    heatmap, cam_class = compute_gradcam_heatmap(
        model=model,
        img_tensor=input_tensor,
        last_conv_layer=conv_layer,
    )
    if cam_class != pred_idx:
        print(
            f"[INFO] Note: arg-max class index from forward pass ({pred_idx}) vs "
            f"Grad-CAM tape index ({cam_class}) — usually identical; tiny float "
            f"differences can rarely disagree."
        )

    mask_gray, localized_rgb, heatmap_rgb, region_ok = localize_tumor_from_gradcam(
        original_image=original_image,
        heatmap=heatmap,
        percentile=82.0,
    )
    if not region_ok:
        print(
            "[WARNING] Post-processing kept no salient blob — panels still save, "
            "but the bounding box may be missing."
        )

    mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

    fig, _ = thesis_localization_four_panel(
        original_rgb=original_image.astype(np.uint8),
        heatmap_rgb=heatmap_rgb,
        mask_rgb=mask_rgb,
        localized_rgb=localized_rgb.astype(np.uint8),
        predicted_class_name=pred_name,
        confidence=confidence,
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved Grad-CAM thesis figure to: {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
