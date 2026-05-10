# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — Single-image inference demo
# Author: [Your Name]
# University: [Your University]
# Purpose: Minimal CLI inference + probability summary for thesis documentation.
# -----------------------------------------------------------------------------
"""Predict tumor class from one MRI image (ResNet50, thesis pipeline)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.constants import CLASS_NAMES_THESIS
from utils.paths import PROJECT_ROOT, ensure_project_directories
from utils.preprocessing import load_and_preprocess_custom_image

CLASS_NAMES = list(CLASS_NAMES_THESIS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict brain MRI tumor class using trained ResNet50."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the MRI image file (e.g. PNG or JPG).",
    )
    args = parser.parse_args()

    ensure_project_directories()

    model_path = PROJECT_ROOT / "models" / "resnet50_model.keras"
    output_dir = PROJECT_ROOT / "outputs" / "final_results"
    output_path = output_dir / "demo_prediction.png"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train with train_resnet50.py or place resnet50_model.keras in models/."
        )

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(str(model_path))

    display_img, x = load_and_preprocess_custom_image(str(image_path), (224, 224))

    print(f"[INFO] Running prediction on: {image_path}")
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence_pct = float(probs[pred_idx]) * 100.0

    print()
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence:      {confidence_pct:.2f}%")
    print("Probabilities per class:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:12s}  {p * 100.0:6.2f}%")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_img)
    ax.axis("off")
    title_text = (
        f"Predicted: {predicted_class}\n" f"Confidence: {confidence_pct:.2f}%"
    )
    ax.set_title(title_text, fontsize=12, pad=12)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved visual result to: {output_path}")


if __name__ == "__main__":
    main()
