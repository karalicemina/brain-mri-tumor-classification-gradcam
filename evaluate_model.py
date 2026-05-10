# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — baseline CNN evaluation
# Author: [Your Name]
# University: [Your University]
# Purpose: Test-set metrics, confusion matrix, and report for the baseline model.
# -----------------------------------------------------------------------------
"""Evaluate a trained baseline Keras CNN on the held-out test set."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from utils.paths import PROJECT_ROOT, ensure_project_directories


def load_test_dataset(
    test_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Load test images from a folder-per-class directory (shuffle=False for aligned labels)."""
    return tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )


def get_true_labels_and_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer class indices for ground truth and argmax predictions."""
    true_labels_list = []
    for _, labels in test_ds:
        true_labels_list.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(true_labels_list, axis=0)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_dir: str,
    figure_name: str = "baseline_cnn_confusion_matrix.png",
) -> None:
    """Plot confusion matrix heatmap and save to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    output_path = os.path.join(output_dir, figure_name)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved confusion matrix figure to: {output_path}")

    plt.show()
    plt.close()


def save_classification_report(
    report_str: str,
    output_dir: str,
    file_name: str = "baseline_cnn_classification_report.txt",
) -> None:
    """Write sklearn classification_report text to disk."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[INFO] Saved classification report to: {output_path}")


def main() -> None:
    ensure_project_directories()

    model_path = PROJECT_ROOT / "models" / "baseline_cnn.keras"
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test"

    image_size = (224, 224)
    batch_size = 32

    figures_output_dir = PROJECT_ROOT / "outputs" / "figures"
    metrics_output_dir = PROJECT_ROOT / "outputs" / "metrics"
    figures_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print(f"[INFO] Loading test dataset from: {test_dir}")
    test_ds = load_test_dataset(
        test_dir=str(test_dir),
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = test_ds.class_names
    print(f"[INFO] Class names: {class_names}")

    print("[INFO] Evaluating model on test dataset...")
    eval_results = model.evaluate(test_ds, verbose=1)

    if isinstance(eval_results, list) and len(eval_results) >= 2:
        test_loss, test_accuracy = eval_results[:2]
    else:
        test_loss = eval_results
        test_accuracy = None

    print(f"[RESULT] Test loss: {test_loss:.4f}")
    if test_accuracy is not None:
        print(f"[RESULT] Test accuracy (from model.evaluate): {test_accuracy:.4f}")

    print("[INFO] Generating predictions and computing metrics...")
    y_true, y_pred = get_true_labels_and_predictions(model, test_ds)

    skl_accuracy = accuracy_score(y_true, y_pred)
    print(f"[RESULT] Test accuracy (from scikit-learn): {skl_accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"[RESULT] Precision (macro): {precision:.4f}")
    print(f"[RESULT] Recall (macro):    {recall:.4f}")
    print(f"[RESULT] F1-score (macro):  {f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n========== Classification Report ==========")
    print(report)
    print("===========================================\n")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_dir=str(figures_output_dir),
        figure_name="baseline_cnn_confusion_matrix.png",
    )

    save_classification_report(
        report_str=report,
        output_dir=str(metrics_output_dir),
        file_name="baseline_cnn_classification_report.txt",
    )

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
