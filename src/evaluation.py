from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class MetricsResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    classification_report: str


def predict_dataset(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_true, y_pred) for a dataset.

    Assumes labels are integer-encoded (label_mode="int").
    """
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_pred.extend(pred.tolist())
        y_true.extend(batch_y.numpy().tolist())

    return np.asarray(y_true), np.asarray(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, class_names: list[str]) -> MetricsResult:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return MetricsResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=cm,
        classification_report=report,
    )


def save_metrics(result: MetricsResult, *, out_dir: Path, run_name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as a simple text file (thesis-friendly).
    summary_path = out_dir / f"{run_name}_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Accuracy: {result.accuracy:.6f}\n")
        f.write(f"Precision (macro): {result.precision_macro:.6f}\n")
        f.write(f"Recall (macro): {result.recall_macro:.6f}\n")
        f.write(f"F1-score (macro): {result.f1_macro:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(result.classification_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(result.confusion_matrix))
        f.write("\n")


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add counts inside cells
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

