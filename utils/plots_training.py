"""Training curves for ResNet50 thesis runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_training_curves(
    history_stage1: tf.keras.callbacks.History,
    history_stage2: tf.keras.callbacks.History,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = history_stage1.history.get("accuracy", []) + history_stage2.history.get(
        "accuracy", []
    )
    val_acc = history_stage1.history.get("val_accuracy", []) + history_stage2.history.get(
        "val_accuracy", []
    )
    loss = history_stage1.history.get("loss", []) + history_stage2.history.get("loss", [])
    val_loss = history_stage1.history.get("val_loss", []) + history_stage2.history.get(
        "val_loss", []
    )
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    acc_path = output_dir / "resnet50_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved accuracy plot: {acc_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / "resnet50_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved loss plot: {loss_path}")
