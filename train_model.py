# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — baseline CNN training
# Author: [Emina Karalic]
# University: [Sarajevo School of Science and Technology]
# Purpose: Train a simple CNN baseline alongside ResNet50 for thesis comparison.
# -----------------------------------------------------------------------------
"""Train and save a baseline CNN on train/val folder-per-class data."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, metrics, models, optimizers

from utils.paths import PROJECT_ROOT, ensure_project_directories


def create_datasets(
    train_dir: str,
    val_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds_raw.class_names
    train_ds = train_ds_raw.prefetch(buffer_size=autotune)
    val_ds = val_ds_raw.prefetch(buffer_size=autotune)
    train_ds.class_names = class_names  # type: ignore[attr-defined]

    return train_ds, val_ds


def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    model = models.Sequential(name="baseline_cnn")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255.0))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy(name="accuracy")],
    )

    return model


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    accuracy_figure_name: str = "baseline_cnn_accuracy.png",
    loss_figure_name: str = "baseline_cnn_loss.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    acc_path = output_dir / accuracy_figure_name
    plt.savefig(str(acc_path), dpi=300)
    print(f"[INFO] Saved accuracy plot to: {acc_path}")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    loss_path = output_dir / loss_figure_name
    plt.savefig(str(loss_path), dpi=300)
    print(f"[INFO] Saved loss plot to: {loss_path}")
    plt.show()
    plt.close()


def main() -> None:
    ensure_project_directories()

    train_dir = str(PROJECT_ROOT / "data" / "dataset" / "train")
    val_dir = str(PROJECT_ROOT / "data" / "dataset" / "val")

    image_size = (224, 224)
    batch_size = 32
    num_classes = 3
    num_epochs = 15

    best_model_path = PROJECT_ROOT / "models" / "baseline_cnn.keras"
    figures_output_dir = PROJECT_ROOT / "outputs" / "figures"

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    figures_output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Class names: {class_names}")

    input_shape = (image_size[0], image_size[1], 3)
    model = build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model_checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[early_stopping_cb, model_checkpoint_cb],
    )

    plot_training_history(history=history, output_dir=figures_output_dir)

    print(f"[INFO] Training complete. Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
