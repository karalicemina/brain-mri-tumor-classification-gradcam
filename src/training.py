from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    fine_tune_epochs: int = 5
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def plot_history(history: tf.keras.callbacks.History, *, title: str, out_path: Path) -> None:
    """
    Save accuracy/loss curves from a Keras History object.
    """
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    if "accuracy" in hist:
        axes[0].plot(hist["accuracy"], label="train_acc")
    if "val_accuracy" in hist:
        axes[0].plot(hist["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    if "loss" in hist:
        axes[1].plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        axes[1].plot(hist["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    cfg: TrainingConfig,
    run_name: str,
) -> tf.keras.callbacks.History:
    """
    Train a model and save plots + checkpoint.
    """
    ensure_dirs(cfg.output_dir / "figures", cfg.output_dir / "metrics", cfg.model_dir)

    ckpt_path = Path(cfg.model_dir) / f"{run_name}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks)

    plot_path = Path(cfg.output_dir) / "figures" / f"{run_name}_training_curves.png"
    plot_history(history, title=f"Training Curves — {run_name}", out_path=plot_path)
    return history

