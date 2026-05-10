# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — ResNet50 training
# Author: [Emina Karalic]
# University: [Sarajevo School of Science and Technology]
# Purpose: Transfer-learning pipeline with metrics and figures for thesis.
# -----------------------------------------------------------------------------
"""ResNet50 transfer learning for 3-class brain MRI classification (glioma, meningioma, pituitary)."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.paths import PROJECT_ROOT, ensure_project_directories
from utils.plots_training import plot_training_curves

RNG_SEED = 42


def set_random_seeds(seed: int = RNG_SEED) -> None:
    """Set Python, NumPy, and TensorFlow RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_train_augmentation() -> tf.keras.Sequential:
    """Training-only augmentation (pixel space, before ``preprocess_input``)."""
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomContrast(0.10),
        ],
        name="train_augmentation",
    )


def compute_class_weights_dict(
    train_ds_raw: tf.data.Dataset,
    num_classes: int,
) -> Dict[int, float]:
    """Balanced class weights from one-hot training labels."""
    y_all: List[np.ndarray] = []
    for _, labels in train_ds_raw:
        y_all.append(np.argmax(labels.numpy(), axis=1))
    y_int = np.concatenate(y_all, axis=0)

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_int,
    )
    return {int(i): float(cw[i]) for i in range(num_classes)}


def create_datasets(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], tf.data.Dataset]:
    """Directory loaders with categorical labels; train applies augmentation + ResNet preprocess."""
    augmentation = build_train_augmentation()

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

    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds_raw.class_names

    def ensure_three_channels(images: tf.Tensor, labels: tf.Tensor):
        """Expand single-channel inputs to RGB for ResNet."""
        if images.shape.rank == 4 and images.shape[-1] == 1:
            images = tf.image.grayscale_to_rgb(images)
        elif images.shape.rank == 3 and images.shape[-1] == 1:
            images = tf.image.grayscale_to_rgb(images)
        return images, labels

    def preprocess_for_resnet(images: tf.Tensor, labels: tf.Tensor):
        """ResNet ``preprocess_input`` (val/test and train after augmentation)."""
        images = tf.cast(images, tf.float32)
        images = preprocess_input(images)
        return images, labels

    def augment_then_preprocess(images: tf.Tensor, labels: tf.Tensor):
        """Augmentation then ``preprocess_input`` (training pipeline only)."""
        images = tf.cast(images, tf.float32)
        images = augmentation(images, training=True)
        images = preprocess_input(images)
        return images, labels

    autotune = tf.data.AUTOTUNE

    train_ds = (
        train_ds_raw
        .map(ensure_three_channels, num_parallel_calls=autotune)
        .map(augment_then_preprocess, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    val_ds = (
        val_ds_raw
        .map(ensure_three_channels, num_parallel_calls=autotune)
        .map(preprocess_for_resnet, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    test_ds = (
        test_ds_raw
        .map(ensure_three_channels, num_parallel_calls=autotune)
        .map(preprocess_for_resnet, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    return train_ds, val_ds, test_ds, class_names, train_ds_raw


def build_resnet50_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.5,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Full classifier and frozen ResNet50 base (for staged unfreezing)."""
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Adam + categorical crossentropy + accuracy."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def save_history(
    history_stage1: tf.keras.callbacks.History,
    history_stage2: tf.keras.callbacks.History,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_payload: Dict[str, Dict[str, List[float]]] = {
        "stage1_head_training": {
            k: [float(vv) for vv in v]
            for k, v in history_stage1.history.items()
        },
        "stage2_fine_tuning": {
            k: [float(vv) for vv in v]
            for k, v in history_stage2.history.items()
        },
    }
    path = output_dir / "resnet50_training_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"[INFO] Saved training history: {path}")


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    metrics_output_dir: Path,
    figures_output_dir: Path,
) -> None:
    metrics_output_dir = Path(metrics_output_dir)
    figures_output_dir = Path(figures_output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    figures_output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Evaluating model on test dataset...")
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss, test_accuracy = eval_results[0], eval_results[1]
    print(f"[RESULT] Test loss: {test_loss:.4f}")
    print(f"[RESULT] Test accuracy (Keras): {test_accuracy:.4f}")

    y_true_parts = []
    for _, labels in test_ds:
        y_true_parts.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(y_true_parts, axis=0)

    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    print(f"[RESULT] Accuracy (sklearn): {accuracy:.4f}")
    print(f"[RESULT] Precision (macro): {precision:.4f}")
    print(f"[RESULT] Recall (macro):    {recall:.4f}")
    print(f"[RESULT] F1-score (macro):  {f1:.4f}")

    report_path = metrics_output_dir / "resnet50_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Saved classification report: {report_path}")

    summary = {
        "test_loss": float(test_loss),
        "test_accuracy_keras": float(test_accuracy),
        "accuracy_sklearn": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    summary_path = metrics_output_dir / "resnet50_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved metrics summary: {summary_path}")

    cm_path = metrics_output_dir / "resnet50_confusion_matrix.npy"
    np.save(str(cm_path), cm)
    print(f"[INFO] Saved confusion matrix array: {cm_path}")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ResNet50 Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_fig_path = figures_output_dir / "resnet50_confusion_matrix.png"
    plt.savefig(str(cm_fig_path), dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix figure: {cm_fig_path}")


def count_trainable_weights(model: tf.keras.Model) -> int:
    """Total trainable scalar weights."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def summarize_trainable_layers(model: tf.keras.Model, max_print: int = 25) -> None:
    """Log trainable layer names (up to ``max_print``)."""
    trainable_named = [
        layer.name
        for layer in model.layers
        if getattr(layer, "trainable", False)
    ]
    print(f"[INFO] Number of trainable top-level layers: {len(trainable_named)}")
    print(
        "[INFO] Total trainable parameters (approx. weight scalars):"
        f" {count_trainable_weights(model)}"
    )
    preview = trainable_named[:max_print]
    if preview:
        print(f"[INFO] First {len(preview)} trainable layers (examples):")
        for name in preview:
            print(f"       - {name}")


def main() -> None:
    """
    End-to-end ResNet50 transfer learning and fine-tuning pipeline.
    """
    set_random_seeds(RNG_SEED)
    ensure_project_directories()

    train_dir = str(PROJECT_ROOT / "data" / "dataset" / "train")
    val_dir = str(PROJECT_ROOT / "data" / "dataset" / "val")
    test_dir = str(PROJECT_ROOT / "data" / "dataset" / "test")

    model_output_path = PROJECT_ROOT / "models" / "resnet50_model.keras"
    figures_output_dir = PROJECT_ROOT / "outputs" / "figures"
    metrics_output_dir = PROJECT_ROOT / "outputs" / "metrics"

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (224, 224)
    batch_size = 32
    input_shape = (224, 224, 3)

    stage1_epochs = 10
    stage2_epochs = 10

    stage1_lr = 1e-3
    stage2_lr = 1e-5

    print("[INFO] Creating datasets...")
    train_ds, val_ds, test_ds, class_names, train_ds_raw = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    num_classes = len(class_names)
    print(f"[INFO] Class names: {class_names}")
    print(f"[INFO] Number of classes: {num_classes}")

    class_weight_dict = compute_class_weights_dict(train_ds_raw, num_classes)
    print("[INFO] Class weights (for imbalanced classes):")
    for idx, cname in enumerate(class_names):
        print(f"       {cname}: {class_weight_dict[idx]:.4f}")

    print("[INFO] Building ResNet50 transfer learning model...")
    model, base_model = build_resnet50_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=0.5,
    )

    print("=" * 72)
    print("[INFO] STAGE 1 START: train classification head (ResNet50 frozen)")
    print(f"[INFO] Stage 1 learning rate: {stage1_lr}")
    print("=" * 72)

    compile_model(model, learning_rate=stage1_lr)
    model.summary()
    summarize_trainable_layers(model)

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(model_output_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=1,
    )

    print("=" * 72)
    print("[INFO] STAGE 2 START: fine-tune top layers of ResNet50")
    print(f"[INFO] Stage 2 base learning rate: {stage2_lr}")
    print("       (ReduceLROnPlateau will lower LR automatically when val_loss stalls)")
    print("=" * 72)

    base_model.trainable = True

    unfreeze_last_n = 50
    fine_tune_at = max(0, len(base_model.layers) - unfreeze_last_n)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    compile_model(model, learning_rate=stage2_lr)
    summarize_trainable_layers(model)

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=1,
    )

    model.save(str(model_output_path))
    print(f"[INFO] Saved trained model: {model_output_path}")

    save_history(
        history_stage1=history_stage1,
        history_stage2=history_stage2,
        output_dir=metrics_output_dir,
    )

    plot_training_curves(
        history_stage1=history_stage1,
        history_stage2=history_stage2,
        output_dir=Path(figures_output_dir),
    )

    evaluate_and_save_metrics(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        metrics_output_dir=metrics_output_dir,
        figures_output_dir=figures_output_dir,
    )

    print("[INFO] ResNet50 pipeline completed successfully.")


if __name__ == "__main__":
    main()

