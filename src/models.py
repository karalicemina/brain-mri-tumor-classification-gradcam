from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ModelConfig:
    image_size: tuple[int, int] = (224, 224)
    num_classes: int = 2
    learning_rate: float = 1e-3


def build_baseline_cnn(cfg: ModelConfig) -> tf.keras.Model:
    """
    Simple baseline CNN for thesis benchmarking.

    Notes
    -----
    - Uses a small number of convolution blocks to keep training fast and
      interpretation straightforward.
    """
    inputs = tf.keras.Input(shape=(cfg.image_size[0], cfg.image_size[1], 3), name="image")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy" if cfg.num_classes > 1 else "binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_resnet50_transfer(cfg: ModelConfig) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    ResNet50 transfer learning model.

    Returns
    -------
    (model, base_model)
        `model` is the full classifier; `base_model` is the pretrained backbone.
    """
    inputs = tf.keras.Input(shape=(cfg.image_size[0], cfg.image_size[1], 3), name="image")
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def build_efficientnetb0_transfer(cfg: ModelConfig) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    EfficientNetB0 transfer learning model.
    """
    inputs = tf.keras.Input(shape=(cfg.image_size[0], cfg.image_size[1], 3), name="image")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, *, n_layers: int) -> None:
    """
    Fine-tuning helper: unfreeze the last `n_layers` in the backbone.
    """
    if n_layers <= 0:
        return
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

