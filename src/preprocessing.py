from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tensorflow as tf


@dataclass(frozen=True)
class SplitConfig:
    """
    Dataset split configuration.

    Fractions must sum to 1.0.
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Preprocessing and augmentation settings.
    """

    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    normalize: bool = True

    # Augmentation for training only
    aug_flip: bool = True
    aug_rotation: float = 0.05  # fraction of a full turn (Keras RandomRotation)
    aug_zoom: float = 0.10


def build_augmentation_layer(cfg: PreprocessConfig) -> tf.keras.Sequential:
    """
    Create Keras augmentation layers to be applied to training data.
    """
    layers: list[tf.keras.layers.Layer] = []
    if cfg.aug_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if cfg.aug_rotation and cfg.aug_rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(cfg.aug_rotation))
    if cfg.aug_zoom and cfg.aug_zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(cfg.aug_zoom))

    return tf.keras.Sequential(layers, name="augmentation")


def build_normalization_layer() -> tf.keras.layers.Layer:
    """
    Normalize uint8 pixels in [0, 255] to float32 in [0, 1].
    """
    return tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")


def load_splits_from_directory(
    data_dir: Path,
    *,
    cfg: PreprocessConfig,
    split_cfg: SplitConfig = SplitConfig(),
    label_mode: Literal["int", "categorical", "binary"] = "int",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Build train/val/test `tf.data.Dataset` objects from a folder-per-class dataset.

    Notes
    -----
    - Uses deterministic shuffling controlled by `split_cfg.seed`.
    - Uses 2-stage split: (train) vs (temp=val+test), then temp into val/test.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if abs((split_cfg.train + split_cfg.val + split_cfg.test) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    # Stage 1: train vs temp
    temp_fraction = split_cfg.val + split_cfg.test
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=split_cfg.seed,
        validation_split=temp_fraction,
        subset="training",
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=split_cfg.seed,
        validation_split=temp_fraction,
        subset="validation",
    )

    class_names = list(train_ds.class_names)

    # Stage 2: temp into val/test (by taking/skipping batches)
    # We operate at the element level (batches) for simplicity and reproducibility.
    temp_cardinality = tf.data.experimental.cardinality(temp_ds).numpy()
    if temp_cardinality <= 0:
        raise ValueError("Temp split is empty; check dataset and split fractions.")

    val_ratio_within_temp = split_cfg.val / temp_fraction if temp_fraction > 0 else 0.5
    val_batches = int(round(temp_cardinality * val_ratio_within_temp))
    val_batches = max(1, min(val_batches, temp_cardinality - 1))

    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    return train_ds, val_ds, test_ds, class_names


def prepare_for_training(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    *,
    cfg: PreprocessConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Apply (optional) normalization + (training-only) augmentation and configure
    performance optimizations (cache/prefetch).
    """
    aug = build_augmentation_layer(cfg)
    norm = build_normalization_layer() if cfg.normalize else None

    def _map_train(x, y):
        x = tf.cast(x, tf.float32)
        if norm is not None:
            x = norm(x)
        x = aug(x, training=True)
        return x, y

    def _map_eval(x, y):
        x = tf.cast(x, tf.float32)
        if norm is not None:
            x = norm(x)
        return x, y

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(_map_train, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(_map_eval, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(_map_eval, num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds, test_ds

