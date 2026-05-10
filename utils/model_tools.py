"""Keras model loading helpers."""

from pathlib import Path
from typing import Optional, Tuple, Union

import tensorflow as tf


def ensure_model_built(
    model: tf.keras.Model,
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    dummy = tf.zeros((1, image_size[0], image_size[1], 3), dtype=tf.float32)
    _ = model(dummy, training=False)


def resolve_model_path(
    project_root: Union[str, Path],
    user_model_arg: Optional[str],
) -> str:
    root = Path(project_root)
    if not user_model_arg:
        return str(root / "models" / "resnet50_model.keras")
    p = Path(user_model_arg)
    if p.is_absolute():
        return str(p)
    return str((root / user_model_arg).resolve())


def classification_output_tensor(model: tf.keras.Model):
    outs = getattr(model, "outputs", None)
    if outs is None:
        return model.output
    if isinstance(outs, (list, tuple)):
        if len(outs) == 1:
            return outs[0]
        raise ValueError(
            f"Expected a single classification output; model has {len(outs)} outputs."
        )
    return outs
