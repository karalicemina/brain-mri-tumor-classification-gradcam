"""Grad-CAM for nested ResNet50 and flat CNNs (Keras 3)."""

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.model_tools import classification_output_tensor


def _iter_nested_layers_ordered(model: tf.keras.Model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            yield from _iter_nested_layers_ordered(layer)
        else:
            yield layer


def find_last_conv_layer(model: tf.keras.Model) -> layers.Layer:
    last: Optional[layers.Layer] = None
    for layer in _iter_nested_layers_ordered(model):
        if isinstance(layer, layers.Conv2D):
            last = layer
    if last is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    return last


RESNET50_LAST_CONV_NAME = "conv5_block3_3_conv"


def _safe_get_child_layer(
    parent: tf.keras.Model, layer_name: str
) -> Optional[layers.Layer]:
    try:
        return parent.get_layer(layer_name)
    except ValueError:
        return None


def find_resnet50_base_submodel(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    for name in ("resnet50",):
        L = _safe_get_child_layer(model, name)
        if isinstance(L, tf.keras.Model) and (
            _safe_get_child_layer(L, RESNET50_LAST_CONV_NAME) is not None
        ):
            return L
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            _safe_get_child_layer(layer, RESNET50_LAST_CONV_NAME) is not None
        ):
            return layer
    return None


def _print_outer_model_layers(model: tf.keras.Model) -> None:
    print("[DEBUG] Outer model.layers (top-level):")
    for i, L in enumerate(model.layers):
        kind = "Model" if isinstance(L, tf.keras.Model) else type(L).__name__
        print(f"       [{i:2d}] {L.name:40s}  ({kind})")


def _head_layers_after_base(
    model: tf.keras.Model, base: tf.keras.Model
) -> List[layers.Layer]:
    idx: Optional[int] = None
    for i, L in enumerate(model.layers):
        if L is base or L.name == base.name:
            idx = i
            break
    if idx is None:
        raise RuntimeError("Could not locate the ResNet base in model.layers.")
    return list(model.layers[idx + 1 :])


def _apply_classification_head(
    head: List[layers.Layer],
    base_spatial: tf.Tensor,
    training: bool = False,
) -> tf.Tensor:
    x = base_spatial
    for layer in head:
        if isinstance(layer, (layers.Dropout, layers.SpatialDropout2D)):
            x = layer(x, training=training)
        elif isinstance(layer, layers.BatchNormalization):
            x = layer(x, training=training)
        else:
            x = layer(x, training=training)
    return x


def _gradcam_from_conv_and_grads(conv_output: tf.Tensor, grads: tf.Tensor) -> np.ndarray:
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feat = conv_output[0]
    heatmap = tf.reduce_sum(feat * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    arr = heatmap.numpy()
    m = float(np.max(arr))
    if m > 0:
        arr /= m
    return arr


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    last_conv_layer: Optional[layers.Layer] = None,
) -> Tuple[np.ndarray, int]:
    pred_tensor = classification_output_tensor(model)
    img_tensor_tf = tf.cast(tf.convert_to_tensor(img_tensor), dtype=tf.float32)
    base = find_resnet50_base_submodel(model)
    conv_layer_name = RESNET50_LAST_CONV_NAME

    _print_outer_model_layers(model)

    if base is not None:
        print(f"[DEBUG] Detected nested base model: name={base.name!r}")
        conv_ref = _safe_get_child_layer(base, conv_layer_name)
        if conv_ref is None:
            conv_ref = find_last_conv_layer(base)
            conv_layer_name = conv_ref.name
        print(f"[DEBUG] Selected conv layer for CAM: {conv_layer_name!r}")

        path_joint_ok = False
        try:
            gm = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_ref.output, pred_tensor],
                name="gradcam_resnet_joint",
            )
            _ = gm(img_tensor_tf, training=False)
            path_joint_ok = True
        except (ValueError, KeyError, TypeError, tf.errors.InvalidArgumentError) as exc:
            print(
                "[DEBUG] Joint conv+softmax Grad-CAM cannot be built or called: "
                f"{type(exc).__name__}: {exc}"
            )

        if path_joint_ok:
            print("[DEBUG] Grad-CAM path: joint nested (conv + softmax).")
            gm = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_ref.output, pred_tensor],
                name="gradcam_resnet_joint",
            )
            with tf.GradientTape() as tape:
                conv_output, preds = gm(img_tensor_tf, training=False)
                pred_idx = tf.argmax(preds[0])
                class_ch = preds[:, pred_idx]
            grads = tape.gradient(class_ch, conv_output)
            if grads is not None:
                heatmap = _gradcam_from_conv_and_grads(conv_output, grads)
                print("[DEBUG] Nested fallback used: False")
                return heatmap, int(pred_idx.numpy())
            print("[DEBUG] Joint path returned None gradients — using manual head fallback.")

        print(
            "[DEBUG] Grad-CAM path: nested **fallback** "
            "(inner base + manual classification head)."
        )

        inner: Optional[tf.keras.Model] = None
        last_inner_err: Optional[BaseException] = None
        for factory in (
            (lambda: model.inputs, "model.inputs"),
            (lambda: base.input, "base.input"),
        ):
            try:
                inner = tf.keras.Model(
                    inputs=factory[0](),
                    outputs=[conv_ref.output, base.output],
                    name="gradcam_resnet_inner_feats",
                )
                _ = inner(img_tensor_tf, training=False)
                print(f"[DEBUG] Built inner feature model using {factory[1]}.")
                break
            except (ValueError, KeyError, TypeError, Exception) as exc:
                inner = None
                last_inner_err = exc
                continue

        if inner is None:
            raise RuntimeError(
                "Grad-CAM nested fallback: could not wire conv + base.output. "
                f"Last error: {last_inner_err!r}"
            )

        head = _head_layers_after_base(model, base)
        hnames = ", ".join(L.name for L in head)
        print(
            f"[DEBUG] Classification head layers after base ({len(head)}): {hnames}"
        )

        with tf.GradientTape() as tape:
            conv_out, base_spatial = inner(img_tensor_tf, training=False)
            tape.watch(conv_out)
            preds = _apply_classification_head(head, base_spatial, training=False)
            pred_idx = tf.argmax(preds[0])
            class_ch = preds[:, pred_idx]

        grads = tape.gradient(class_ch, conv_out)
        if grads is None:
            raise RuntimeError(
                "Grad-CAM nested fallback: tape.gradient returned None."
            )

        heatmap = _gradcam_from_conv_and_grads(conv_out, grads)
        print("[DEBUG] Nested fallback used: True")
        return heatmap, int(pred_idx.numpy())

    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer(model)
    print(
        f"[DEBUG] No embedded ResNet50 base — flat Grad-CAM on layer "
        f"{last_conv_layer.name!r}"
    )
    print("[DEBUG] Nested fallback used: n/a (flat model)")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, pred_tensor],
        name="gradcam_flat",
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor_tf, training=False)
        predicted_class_index = tf.argmax(preds[0])
        class_channel = preds[:, predicted_class_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError("Grad-CAM failed: gradients are None for flat model.")

    heatmap = _gradcam_from_conv_and_grads(conv_output, grads)
    return heatmap, int(predicted_class_index.numpy())
