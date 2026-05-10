# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — batch Grad-CAM (baseline CNN)
# Author: [Your Name]
# University: [Your University]
# Purpose: Export multi-image Grad-CAM panels for qualitative thesis analysis.
# -----------------------------------------------------------------------------
"""Baseline CNN batch Grad-CAM: three-panel figures per test image."""

import os
import random
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.paths import PROJECT_ROOT, ensure_project_directories


def get_class_names(dataset_dir: str) -> List[str]:
    """Sorted subdirectory names under ``dataset_dir``."""
    class_names = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    class_names.sort()
    return class_names


def collect_image_paths_by_class(
    dataset_dir: str,
    valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> Dict[str, List[str]]:
    """Map each class folder name to sorted image paths."""
    class_names = get_class_names(dataset_dir)
    images_by_class: Dict[str, List[str]] = {}

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        image_paths: List[str] = []
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_ext):
                image_paths.append(os.path.join(class_dir, fname))
        images_by_class[class_name] = sorted(image_paths)

    return images_by_class


def load_and_preprocess_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """RGB preview uint8 and float batch in [0,255]; model applies ``Rescaling(1/255)``."""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    input_tensor = img_rgb_resized.astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return img_rgb_resized, input_tensor


def find_last_conv_layer(model: tf.keras.Model) -> layers.Layer:
    """Last top-level ``Conv2D`` (flat Sequential baseline)."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model. Cannot compute Grad-CAM.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    last_conv_layer: layers.Layer,
) -> Tuple[np.ndarray, int]:
    """Grad-CAM for flat CNN via conv trunk + rebuilt tail (tape on conv activations)."""
    last_conv_layer_name = last_conv_layer.name
    last_conv_layer = model.get_layer(last_conv_layer_name)

    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output,
    )

    last_conv_index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            last_conv_index = idx
            break

    if last_conv_index is None:
        raise RuntimeError(
            f"Could not find last conv layer '{last_conv_layer_name}' in the model."
        )

    conv_output_shape = last_conv_layer.output.shape[1:]
    classifier_input = tf.keras.Input(shape=conv_output_shape)

    x = classifier_input
    for layer in model.layers[last_conv_index + 1 :]:
        x = layer(x)

    classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)

    img_tensor_tf = tf.cast(tf.convert_to_tensor(img_tensor), dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(img_tensor_tf, training=False)
        tape.watch(conv_output)

        preds = classifier_model(conv_output, training=False)
        predicted_class_index = tf.argmax(preds[0])
        class_channel = preds[:, predicted_class_index]

    grads = tape.gradient(class_channel, conv_output)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM failed: gradients are None. "
            "Please ensure the model is correctly built and the last Conv2D "
            "layer is part of the forward pass."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap, int(predicted_class_index.numpy())


def create_heatmap_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """JET colormap + weighted blend with ``original_image``."""
    h, w, _ = original_image.shape

    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))

    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    original_float = original_image.astype(np.float32)
    heatmap_float = heatmap_color.astype(np.float32)

    overlay = cv2.addWeighted(
        src1=heatmap_float,
        alpha=alpha,
        src2=original_float,
        beta=1 - alpha,
        gamma=0,
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap_color, overlay


def generate_and_save_gradcam_for_image(
    model: tf.keras.Model,
    last_conv_layer: layers.Layer,
    img_path: str,
    true_class_name: str,
    class_names: List[str],
    output_dir: str,
    index_within_class: int,
) -> None:
    """Save a three-panel figure (MRI, heatmap, overlay) for one path."""
    original_image, input_tensor = load_and_preprocess_image(img_path)

    preds = model.predict(input_tensor, verbose=0)
    predicted_class_index = int(np.argmax(preds[0]))
    predicted_class_name = (
        class_names[predicted_class_index]
        if 0 <= predicted_class_index < len(class_names)
        else f"index_{predicted_class_index}"
    )

    heatmap, _ = compute_gradcam_heatmap(
        model=model,
        img_tensor=input_tensor,
        last_conv_layer=last_conv_layer,
    )

    heatmap_color, overlay = create_heatmap_overlay(
        original_image=original_image,
        heatmap=heatmap,
        alpha=0.4,
    )

    print(f"\n[INFO] Image: {img_path}")
    print(f"       True class:      {true_class_name}")
    print(f"       Predicted class: {predicted_class_name}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(heatmap_color)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("MRI + Grad-CAM Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    safe_class_name = true_class_name.lower()
    filename = f"gradcam_{safe_class_name}_{index_within_class}.png"
    output_path = os.path.join(output_dir, filename)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved Grad-CAM example to: {output_path}")


def main() -> None:
    ensure_project_directories()

    model_path = PROJECT_ROOT / "models" / "baseline_cnn.keras"
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test"
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "gradcam_examples"
    max_images_per_class = 10
    image_size = (224, 224)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    dummy_input = tf.zeros(
        (1, image_size[0], image_size[1], 3), dtype=tf.float32
    )
    _ = model(dummy_input, training=False)

    test_dir_str = str(test_dir)
    class_names = get_class_names(test_dir_str)
    print(f"[INFO] Detected class names: {class_names}")

    last_conv_layer = find_last_conv_layer(model)
    print(f"[INFO] Using last Conv2D layer for Grad-CAM: {last_conv_layer.name}")

    images_by_class = collect_image_paths_by_class(test_dir_str)

    out_str = str(output_dir)

    for class_name, image_paths in images_by_class.items():
        if not image_paths:
            print(f"[WARNING] No images found for class '{class_name}'. Skipping.")
            continue

        random.shuffle(image_paths)
        selected_paths = image_paths[:max_images_per_class]

        print(
            f"\n[INFO] Generating Grad-CAM examples for class '{class_name}' "
            f"(selected {len(selected_paths)} images)."
        )

        for idx, img_path in enumerate(selected_paths, start=1):
            try:
                generate_and_save_gradcam_for_image(
                    model=model,
                    last_conv_layer=last_conv_layer,
                    img_path=img_path,
                    true_class_name=class_name,
                    class_names=class_names,
                    output_dir=out_str,
                    index_within_class=idx,
                )
            except Exception as e:
                print(
                    f"[ERROR] Failed to process image '{img_path}': {e}"
                )


if __name__ == "__main__":
    main()

