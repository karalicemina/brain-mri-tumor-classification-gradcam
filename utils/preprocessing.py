"""MRI → ResNet50 input (OpenCV + ImageNet preprocess_input)."""

from typing import List, Tuple

import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input


def load_and_preprocess_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    batch = np.expand_dims(img_rgb_resized.astype(np.float32), axis=0)
    input_tensor = resnet_preprocess_input(batch.copy())

    return img_rgb_resized, input_tensor


def numpy_rgb_to_resnet_input(rgb: np.ndarray) -> np.ndarray:
    b = np.expand_dims(rgb.astype(np.float32), axis=0)
    return resnet_preprocess_input(b.copy())


def load_and_preprocess_custom_image(
    img_path: str, target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read custom image: {img_path}")

    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    preview_uint8 = rgb_resized.astype(np.uint8)
    input_tensor = numpy_rgb_to_resnet_input(preview_uint8)

    return preview_uint8, input_tensor


def create_tta_versions(input_tensor: np.ndarray) -> List[np.ndarray]:
    versions: List[np.ndarray] = []

    versions.append(input_tensor)

    flipped = input_tensor[:, :, ::-1, :]
    versions.append(flipped)

    h = input_tensor.shape[1]
    w = input_tensor.shape[2]
    crop_h = int(h * 0.9)
    crop_w = int(w * 0.9)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = input_tensor[:, start_y : start_y + crop_h, start_x : start_x + crop_w, :]

    resized = np.zeros_like(input_tensor)
    resized[0] = cv2.resize(cropped[0], (w, h))

    versions.append(resized)

    return versions
