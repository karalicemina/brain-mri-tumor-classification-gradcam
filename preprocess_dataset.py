# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI dataset — train/val/test split and resizing
# Author: [Your Name]
# University: [Your University]
# Purpose: Build folder-per-split dataset from processed PNGs for Keras loaders.
# -----------------------------------------------------------------------------
"""Split processed class folders into train/val/test with crop, normalize, and resize."""

import random
from pathlib import Path

import cv2
import numpy as np

from utils.paths import PROJECT_ROOT, ensure_project_directories

INPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "dataset"

IMG_SIZE = 224
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLASSES = ["glioma", "meningioma", "pituitary"]


def normalize_image(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def crop_brain(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y : y + h, x : x + w]
    return image


def preprocess_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    img = crop_brain(img)
    img = normalize_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def main() -> None:
    ensure_project_directories()

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (OUTPUT_FOLDER / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        class_path = INPUT_FOLDER / cls
        if not class_path.is_dir():
            print(f"[WARNING] Missing class folder: {class_path}")
            continue

        images = [p.name for p in class_path.iterdir() if p.is_file()]
        random.shuffle(images)
        total = len(images)
        if total == 0:
            print(f"[WARNING] No images in {class_path}")
            continue

        train_end = int(total * TRAIN_SPLIT)
        val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        splits = [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]

        for split_name, split_images in splits:
            for img_name in split_images:
                img_path = class_path / img_name
                processed_img = preprocess_image(img_path)
                save_path = OUTPUT_FOLDER / split_name / cls / img_name
                cv2.imwrite(str(save_path), processed_img)

    print("Dataset preprocessing finished!")


if __name__ == "__main__":
    main()
