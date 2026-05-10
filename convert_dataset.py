# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI dataset — .mat to PNG conversion
# Author: [Your Name]
# University: [Your University]
# Purpose: Export public brain tumor .mat volumes to class folders for preprocessing.
# -----------------------------------------------------------------------------
"""Convert legacy .mat brain tumor archives under data/ to PNG files in data/processed/."""

from pathlib import Path

import cv2
import h5py
import numpy as np

from utils.paths import PROJECT_ROOT, ensure_project_directories

LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}

ROOT_DATA = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    ensure_project_directories()

    for class_name in LABEL_MAP.values():
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    if not ROOT_DATA.is_dir():
        print(f"[ERROR] Data root not found: {ROOT_DATA}")
        return

    for dataset_folder in ROOT_DATA.iterdir():
        if not dataset_folder.is_dir():
            continue
        if not dataset_folder.name.startswith("brainTumorDataPublic"):
            continue

        print("Processing folder:", dataset_folder)

        for file_path in dataset_folder.iterdir():
            if file_path.suffix.lower() != ".mat":
                continue

            with h5py.File(file_path, "r") as f:
                cjdata = f["cjdata"]
                image = np.array(cjdata["image"]).T
                label = int(np.array(cjdata["label"])[0][0])

            if label not in LABEL_MAP:
                continue

            class_name = LABEL_MAP[label]
            save_path = OUTPUT_DIR / class_name / (file_path.stem + ".png")
            cv2.imwrite(str(save_path), image)

    print("Dataset conversion finished.")


if __name__ == "__main__":
    main()
