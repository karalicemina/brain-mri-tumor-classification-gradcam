"""Project root and standard directory layout (pathlib)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_directories() -> None:
    """Create standard project folders (data, models, outputs, docs, demos)."""
    subdirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data" / "dataset",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "demo_examples",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "outputs" / "metrics",
        PROJECT_ROOT / "outputs" / "final_results",
        PROJECT_ROOT / "screenshots",
        PROJECT_ROOT / "docs" / "poster",
        PROJECT_ROOT / "docs" / "thesis_figures",
    ]
    for p in subdirs:
        p.mkdir(parents=True, exist_ok=True)


def resnet_model_path() -> Path:
    return PROJECT_ROOT / "models" / "resnet50_model.keras"


def baseline_model_path() -> Path:
    return PROJECT_ROOT / "models" / "baseline_cnn.keras"
