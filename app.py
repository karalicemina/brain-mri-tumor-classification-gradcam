# -----------------------------------------------------------------------------
# PROJECT_INFO
# Title: Brain MRI tumor classification — Streamlit demonstration
# Author: [Emina Karalic]
# University: [Sarajevo School of Science and Technology]
# Purpose: Interactive thesis demo (inference + Grad-CAM + localization).
# -----------------------------------------------------------------------------
"""Streamlit: ResNet50 brain MRI classification + Grad-CAM."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.constants import CLASS_NAMES_THESIS
from utils.gradcam import compute_gradcam_heatmap
from utils.model_tools import ensure_model_built, resolve_model_path
from utils.paths import PROJECT_ROOT, ensure_project_directories
from utils.preprocessing import load_and_preprocess_image
from tumor_localization import localize_tumor_from_gradcam

ensure_project_directories()


class AnalysisResult(NamedTuple):
    original_rgb: np.ndarray
    probs: np.ndarray
    pred_idx: int
    confidence: float
    heatmap_rgb: np.ndarray
    localized_rgb: np.ndarray


DEMO_DIR = PROJECT_ROOT / "demo_examples"
DEMO_FILENAMES = {
    "glioma": DEMO_DIR / "glioma.png",
    "meningioma": DEMO_DIR / "meningioma.png",
    "pituitary": DEMO_DIR / "pituitary.png",
}
DEMO_ORDER = ("glioma", "meningioma", "pituitary")
IMAGE_SIZE: Tuple[int, int] = (224, 224)
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

_COL_BG_DEEP = "#060d18"
_COL_NAVY = "#0a1628"
_COL_CARD = "#0d1f35"
_COL_CYAN = "#00d4aa"
_COL_CYAN_BRIGHT = "#2ee6ff"
_COL_TEXT = "#f0fbff"
_COL_MUTED = "#8ba4b8"


def ensure_demo_examples_dir() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner="Loading neural network weights…")
def load_model_once() -> tf.keras.Model:
    path = resolve_model_path(PROJECT_ROOT, None)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Expected models/resnet50_model.keras."
        )
    model = tf.keras.models.load_model(path)
    ensure_model_built(model, IMAGE_SIZE)
    return model


def save_upload_to_temp(uploaded_bytes: bytes, suffix: str) -> str:
    tmp_root = Path(tempfile.gettempdir()) / "mri_thesis_streamlit_uploads"
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f"{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(uploaded_bytes)
    return str(out_path)


def preprocess_from_path(path: str) -> Tuple[np.ndarray, np.ndarray]:
    return load_and_preprocess_image(path, target_size=IMAGE_SIZE)


def pil_preview_from_path(path: str) -> Image.Image:
    pil = Image.open(path).convert("RGB")
    pil.thumbnail((420, 420), Image.Resampling.LANCZOS)
    return pil


def pil_preview_from_bytes(raw: bytes) -> Image.Image:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    pil.thumbnail((420, 420), Image.Resampling.LANCZOS)
    return pil


def run_inference(
    model: tf.keras.Model, input_batch: np.ndarray
) -> Tuple[np.ndarray, int, float]:
    probs = model.predict(input_batch, verbose=0)[0].astype(np.float64)
    k = int(np.argmax(probs))
    return probs, k, float(probs[k])


def gradcam_quiet(model: tf.keras.Model, input_batch: np.ndarray) -> np.ndarray:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        heatmap, _ = compute_gradcam_heatmap(model, input_batch, None)
    return heatmap


def analyze_from_disk_path(model: tf.keras.Model, img_path: str) -> AnalysisResult:
    """Preprocess, classify, Grad-CAM, and localize from a single image path."""
    original_rgb, input_batch = preprocess_from_path(img_path)
    probs, pred_idx, confidence = run_inference(model, input_batch)
    heatmap = gradcam_quiet(model, input_batch)
    _mask, localized_rgb, heatmap_rgb, _ok = localize_tumor_from_gradcam(
        original_image=original_rgb,
        heatmap=heatmap,
        percentile=82.0,
    )
    return AnalysisResult(
        original_rgb=original_rgb,
        probs=probs,
        pred_idx=pred_idx,
        confidence=confidence,
        heatmap_rgb=np.clip(heatmap_rgb, 0, 255).astype(np.uint8),
        localized_rgb=np.clip(localized_rgb, 0, 255).astype(np.uint8),
    )


def apply_dark_medical_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(
                    ellipse 120% 80% at 50% -10%,
                    rgba(46, 230, 255, 0.12) 0%,
                    {_COL_BG_DEEP} 45%,
                    {_COL_NAVY} 100%
                ) !important;
            }}
            [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                background: transparent !important;
            }}
            .block-container {{
                padding-top: 1.25rem;
                padding-bottom: 4rem;
                max-width: 1120px;
            }}
            .main .block-container {{
                color: {_COL_TEXT};
            }}
            .stMarkdown, .stMarkdown p, .stMarkdown li {{
                color: {_COL_TEXT} !important;
            }}
            div[data-testid="stDecoration"] {{
                display: none;
            }}
            div[data-testid="stFileUploader"] {{
                background: {_COL_CARD} !important;
                border-radius: 16px !important;
                padding: 1rem 1.1rem !important;
                border: 1px solid rgba(46, 230, 255, 0.25) !important;
                box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
            }}
            div[data-testid="stFileUploader"] label,
            div[data-testid="stFileUploader"] small {{
                color: {_COL_MUTED} !important;
            }}
            div[data-testid="stAlert"] {{
                border-radius: 12px !important;
            }}
            div[data-testid="stImage"] img {{
                border-radius: 16px !important;
                box-shadow: 0 8px 28px rgba(0, 212, 170, 0.18),
                            0 0 40px rgba(46, 230, 255, 0.08) !important;
                object-fit: contain !important;
                max-height: 300px !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }}
            div[data-testid="stCaption"] {{
                color: {_COL_MUTED} !important;
                text-align: center !important;
                font-weight: 500 !important;
                letter-spacing: 0.02em;
            }}
            h3 {{
                color: {_COL_CYAN_BRIGHT} !important;
                font-weight: 600 !important;
                letter-spacing: -0.02em;
                border-bottom: 1px solid rgba(46, 230, 255, 0.2);
                padding-bottom: 0.35rem;
            }}
            .hero-wrap {{
                position: relative;
                text-align: center;
                padding: 2.5rem 1.5rem 2rem;
                margin: 0 auto 2rem;
                max-width: 900px;
                border-radius: 24px;
                background: linear-gradient(
                    145deg,
                    rgba(13, 31, 53, 0.95) 0%,
                    rgba(10, 22, 40, 0.88) 50%,
                    rgba(8, 24, 40, 0.92) 100%
                );
                border: 1px solid rgba(46, 230, 255, 0.22);
                box-shadow: 0 0 60px rgba(0, 212, 170, 0.07),
                            inset 0 1px 0 rgba(255, 255, 255, 0.06);
                overflow: hidden;
            }}
            .hero-wrap::before {{
                content: "";
                position: absolute;
                inset: -40% -20% auto -20%;
                height: 70%;
                background: radial-gradient(
                    ellipse at 50% 0%,
                    rgba(46, 230, 255, 0.15) 0%,
                    transparent 55%
                );
                pointer-events: none;
            }}
            .hero-title {{
                position: relative;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-size: clamp(1.75rem, 4.2vw, 2.55rem);
                font-weight: 800;
                line-height: 1.22;
                color: #ffffff !important;
                text-shadow: 0 0 40px rgba(46, 230, 255, 0.45),
                             0 0 80px rgba(0, 212, 170, 0.2),
                             0 2px 4px rgba(0, 0, 0, 0.45);
                margin: 0 0 0.65rem;
                letter-spacing: -0.03em;
            }}
            .hero-subtitle {{
                position: relative;
                font-size: clamp(0.95rem, 2vw, 1.125rem);
                font-weight: 400;
                color: rgba(255, 255, 255, 0.78) !important;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin: 0;
            }}
            .disclaimer-card {{
                position: relative;
                background: rgba(13, 31, 53, 0.75);
                border-left: 4px solid {_COL_CYAN};
                border-radius: 12px;
                padding: 1rem 1.15rem;
                color: rgba(240, 251, 255, 0.92) !important;
                font-size: 0.9rem;
                line-height: 1.55;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
                margin-bottom: 1.75rem;
            }}
            .metric-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                justify-content: center;
                margin: 1.5rem 0 2rem;
            }}
            .metric-card {{
                flex: 1 1 200px;
                max-width: 340px;
                background: {_COL_CARD};
                border-radius: 16px;
                padding: 1.25rem 1.4rem;
                border: 1px solid rgba(46, 230, 255, 0.18);
                box-shadow: 0 10px 32px rgba(0, 0, 0, 0.35),
                            inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }}
            .metric-card-accent {{
                border-top: 3px solid {_COL_CYAN};
            }}
            .metric-card-green {{
                border-top: 3px solid #4cd964;
            }}
            .metric-label {{
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: {_COL_MUTED};
                margin-bottom: 0.35rem;
            }}
            .metric-value {{
                font-size: 1.45rem;
                font-weight: 700;
                color: #ffffff;
                text-shadow: 0 0 24px rgba(46, 230, 255, 0.25);
            }}
            .metric-value-small {{
                font-size: 1.05rem;
                font-weight: 600;
                color: rgba(240, 251, 255, 0.95);
            }}
            .viz-label {{
                text-align: center;
                font-size: 0.92rem;
                font-weight: 600;
                color: {_COL_CYAN_BRIGHT};
                letter-spacing: 0.06em;
                margin: 0.5rem 0 0.25rem;
                text-transform: uppercase;
            }}
            .section-rule {{
                border: none;
                height: 1px;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(46, 230, 255, 0.35),
                    rgba(0, 212, 170, 0.25),
                    transparent
                );
                margin: 2rem 0 2rem;
            }}
            .help-muted {{
                color: {_COL_MUTED};
                font-size: 0.88rem;
                margin: -0.5rem 0 1rem;
                line-height: 1.45;
            }}
            .thesis-footer {{
                text-align: center;
                padding: 2.5rem 1rem 1rem;
                margin-top: 2rem;
                color: {_COL_MUTED};
                font-size: 0.88rem;
                border-top: 1px solid rgba(46, 230, 255, 0.12);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<p class="thesis-footer">Developed as part of a Bachelor Thesis at SSST</p>',
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer-card"><strong>Disclaimer:</strong> This system is '
        "for research and educational purposes only and is not a clinical "
        "diagnostic tool.</div>",
        unsafe_allow_html=True,
    )


def render_prediction_cards(predicted_label: str, confidence_pct: float) -> None:
    conf_str = f"{confidence_pct:.2f}%"
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card metric-card-accent">
                <div class="metric-label">Predicted Tumor Type</div>
                <div class="metric-value">{predicted_label}</div>
            </div>
            <div class="metric-card metric-card-accent">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{conf_str}</div>
            </div>
            <div class="metric-card metric-card-green">
                <div class="metric-label">Model Used</div>
                <div class="metric-value-small">ResNet50 Transfer Learning</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probabilities_figure_dark(probs: np.ndarray) -> plt.Figure:
    labels = [CLASS_NAMES_THESIS[i].capitalize() for i in range(len(CLASS_NAMES_THESIS))]
    values = [float(probs[i]) * 100.0 for i in range(len(CLASS_NAMES_THESIS))]
    colors = [_COL_CYAN, _COL_CYAN_BRIGHT, "#4cd964"]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    fig.patch.set_facecolor(_COL_CARD)
    ax.set_facecolor(_COL_CARD)
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55, alpha=0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability / %", color=_COL_MUTED, fontsize=10)
    ax.tick_params(colors=_COL_TEXT, labelsize=10)
    ax.spines["bottom"].set_color(_COL_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_COL_MUTED)
    ax.set_title("Class probabilities", color=_COL_CYAN_BRIGHT, fontsize=12, fontweight="bold", pad=10)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", color=_COL_TEXT, fontsize=9)
    plt.tight_layout()
    return fig


def render_outputs(res: AnalysisResult) -> None:
    pred_name = CLASS_NAMES_THESIS[res.pred_idx].capitalize()
    render_prediction_cards(pred_name, res.confidence * 100.0)
    st.markdown("### Probability distribution")
    fig_probs = probabilities_figure_dark(res.probs)
    st.pyplot(fig_probs, transparent=True)
    plt.close(fig_probs)
    st.caption(
        "Grad-CAM is qualitative; localization shows regions weighted by the classifier."
    )
    st.markdown("### Grad-CAM & localization")
    imgs = (res.original_rgb, res.heatmap_rgb, res.localized_rgb)
    lc, mc, rc = st.columns(3)
    with lc:
        st.markdown('<p class="viz-label">Original MRI</p>', unsafe_allow_html=True)
        st.image(imgs[0], use_container_width=True)
    with mc:
        st.markdown('<p class="viz-label">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
        st.image(imgs[1], use_container_width=True)
    with rc:
        st.markdown('<p class="viz-label">Tumor Localization</p>', unsafe_allow_html=True)
        st.image(imgs[2], use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="MRI Tumor Classification • Grad-CAM",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    ensure_demo_examples_dir()
    if "demo_choice" not in st.session_state:
        st.session_state.demo_choice = None

    apply_dark_medical_theme()

    render_hero(
        "Brain MRI Tumor Classification and Grad-CAM Analysis",
        "Bachelor Thesis Demonstration System",
    )
    render_disclaimer()

    try:
        model = load_model_once()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    st.markdown(
        '<p style="color:#8ba4b8;font-size:0.9rem;margin:-0.5rem 0 1rem;">'
        "Demo mode uses curated files from <code>demo_examples/</code>. "
        "An uploaded file overrides the last demo selection."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### Try Prepared Demo Examples")
    d1, d2, d3 = st.columns(3)

    demos_ok = [DEMO_FILENAMES[k].is_file() for k in DEMO_ORDER]

    with d1:
        if st.button("Glioma example", use_container_width=True, disabled=not demos_ok[0]):
            if demos_ok[0]:
                st.session_state.demo_choice = "glioma"
    with d2:
        if st.button(
            "Meningioma example", use_container_width=True, disabled=not demos_ok[1]
        ):
            if demos_ok[1]:
                st.session_state.demo_choice = "meningioma"
    with d3:
        if st.button(
            "Pituitary example", use_container_width=True, disabled=not demos_ok[2]
        ):
            if demos_ok[2]:
                st.session_state.demo_choice = "pituitary"

    if not all(demos_ok):
        st.warning(
            f"Missing demo PNGs under `{DEMO_DIR}`. "
            "Add glioma.png, meningioma.png, pituitary.png (see demo_examples/README.txt)."
        )

    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)

    st.markdown("### Upload your own MRI")
    st.markdown(
        '<p class="help-muted">PNG or JPEG · same preprocessing as training '
        "(recommended ≥224×). Upload overrides demo selection.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose image file",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in ALLOWED_EXT:
            st.warning("Use .png, .jpg, or .jpeg.")
            render_footer()
            return
        st.session_state.demo_choice = None
        raw_bytes = uploaded.getvalue()
        st.markdown(
            '<p class="viz-label" style="margin-top:0.5rem;">Preview</p>',
            unsafe_allow_html=True,
        )
        st.image(pil_preview_from_bytes(raw_bytes), caption=None, use_container_width=True)

        temp_path: Optional[str] = None
        try:
            with st.spinner("Analyzing MRI image and generating Grad-CAM visualization..."):
                temp_path = save_upload_to_temp(raw_bytes, suffix)

                try:
                    res = analyze_from_disk_path(model, temp_path)
                except ValueError as exc:
                    st.error(f"Could not read image: {exc}")
                    render_footer()
                    return
                render_outputs(res)
        finally:
            if temp_path and Path(temp_path).is_file():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        render_footer()
        return

    choice = st.session_state.demo_choice
    demo_path = DEMO_FILENAMES.get(choice) if choice in DEMO_FILENAMES else None

    if choice and demo_path and demo_path.is_file():
        st.markdown(
            f'<p class="viz-label" style="margin-top:0.5rem;">Demo: {choice.capitalize()}</p>',
            unsafe_allow_html=True,
        )
        try:
            st.image(pil_preview_from_path(str(demo_path)), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not preview demo file: {exc}")
            render_footer()
            return

        try:
            with st.spinner("Analyzing MRI image and generating Grad-CAM visualization..."):
                res = analyze_from_disk_path(model, str(demo_path))
        except ValueError as exc:
            st.error(f"Could not read demo image: {exc}")
            render_footer()
            return

        render_outputs(res)
        render_footer()
        return

    st.info("Select a prepared demo above or upload an MRI.")
    render_footer()


main()
