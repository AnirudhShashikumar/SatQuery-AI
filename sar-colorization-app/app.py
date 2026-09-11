"""Streamlit interface for independent Pix2Pix and SARFusionFormer workflows."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Optional

import numpy as np
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

API_URL = os.environ.get("SAR_COLORIZATION_API_URL", "http://127.0.0.1:8010").rstrip("/")
PIX2PIX_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
SAR_TYPES = ["npy", "png", "jpg", "jpeg", "tif", "tiff"]

st.set_page_config(page_title="SAR Optical Reconstruction", page_icon="🛰️", layout="wide")


def initialise_state() -> None:
    defaults = {
        "pix2pix_state": {
            "input_preview": None,
            "output": None,
            "ground_truth": None,
            "metrics": None,
            "inference_time_ms": None,
            "checkpoint": None,
            "error": None,
        },
        "sarfusionformer_state": {
            "vv_preview": None,
            "vh_preview": None,
            "sar_preview": None,
            "raw_output": None,
            "display_output": None,
            "corrected_raw_output": None,
            "corrected_display_output": None,
            "ground_truth": None,
            "metrics_raw": None,
            "metrics_corrected": None,
            "inference_time_ms": None,
            "checkpoint": None,
            "color_checkpoint": None,
            "warning": None,
            "error": None,
            "input_mode": "Combined VV/VH .npy",
            "detected_shape": None,
            "channel_layout": None,
            "selected_output_mode": "Raw SARFusionFormer",
            "selected_visualization_mode": "Enhanced display",
            "diagnostics": None,
            "corrected_diagnostics": None,
        },
        "comparison_state": {
            "manual_pix2pix": None,
            "manual_sarfusionformer": None,
            "common_ground_truth": None,
            "selected_sarfusionformer_mode": "Raw SARFusionFormer",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                st.session_state[key].setdefault(nested_key, nested_value)


def image_from_b64(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def b64_from_upload(upload) -> Optional[str]:
    if upload is None:
        return None
    return base64.b64encode(upload.getvalue()).decode("ascii")


def api_error(error: requests.HTTPError) -> str:
    try:
        return error.response.json().get("detail", error.response.text)
    except ValueError:
        return error.response.text


def post_json(
    route: str, files: Dict[str, Any], data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    response = requests.post(
        "{}/{}".format(API_URL, route), files=files, data=data, timeout=180
    )
    response.raise_for_status()
    return response.json()


def preview_upload(upload, grayscale: bool = False) -> Optional[Image.Image]:
    if upload is None or upload.name.lower().endswith(".npy"):
        return None
    try:
        image = Image.open(io.BytesIO(upload.getvalue()))
        image.load()
        return image.convert("L" if grayscale else "RGB")
    except (UnidentifiedImageError, OSError):
        return None


def render_image(value: Optional[str], label: str) -> None:
    if value:
        st.image(image_from_b64(value), caption=label, width="stretch")
    else:
        st.caption("{} unavailable".format(label))


def render_metrics(metrics: Optional[Dict[str, float]]) -> None:
    if not metrics:
        st.caption("Upload a ground-truth optical image to calculate PSNR, SSIM, and RGB L1.")
        return
    columns = st.columns(3)
    columns[0].metric("PSNR", "{:.2f} dB".format(metrics["psnr"]) if metrics["psnr"] is not None else "N/A")
    columns[1].metric("SSIM", "{:.4f}".format(metrics["ssim"]))
    columns[2].metric("RGB L1", "{:.4f}".format(metrics["rgb_l1"]))


def model_health() -> Dict[str, Any]:
    try:
        response = requests.get("{}/health".format(API_URL), timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "offline", "models": {}}


def current_pix2pix_result() -> Optional[Dict[str, Any]]:
    state = st.session_state.pix2pix_state
    output = state["output"] or st.session_state.comparison_state["manual_pix2pix"]
    if not output:
        return None
    return {
        "output": output,
        "input_preview": state["input_preview"],
        "metrics": state["metrics"],
        "time": state["inference_time_ms"],
        "checkpoint": state["checkpoint"] or "Manual upload",
    }


def current_sarfusionformer_result() -> Optional[Dict[str, Any]]:
    state = st.session_state.sarfusionformer_state
    comparison = st.session_state.comparison_state
    mode = comparison["selected_sarfusionformer_mode"]
    visualization = state.get("selected_visualization_mode", "Enhanced display")
    corrected_raw_output = state.get("corrected_raw_output")
    if mode == "Color-corrected SARFusionFormer" and corrected_raw_output:
        output = corrected_raw_output
        display_output = (
            state.get("corrected_display_output")
            if visualization == "Enhanced display"
            else corrected_raw_output
        ) or corrected_raw_output
        metrics = state["metrics_corrected"]
    else:
        output = state["raw_output"] or comparison["manual_sarfusionformer"]
        display_output = (
            state.get("display_output")
            if visualization == "Enhanced display"
            else state["raw_output"]
        ) or output
        metrics = state["metrics_raw"]
    if not output:
        return None
    return {
        "output": output,
        "display_output": display_output,
        "vv_preview": state["vv_preview"],
        "vh_preview": state["vh_preview"],
        "sar_preview": state["sar_preview"],
        "metrics": metrics,
        "time": state["inference_time_ms"],
        "checkpoint": state["checkpoint"] or "Manual upload",
        "mode": mode,
    }


def comparison_metrics(output: str, target: str) -> Dict[str, float]:
    prediction = image_from_b64(output).convert("RGB")
    ground_truth = image_from_b64(target).convert("RGB").resize(
        prediction.size, Image.Resampling.BICUBIC
    )
    predicted_array = np.asarray(prediction, dtype=np.float32) / 255.0
    target_array = np.asarray(ground_truth, dtype=np.float32) / 255.0
    return {
        "psnr": float(peak_signal_noise_ratio(target_array, predicted_array, data_range=1.0)),
        "ssim": float(
            structural_similarity(target_array, predicted_array, channel_axis=2, data_range=1.0)
        ),
        "rgb_l1": float(np.mean(np.abs(target_array - predicted_array))),
    }


def pix2pix_page() -> None:
    state = st.session_state.pix2pix_state
    st.title("Pix2Pix")
    st.caption("Visual-quality optical reconstruction")
    st.info(
        "Pix2Pix prioritizes visually realistic optical reconstruction. It may generate "
        "plausible visual details that are not directly observable in the SAR input."
    )

    input_file = st.file_uploader(
        "Pix2Pix SAR input", type=PIX2PIX_TYPES, key="pix2pix_input"
    )
    ground_truth = st.file_uploader(
        "Optional ground-truth optical image", type=PIX2PIX_TYPES, key="pix2pix_ground_truth"
    )
    if input_file:
        input_column, output_column = st.columns(2)
        with input_column:
            preview = preview_upload(input_file)
            if preview:
                st.image(preview, caption="Pix2Pix SAR input", width="stretch")
        with output_column:
            render_image(state["output"], "Generated optical image")

    if st.button("Generate Pix2Pix result", type="primary", key="pix2pix_generate"):
        if not input_file:
            state["error"] = "Upload a Pix2Pix SAR image before running inference."
        else:
            files = {
                "file": (input_file.name, input_file.getvalue(), input_file.type or "image/png")
            }
            if ground_truth:
                files["ground_truth"] = (
                    ground_truth.name,
                    ground_truth.getvalue(),
                    ground_truth.type or "image/png",
                )
            with st.spinner("Running Pix2Pix inference…"):
                try:
                    payload = post_json("api/pix2pix/infer", files)
                    state.update(
                        {
                            "input_preview": payload["input_preview"],
                            "output": payload["output"],
                            "ground_truth": b64_from_upload(ground_truth),
                            "metrics": payload["metrics"],
                            "inference_time_ms": payload["inference_time_ms"],
                            "checkpoint": payload["checkpoint"],
                            "error": None,
                        }
                    )
                except requests.HTTPError as error:
                    state["error"] = api_error(error)
                except requests.RequestException:
                    state["error"] = "The Pix2Pix backend could not be reached."

    if state["error"]:
        st.error(state["error"])
    if state["output"]:
        st.subheader("Latest Pix2Pix result")
        result_column, target_column = st.columns(2)
        with result_column:
            render_image(state["output"], "Generated optical image")
        with target_column:
            render_image(state["ground_truth"], "Ground-truth optical image")
        render_metrics(state["metrics"])
        details_left, details_right = st.columns(2)
        details_left.caption("Checkpoint: {}".format(state["checkpoint"]))
        details_right.caption("Inference time: {:.2f} ms".format(state["inference_time_ms"]))
        st.download_button(
            "Download Pix2Pix PNG",
            data=base64.b64decode(state["output"]),
            file_name="pix2pix_optical.png",
            mime="image/png",
        )


def sarfusionformer_page() -> None:
    state = st.session_state.sarfusionformer_state
    st.title("SARFusionFormer — Structure-Preserving Reconstruction")
    st.info(
        "SARFusionFormer emphasizes terrain geometry and structural consistency. The result "
        "is an AI-generated optical reconstruction and is not exact ground truth."
    )

    input_mode = st.radio(
        "SARFusionFormer input mode",
        ["Combined VV/VH .npy", "Separate VV and VH files"],
        index=0,
        key="sarfusionformer_input_mode",
        horizontal=True,
    )
    state["input_mode"] = input_mode
    combined_file = None
    vv_file = None
    vh_file = None
    if input_mode == "Combined VV/VH .npy":
        combined_file = st.file_uploader(
            "Upload combined VV/VH NumPy file",
            type=["npy"],
            key="sarfusionformer_combined",
        )
        st.caption(
            "Supported layouts: [2, H, W], [H, W, 2], [1, 2, H, W], and [1, H, W, 2]."
        )
    else:
        upload_left, upload_right = st.columns(2)
        with upload_left:
            vv_file = st.file_uploader("Upload VV", type=SAR_TYPES, key="sarfusionformer_vv")
        with upload_right:
            vh_file = st.file_uploader("Upload VH", type=SAR_TYPES, key="sarfusionformer_vh")
    ground_truth = st.file_uploader(
        "Optional ground-truth optical image", type=PIX2PIX_TYPES, key="sarfusionformer_ground_truth"
    )
    apply_color_correction = st.checkbox(
        "Apply optional colour correction",
        value=False,
        key="sarfusionformer_apply_color_correction",
        help="Off by default so the raw SARFusionFormer output can be verified first.",
    )

    if vv_file and vh_file:
        preview_left, preview_right = st.columns(2)
        with preview_left:
            preview = preview_upload(vv_file, grayscale=True)
            if preview:
                st.image(preview, caption="VV input", width="stretch")
            else:
                st.caption("VV preview is generated after normalization.")
        with preview_right:
            preview = preview_upload(vh_file, grayscale=True)
            if preview:
                st.image(preview, caption="VH input", width="stretch")
            else:
                st.caption("VH preview is generated after normalization.")

    if st.button("Generate structural reconstruction", type="primary", key="sarfusionformer_generate"):
        if input_mode == "Combined VV/VH .npy" and not combined_file:
            state["error"] = "Upload one combined VV/VH NumPy file before running inference."
        elif input_mode == "Separate VV and VH files" and not vv_file:
            state["error"] = "Upload a VV input before running inference."
        elif input_mode == "Separate VV and VH files" and not vh_file:
            state["error"] = "Upload a VH input before running inference."
        else:
            if input_mode == "Combined VV/VH .npy":
                files = {
                    "combined_file": (
                        combined_file.name,
                        combined_file.getvalue(),
                        combined_file.type or "application/octet-stream",
                    )
                }
            else:
                files = {
                    "vv_file": (
                        vv_file.name,
                        vv_file.getvalue(),
                        vv_file.type or "application/octet-stream",
                    ),
                    "vh_file": (
                        vh_file.name,
                        vh_file.getvalue(),
                        vh_file.type or "application/octet-stream",
                    ),
                }
            if ground_truth:
                files["ground_truth"] = (
                    ground_truth.name,
                    ground_truth.getvalue(),
                    ground_truth.type or "image/png",
                )
            with st.spinner("Running SARFusionFormer inference…"):
                try:
                    payload = post_json(
                        "api/sarfusionformer/infer",
                        files,
                        data={"apply_color_correction": str(apply_color_correction).lower()},
                    )
                    state.update(
                        {
                            "vv_preview": payload["vv_preview"],
                            "vh_preview": payload["vh_preview"],
                            "sar_preview": payload["sar_preview"],
                            "raw_output": payload["raw_output"],
                            "display_output": payload["display_output"],
                            "corrected_raw_output": payload["corrected_raw_output"],
                            "corrected_display_output": payload["corrected_display_output"],
                            "ground_truth": b64_from_upload(ground_truth),
                            "metrics_raw": payload["metrics_raw"],
                            "metrics_corrected": payload["metrics_corrected"],
                            "inference_time_ms": payload["inference_time_ms"],
                            "checkpoint": payload["checkpoint"],
                            "color_checkpoint": payload["color_checkpoint"],
                            "warning": payload["warning"],
                            "detected_shape": payload["detected_shape"],
                            "channel_layout": payload["channel_layout"],
                            "diagnostics": payload["diagnostics"],
                            "corrected_diagnostics": payload["corrected_diagnostics"],
                            "error": None,
                        }
                    )
                except requests.HTTPError as error:
                    state["error"] = api_error(error)
                except requests.RequestException:
                    state["error"] = "The SARFusionFormer backend could not be reached."

    if state["error"]:
        st.error(state["error"])
    if state["warning"]:
        st.warning(state["warning"])
    if state["raw_output"]:
        st.subheader("Latest structural result")
        st.caption(
            "Detected array shape: {} · Channel layout: {} · channel 0 = VV, channel 1 = VH".format(
                state["detected_shape"], state["channel_layout"]
            )
        )
        selected = st.radio(
            "Model output",
            ["Raw SARFusionFormer", "Color-corrected SARFusionFormer"]
            if state["corrected_raw_output"]
            else ["Raw SARFusionFormer"],
            key="sarfusionformer_display_mode",
            horizontal=True,
        )
        state["selected_output_mode"] = selected
        visualization = st.radio(
            "Display",
            ["Enhanced display", "Raw radiometric output"],
            index=0,
            key="sarfusionformer_visualization_mode",
            horizontal=True,
        )
        state["selected_visualization_mode"] = visualization
        is_corrected = selected == "Color-corrected SARFusionFormer" and state["corrected_raw_output"]
        raw_output = state["corrected_raw_output"] if is_corrected else state["raw_output"]
        enhanced_output = (
            state["corrected_display_output"] if is_corrected else state["display_output"]
        )
        output = enhanced_output if visualization == "Enhanced display" else raw_output
        metrics = (
            state["metrics_corrected"]
            if is_corrected
            else state["metrics_raw"]
        )
        previews = st.columns(3)
        with previews[0]:
            render_image(state["vv_preview"], "VV preview")
        with previews[1]:
            render_image(state["vh_preview"], "VH preview")
        with previews[2]:
            render_image(state["sar_preview"], "Combined SAR preview")
        render_image(
            output,
            "Enhanced colour-corrected structural reconstruction"
            if is_corrected and visualization == "Enhanced display"
            else "Raw radiometric colour-corrected structural reconstruction"
            if is_corrected
            else "Enhanced SARFusionFormer structural reconstruction"
            if visualization == "Enhanced display"
            else "Raw radiometric SARFusionFormer structural reconstruction",
        )
        st.info(
            "Enhanced display uses percentile contrast stretching for visualization only. "
            "Raw radiometric output is the direct network prediction."
        )
        render_image(state["ground_truth"], "Ground-truth optical image")
        render_metrics(metrics)
        detail_left, detail_right = st.columns(2)
        detail_left.caption("Checkpoint: {}".format(state["checkpoint"]))
        detail_right.caption("Inference time: {:.2f} ms".format(state["inference_time_ms"]))
        if state["color_checkpoint"]:
            st.caption("Color-corrector checkpoint: {}".format(state["color_checkpoint"]))
        selected_diagnostics = (
            state.get("corrected_diagnostics") if is_corrected else state.get("diagnostics")
        )
        if selected_diagnostics:
            statistics = st.columns(3)
            statistics[0].metric(
                "Raw prediction min", "{:.5f}".format(selected_diagnostics["raw_rgb_min"])
            )
            statistics[1].metric(
                "Raw prediction max", "{:.5f}".format(selected_diagnostics["raw_rgb_max"])
            )
            statistics[2].metric(
                "Raw prediction mean", "{:.5f}".format(selected_diagnostics["raw_rgb_mean"])
            )
            statistics = st.columns(3)
            statistics[0].metric(
                "Raw prediction std", "{:.5f}".format(selected_diagnostics["raw_rgb_std"])
            )
            statistics[1].metric(
                "Enhanced prediction min",
                "{:.5f}".format(selected_diagnostics["display_rgb_min"]),
            )
            statistics[2].metric(
                "Enhanced prediction max",
                "{:.5f}".format(selected_diagnostics["display_rgb_max"]),
            )
            with st.expander("Output diagnostics"):
                st.json(selected_diagnostics)
        st.download_button(
            "Download raw scientific PNG",
            data=base64.b64decode(raw_output),
            file_name=(
                "sarfusionformer_color_corrected_raw.png"
                if is_corrected
                else "sarfusionformer_raw.png"
            ),
            mime="image/png",
        )


def comparison_page() -> None:
    comparison = st.session_state.comparison_state
    st.title("Model Comparison")
    st.info(
        "AI-generated optical reconstructions may contain plausible but incorrect visual "
        "details and should not be treated as exact ground truth."
    )
    pix = current_pix2pix_result()
    sar = current_sarfusionformer_result()

    controls = st.columns(3)
    with controls[0]:
        manual_pix = st.file_uploader(
            "Manual Pix2Pix result", type=PIX2PIX_TYPES, key="comparison_manual_pix"
        )
        if manual_pix:
            comparison["manual_pix2pix"] = b64_from_upload(manual_pix)
            pix = current_pix2pix_result()
    with controls[1]:
        manual_sar = st.file_uploader(
            "Manual SARFusionFormer result", type=PIX2PIX_TYPES, key="comparison_manual_sar"
        )
        if manual_sar:
            comparison["manual_sarfusionformer"] = b64_from_upload(manual_sar)
            sar = current_sarfusionformer_result()
    with controls[2]:
        common_ground_truth = st.file_uploader(
            "Common ground-truth optical image", type=PIX2PIX_TYPES, key="comparison_ground_truth"
        )
        if common_ground_truth:
            comparison["common_ground_truth"] = b64_from_upload(common_ground_truth)

    if st.session_state.sarfusionformer_state.get("corrected_raw_output"):
        comparison["selected_sarfusionformer_mode"] = st.radio(
            "SARFusionFormer comparison output",
            ["Raw SARFusionFormer", "Color-corrected SARFusionFormer"],
            horizontal=True,
            key="comparison_sarfusionformer_mode",
        )
        sar = current_sarfusionformer_result()

    pix_column, sar_column, target_column = st.columns(3)
    with pix_column:
        st.subheader("Pix2Pix result")
        if pix:
            render_image(pix["input_preview"], "Input preview")
            render_image(pix["output"], "Generated optical image")
            st.caption("Higher visual realism")
            st.caption("Checkpoint: {}".format(pix["checkpoint"]))
            st.caption(
                "Inference time: {}".format(
                    "{:.2f} ms".format(pix["time"]) if pix["time"] is not None else "N/A"
                )
            )
            render_metrics(pix["metrics"])
        else:
            st.info("Generate a Pix2Pix result or upload one above.")
    with sar_column:
        st.subheader("SARFusionFormer result")
        if sar:
            preview_left, preview_right = st.columns(2)
            with preview_left:
                render_image(sar["vv_preview"], "VV preview")
            with preview_right:
                render_image(sar["vh_preview"], "VH preview")
            render_image(sar["sar_preview"], "Combined VV/VH preview")
            render_image(sar["display_output"], "Structural reconstruction")
            st.caption("Displayed with the selected visualisation mode; metrics below use raw RGB.")
            st.caption("Higher structural consistency")
            st.caption("Checkpoint: {}".format(sar["checkpoint"]))
            st.caption(
                "Inference time: {}".format(
                    "{:.2f} ms".format(sar["time"]) if sar["time"] is not None else "N/A"
                )
            )
            render_metrics(sar["metrics"])
        else:
            st.info("Generate a SARFusionFormer result or upload one above.")
    with target_column:
        st.subheader("Ground truth")
        target = comparison["common_ground_truth"]
        if target:
            image = image_from_b64(target)
            st.image(image, caption="Common ground-truth optical image", width="stretch")
            st.caption("Source: comparison upload · {} × {}".format(*image.size))
        else:
            st.caption("No common ground truth supplied.")

    target = comparison["common_ground_truth"]
    if not (pix and sar and target):
        st.warning(
            "Metrics require a common ground-truth optical image evaluated at the same spatial resolution."
        )
        return

    pix_metrics = comparison_metrics(pix["output"], target)
    sar_metrics = comparison_metrics(sar["output"], target)
    times = [pix["time"], sar["time"]]
    best_psnr = max(pix_metrics["psnr"], sar_metrics["psnr"])
    best_ssim = max(pix_metrics["ssim"], sar_metrics["ssim"])
    best_l1 = min(pix_metrics["rgb_l1"], sar_metrics["rgb_l1"])
    numeric_times = [value for value in times if value is not None]
    fastest = min(numeric_times) if numeric_times else None

    def mark(value: float, best: float, suffix: str = "") -> str:
        is_best = value == best
        return "{:.4f}{}{}".format(value, suffix, " ★" if is_best else "")

    rows = [
        {
            "Model": "Pix2Pix",
            "PSNR": mark(pix_metrics["psnr"], best_psnr, suffix=" dB"),
            "SSIM": mark(pix_metrics["ssim"], best_ssim),
            "RGB L1": mark(pix_metrics["rgb_l1"], best_l1),
            "Inference Time": (
                "{:.2f} ms{}".format(pix["time"], " ★" if pix["time"] == fastest else "")
                if pix["time"] is not None
                else "N/A"
            ),
        },
        {
            "Model": "SARFusionFormer",
            "PSNR": mark(sar_metrics["psnr"], best_psnr, suffix=" dB"),
            "SSIM": mark(sar_metrics["ssim"], best_ssim),
            "RGB L1": mark(sar_metrics["rgb_l1"], best_l1),
            "Inference Time": (
                "{:.2f} ms{}".format(sar["time"], " ★" if sar["time"] == fastest else "")
                if sar["time"] is not None
                else "N/A"
            ),
        },
    ]
    st.subheader("Comparison on this sample")
    st.table(rows)
    st.caption(
        "★ marks the better score on this sample. It does not mean either model is universally better."
    )


initialise_state()
health = model_health()

with st.sidebar:
    st.title("SAR Optical")
    st.caption("Independent reconstruction workflows")
    page = st.radio(
        "Navigation",
        ["Pix2Pix", "Detailed Structure", "Model Comparison"],
        label_visibility="collapsed",
    )
    st.divider()
    for name, details in health.get("models", {}).items():
        label = name.replace("_", " ").title()
        if details.get("available"):
            st.success("{} ready".format(label))
        else:
            st.error("{} unavailable".format(label))
    if health.get("status") == "offline":
        st.warning("Backend offline")

if page == "Pix2Pix":
    pix2pix_page()
elif page == "Detailed Structure":
    sarfusionformer_page()
else:
    comparison_page()
