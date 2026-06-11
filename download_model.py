#!/usr/bin/env python3
"""
Downloads and exports brawlhalla-vision (version 5) as a local ONNX model.

Usage:
    Set your Roboflow API key via ROBOFLOW_API_KEY env var.
    Run: python download_model.py
"""

import os
import shutil
import logging
from roboflow import Roboflow
from config import ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT_NAME, ROBOFLOW_VERSION

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_model_dir():
    os.makedirs("./model", exist_ok=True)


def main():
    api_key = ROBOFLOW_API_KEY or os.getenv("RF_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. "
            "Set ROBOFLOW_API_KEY or RF_API_KEY environment variable."
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT_NAME)
    version = project.version(ROBOFLOW_VERSION)

    ensure_model_dir()

    logger.info("Exporting brawlhalla-vision model as ONNX (local, CPU-friendly)...")
    model = version.download(
        model_format="yolov8",
        location="./model/yolov8_brawlhalla_vision",
    )

    logger.info("Model downloaded. Exporting to ONNX from local weights...")

    try:
        import importlib.util
        has_ultralytics = importlib.util.find_spec("ultralytics") is not None
    except Exception:
        has_ultralytics = False

    onnx_path = "./model/brawlhalla_vision.onnx"

    if has_ultralytics:
        from ultralytics import YOLO

        weights_dir = model.model_path
        yolo_model = YOLO(os.path.join(weights_dir, "weights", "best.pt"))
        logger.info("Running ONNX export (CPU)...")
        yolo_model.export(format="onnx", dynamic=False, half=False)
        exported = os.path.join(weights_dir, "weights", "best.onnx")

        if os.path.exists(exported):
            shutil.copy2(exported, onnx_path)
            logger.info(f"ONNX model saved to {onnx_path}")
        else:
            raise RuntimeError("ONNX export did not produce expected file.")
    else:
        logger.warning(
            "ultralytics not installed. "
            "Install with: pip install ultralytics, then re-run this script."
        )


if __name__ == "__main__":
    main()
