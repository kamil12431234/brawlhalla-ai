#!/usr/bin/env python3
"""
Unified YOLO training script for Brawlhalla vision model.
Supports multiple model sizes and training configurations.
"""

import os
import sys
import shutil
import logging

from ultralytics import YOLO

from config import get_training_parser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DATA_YAML = "model/yolov8_brawlhalla_vision/data.yaml"
OUTPUT_MODEL = "model/brawlhalla_vision.pt"


def train(args):
    """Train YOLO model with given arguments."""
    logger.info(f"Starting training: model={args.model}, epochs={args.epochs}, batch={args.batch}")

    # Verify data.yaml exists
    if not os.path.exists(DATA_YAML):
        logger.error(f"Data config not found: {DATA_YAML}")
        logger.error("Run download_model.py first to prepare dataset")
        return 1

    model = YOLO(args.model)

    results = model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        half=args.half,
        amp=True,
        workers=args.workers,
        patience=args.patience,
        cos_lr=args.cos_lr if hasattr(args, 'cos_lr') else False,
        close_mosaic=args.close_mosaic if hasattr(args, 'close_mosaic') else 10,
    )

    # Save best model
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, OUTPUT_MODEL)
        logger.info(f"Best model saved to: {OUTPUT_MODEL}")
    else:
        logger.warning("Best model not found at expected path")

    # Validate
    logger.info("Running validation...")
    val = model.val(data=DATA_YAML, device=args.device, half=args.half)
    logger.info(f"mAP50: {val.box.map50:.4f}  mAP50-95: {val.box.map:.4f}")

    # Export to ONNX
    if args.half:
        logger.info("Exporting to ONNX (FP16)...")
        onnx_path = model.export(format="onnx", half=True, simplify=True)
        logger.info(f"ONNX model: {onnx_path}")

    return 0


def main():
    parser = get_training_parser()
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())