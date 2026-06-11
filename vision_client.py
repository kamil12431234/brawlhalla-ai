"""Vision inference for brawlhalla-vision using YOLO on GPU (RTX 3090)."""

import os
import logging
import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import LOCAL_MODEL_ONNX_PATH, CONFIDENCE_THRESHOLD, DEVICE

logger = logging.getLogger(__name__)

# Global model state
_yolo_model = None
_model_failed: bool = False
_class_labels: list[str] = ["enemy", "gadget", "player", "weapon"]
_load_lock = threading.Lock()
_warmup_done: bool = False
_inference_count: int = 0
_last_error_time: float = 0.0
_error_count: int = 0
_adaptive_confidence: float = CONFIDENCE_THRESHOLD

# Performance tracking
_frame_times: deque[float] = deque(maxlen=60)
_inference_times: deque[float] = deque(maxlen=60)

# GPU settings
_GPU_DEVICE = 0 if DEVICE == "cuda" else "cpu"
_half_supported = True   # will be set False if half-precision fails

@dataclass
class TemporalTrack:
    """Tracks a single object across frames with temporal smoothing."""
    class_id: int
    label: str
    x: float
    y: float
    w: float
    h: float
    conf: float
    track_id: int
    age: int = 0
    total_conf: float = 0.0
    avg_conf: float = 0.0
    
    def update(self, det: dict):
        self.x = det.get("x", self.x)
        self.y = det.get("y", self.y)
        self.w = det.get("w", det.get("width", self.w))
        self.h = det.get("h", det.get("height", self.h))
        self.conf = det.get("conf", self.conf)
        self.age += 1
        self.total_conf += self.conf
        self.avg_conf = self.total_conf / self.age
    
    @property
    def smoothed_conf(self) -> float:
        age_factor = min(1.0, self.age / 10.0)
        return self.avg_conf * 0.7 + self.conf * 0.3 * age_factor
    
    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)


class TemporalSmoother:
    def __init__(self, max_tracks: int = 20, iou_threshold: float = 0.3):
        self._tracks: list[TemporalTrack] = []
        self._max_tracks = max_tracks
        self._iou_threshold = iou_threshold
        self._next_id: int = 0
        self._frame_count: int = 0
    
    def _iou(self, a: dict, b: dict) -> float:
        ax = a.get("x", 0)
        ay = a.get("y", 0)
        aw = a.get("w", a.get("width", 0))
        ah = a.get("h", a.get("height", 0))
        bx = b.get("x", 0)
        by = b.get("y", 0)
        bw = b.get("w", b.get("width", 0))
        bh = b.get("h", b.get("height", 0))
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
    
    def update(self, detections: list[dict]) -> list[dict]:
        self._frame_count += 1
        self._tracks = [t for t in self._tracks if t.age < 30]
        
        for det in detections:
            best_match = None
            best_iou = 0.0
            for track in self._tracks:
                if track.label == det.get("class_name", ""):
                    iou = self._iou(track.__dict__, det)
                    if iou > best_iou and iou > self._iou_threshold:
                        best_match = track
                        best_iou = iou
            
            if best_match:
                best_match.update(det)
            else:
                self._tracks.append(TemporalTrack(
                    class_id=det.get("class_id", 0),
                    label=det.get("class_name", "unknown"),
                    x=det.get("x", 0),
                    y=det.get("y", 0),
                    w=det.get("w", 0),
                    h=det.get("h", 0),
                    conf=det.get("conf", 0),
                    track_id=self._next_id,
                ))
                self._next_id += 1
        
        smoothed = []
        for track in self._tracks:
            smoothed.append({
                "x": track.x,
                "y": track.y,
                "w": track.w,
                "h": track.h,
                "conf": track.smoothed_conf,
                "class_id": track.class_id,
                "class_name": track.label,
                "track_id": track.track_id,
            })
        return smoothed


class AdaptiveConfidenceCalibrator:
    def __init__(self, initial_conf: float = 0.30, window_size: int = 60):
        self._conf = initial_conf
        self._window_size = window_size
        self._conf_history: deque[float] = deque(maxlen=window_size)
        self._detection_count_history: deque[int] = deque(maxlen=window_size)
        self._last_conf_adjust = time.time()
    
    def update(self, num_detections: int, avg_confidence: float):
        self._conf_history.append(avg_confidence)
        self._detection_count_history.append(num_detections)
        if time.time() - self._last_conf_adjust < 1.0:
            return
        self._last_conf_adjust = time.time()
        recent_dets = sum(self._detection_count_history) / len(self._detection_count_history) if self._detection_count_history else 0
        recent_avg_conf = sum(self._conf_history) / len(self._conf_history) if self._conf_history else 0.5
        if recent_dets > 5 and recent_avg_conf < 0.4:
            self._conf = max(0.20, self._conf - 0.02)
        elif recent_dets < 1 and recent_avg_conf > 0.7:
            self._conf = min(CONFIDENCE_THRESHOLD, self._conf + 0.01)
    
    @property
    def confidence(self) -> float:
        return self._conf


_temporal_smoother = TemporalSmoother()
_confidence_calibrator = AdaptiveConfidenceCalibrator(CONFIDENCE_THRESHOLD)


def load_model(preload: bool = True, warmup: bool = True):
    """Load YOLO model using ultralytics with CUDA support."""
    global _yolo_model, _model_failed, _warmup_done
    
    with _load_lock:
        if _yolo_model is not None:
            return
        
        try:
            from ultralytics import YOLO
            
            model_path = None
            
            # Prefer PyTorch for better CUDA support
            pt_path = LOCAL_MODEL_ONNX_PATH.replace(".onnx", ".pt")
            if os.path.exists(pt_path):
                model_path = pt_path
                logger.info(f"[VISION] Using PyTorch model: {model_path}")
            else:
                default_pt = "./model/brawlhalla_vision.pt"
                if os.path.exists(default_pt):
                    model_path = default_pt
                    logger.info(f"[VISION] Using default PyTorch model: {model_path}")
            
            if model_path is None and os.path.exists(LOCAL_MODEL_ONNX_PATH):
                model_path = LOCAL_MODEL_ONNX_PATH
                logger.info(f"[VISION] Using ONNX model: {model_path}")
            
            if model_path is None:
                raise FileNotFoundError(f"No model at {LOCAL_MODEL_ONNX_PATH}")
            
            _yolo_model = YOLO(model_path)
            
            if warmup:
                _do_warmup()
            
            _model_failed = False
            _warmup_done = True
            device_info = "cuda:0" if DEVICE == "cuda" else "cpu"
            logger.info(f"[VISION] Model loaded (device: {device_info})")
            
        except Exception as e:
            logger.error(f"[VISION] Failed to load model: {e}")
            _model_failed = True


def _do_warmup():
    global _warmup_done
    if _warmup_done:
        return
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _yolo_model.predict(dummy, verbose=False, device=_GPU_DEVICE)
        _warmup_done = True
        logger.info("[VISION] Warmup complete")
    except Exception as e:
        logger.warning("[VISION] Warmup failed: %s", e)


_conf_history: deque[float] = deque(maxlen=60)


def infer(frame_bgr: np.ndarray, conf: Optional[float] = None) -> list[dict]:
    """Run inference on a frame with temporal smoothing."""
    global _inference_count, _model_failed, _adaptive_confidence, _last_error_time, _error_count, _half_supported
    
    start_time = time.perf_counter()
    
    if conf is None:
        conf = _adaptive_confidence
    
    if _model_failed:
        return []
    
    try:
        if _yolo_model is None:
            load_model(preload=True, warmup=True)
        
        if _yolo_model is None:
            return []
        
        detections = []
        
        # Use 640 for better detection accuracy (enemy detection requires higher resolution)
        imgsz = 640
        
        # Half precision: try based on device, but fall back if it fails
        half = (_GPU_DEVICE == 0 and _half_supported)
        
        results = _yolo_model.predict(
            frame_bgr,
            conf=conf,
            verbose=False,
            device=_GPU_DEVICE,
            imgsz=imgsz,
            half=half,
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()
                img_h, img_w = frame_bgr.shape[:2]
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        "x": float(x1) / img_w,
                        "y": float(y1) / img_h,
                        "w": float(x2 - x1) / img_w,
                        "h": float(y2 - y1) / img_h,
                        "conf": confidence,
                        "class_id": cls_id,
                        "class_name": _class_labels[cls_id] if cls_id < len(_class_labels) else "unknown",
                    })
        
        _inference_count += 1
        if detections:
            avg_conf = sum(d["conf"] for d in detections) / len(detections)
            _conf_history.append(avg_conf)
            _confidence_calibrator.update(len(detections), avg_conf)
        
        if _inference_count > 1:
            detections = _temporal_smoother.update(detections)
        
        inference_time = time.perf_counter() - start_time
        _inference_times.append(inference_time)
        _frame_times.append(inference_time)
        
        return detections
    
    except Exception as e:
        _error_count += 1
        _last_error_time = time.time()
        
        # If many early errors, likely due to half-precision; disable it.
        if _half_supported and _inference_count < 30 and _error_count >= 5:
            logger.warning("[VISION] Disabling FP16 after repeated errors")
            _half_supported = False
        
        # Escalate logging when errors keep happening so the system isn't silent.
        if _error_count < 5 or (time.time() - _last_error_time) > 10:
            logger.warning("[VISION] Inference error #%d: %s", _error_count, e)
        
        # If repeated failures become severe, log as ERROR and mark model failed.
        if _error_count >= 50:
            logger.error(
                "[VISION] Too many consecutive errors (%d); marking model FAILED to stop silent degradation.",
                _error_count,
            )
            _model_failed = True
        
        return []


def predict(frame_bgr: np.ndarray, conf: Optional[float] = None) -> list[dict]:
    """Alias for infer() for compatibility."""
    return infer(frame_bgr, conf)


def reset():
    global _yolo_model, _warmup_done, _inference_count, _adaptive_confidence, _error_count
    _yolo_model = None
    _warmup_done = False
    _inference_count = 0
    _adaptive_confidence = CONFIDENCE_THRESHOLD
    _error_count = 0
    _temporal_smoother._tracks.clear()
    _confidence_calibrator._conf = CONFIDENCE_THRESHOLD


def get_stats() -> dict:
    avg_inf = sum(_inference_times) / len(_inference_times) if _inference_times else 0
    return {
        "inferences": _inference_count,
        "avg_inference_ms": avg_inf * 1000,
        "adaptive_confidence": _adaptive_confidence,
        "errors": _error_count,
        "fps": 1.0 / avg_inf if avg_inf > 0 else 0,
    }


def set_adaptive_confidence(conf: float):
    global _adaptive_confidence
    _adaptive_confidence = max(0.1, min(0.9, conf))