__all__ = [
    "FrameData",
    "PerformanceStats",
    "StallRecovery",
    "AsyncPipeline",
    "AIRunner",
    "run_ai_loop",
    "stop_ai_loop",
    "get_ai_stats",
]
"""Shared AI runner loop with frame budget management, stall recovery, and adaptive performance optimization."""

import time
import logging
import threading
import queue
from typing import Optional, Callable
from dataclasses import dataclass, field
from collections import deque

from ai_brain import BrawlAI
from input_controller import InputController, release_all
import vision_client
from game_state import build_world_state, reset_tracking, get_tracker_stats, update_player_input
from screen_capture import capture_frame
from config import AUTO_DETECT_WINDOW, _detect_game_window, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

StatsCallback = Callable[[float, float, int, bool, dict], None]
LogCallback = Callable[[str], None]

# Performance settings
TARGET_FPS = 45  # Realistic target for vision AI with deep learning inference
MIN_FRAME_TIME = 1.0 / TARGET_FPS
MAX_FRAME_SKIP = 3
FPS_ALERT_THRESHOLD = 30  # Only alert if FPS drops below 30 (35-40 is acceptable)

# ── Frame budget management ──
_FRAME_BUDGET_MS = 22.22  # 45 FPS = 22.22ms per frame
_CAPTURE_BUDGET_MS = 4.0
_INFERENCE_BUDGET_MS = 8.0
_DECISION_BUDGET_MS = 3.0
_INPUT_BUDGET_MS = 1.0


@dataclass
class FrameData:
    """Container for frame processing data."""
    frame_idx: int
    timestamp: float
    capture_time: float = 0.0
    inference_time: float = 0.0
    decision_time: float = 0.0
    total_time: float = 0.0
    frame: object = None
    detections: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    action: Optional[str] = None
    fps: float = 0.0
    within_budget: bool = True


@dataclass 
class PerformanceStats:
    """Aggregated performance statistics."""
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    avg_capture_ms: float = 0.0
    avg_inference_ms: float = 0.0
    avg_decision_ms: float = 0.0
    frame_drops: int = 0
    total_frames: int = 0
    action_distribution: dict = field(default_factory=dict)
    budget_violations: int = 0
    stall_recoveries: int = 0
    adaptive_quality: float = 1.0  # 0-1, drops quality when struggling


@dataclass
class StallRecovery:
    """Stall detection and recovery state."""
    detected: bool = False
    start_frame: int = 0
    recovery_attempts: int = 0
    last_stall_time: float = 0.0


class AsyncPipeline:
    """Async pipeline for frame processing with lookahead buffer."""
    
    def __init__(self, buffer_size: int = 3):
        self._buffer_size = buffer_size
        self._capture_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._inference_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._decision_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        
        self._capture_thread: threading.Thread = None
        self._inference_thread: threading.Thread = None
        self._decision_thread: threading.Thread = None
    
    def start(self, get_frame_callback: callable):
        self._running = True
        self._get_frame = get_frame_callback
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        
        self._capture_thread.start()
        self._inference_thread.start()
        self._decision_thread.start()
    def _capture_loop(self):
        # Track dropped captures due to full queue (log periodically).
        dropped = 0
        while self._running:
            try:
                frame_idx = next(self._frame_counter)
                frame = self._get_frame()
                fd = FrameData(frame_idx=frame_idx, timestamp=time.time(), frame=frame)
                self._capture_queue.put(fd, timeout=0.1)
            except queue.Full:
                dropped += 1
                # Log every 50 drops to avoid spamming logs.
                if dropped % 50 == 0:
                    logger.debug("[PIPELINE] Dropped %d frames due to full capture queue", dropped)
            except Exception as e:
                logger.warning(f"[PIPELINE] Capture error: {e}")

    def _inference_loop(self):
        while self._running:
            try:
                fd = self._capture_queue.get(timeout=0.1)
                fd.capture_time = time.time() - fd.timestamp

                if fd.frame is not None:
                    start_infer = time.perf_counter()
                    try:
                        fd.detections = vision_client.infer(fd.frame, conf=CONFIDENCE_THRESHOLD)
                    except Exception as e:
                        logger.warning("[PIPELINE] Inference error: %s", e)
                        fd.detections = []
                    end_infer = time.perf_counter()
                    fd.inference_time = (end_infer - start_infer)

                try:
                    self._inference_queue.put(fd, timeout=0.1)
                except queue.Full:
                    # Drop frame if pipeline is congested.
                    pass
            except queue.Empty:
                pass
            except Exception as e:
                logger.warning(f"[PIPELINE] Inference error: {e}")
    
    def _decision_loop(self):
        # NOTE: This is a relay loop, not the AI decision engine.
        # Actual game decisions are performed on the main thread in run_loop (ai.choose_action).
        # This thread's job is to drain inference results and track timing only.
        while self._running:
            try:
                fd = self._inference_queue.get(timeout=0.1)
                fd.decision_time = time.time() - fd.timestamp - fd.capture_time - fd.inference_time
                self._result_queue.put(fd, timeout=0.1)
            except queue.Empty:
                pass
            except Exception as e:
                logger.warning(f"[PIPELINE] Relay/decision error: {e}")
    
    def get_result(self, timeout: float = 0.05) -> Optional[FrameData]:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        self._running = False


class AIRunner:
    def __init__(
        self,
        mode: str = "full",
        aggressive: float = 0.5,
        dry_run: bool = False,
        async_mode: bool = True,
    ):
        self.mode = mode
        self.aggressive = aggressive
        self.dry_run = dry_run
        self.async_mode = async_mode
        self.running = False
        self._loop_count = 0
        self._last_stats_time = time.time()
        self._frame_times: list[float] = []
        
        # Performance tracking
        self._perf_stats = PerformanceStats()
        self._fps_history: deque[float] = deque(maxlen=60)
        self._capture_times: deque[float] = deque(maxlen=60)
        self._inference_times: deque[float] = deque(maxlen=60)
        self._decision_times: deque[float] = deque(maxlen=60)
        self._action_counts: dict = {}
        
        # Frame budget tracking
        self._budget_violations: deque[int] = deque(maxlen=30)
        self._adaptive_quality: float = 1.0
        
        # Stall recovery (with hysteresis)
        self._stall = StallRecovery()
        self._stall_threshold_frames: int = 8  # consecutive frames with no new detections = stall
        self._stall_cooldown_ms: int = 750      # ms between allowed recoveries
        
        # Async pipeline (optional)
        self._pipeline: Optional[AsyncPipeline] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        
        # Frame skip detection
        self._skip_count: int = 0
        self._last_good_frame_time: float = 0.0
        self._no_detection_frames: int = 0   # consecutive frames without detections (for hysteresis)
    
    def _check_stall(self, current_time: float) -> bool:
        """Detect if we're in a stall (no progress on enemy tracking)."""
        # Cooldown between stalls is in ms → convert to seconds for comparison.
        cooldown_s = self._stall_cooldown_ms / 1000.0
        if current_time - self._stall.last_stall_time < cooldown_s:
            return False

        # Use consecutive-no-detection frames (hysteresis) + time-based check to avoid thrashing.
        if (self._no_detection_frames >= self._stall_threshold_frames and
                (current_time - self._last_good_frame_time) > 1.0):
            self._stall.detected = True
            self._stall.recovery_attempts += 1
            self._stall.start_frame = self._loop_count
            return True

        return False
    
    def _recover_stall(self, ai: BrawlAI, input_ctrl: InputController) -> str:
        """Attempt to recover from a stall condition."""
        self._perf_stats.stall_recoveries += 1

        # Release all keys first
        release_all()

        # Force idle frame to reset state
        logger.info("[STALL] Recovery attempt #%d", self._stall.recovery_attempts)

        # Reset tracking
        reset_tracking()

        self._stall.last_stall_time = time.time()
        self._stall.detected = False

        return "stall_recovery"

    def _stale_or_skip_on_capture_error(self, frame_time: float):
        """Handle stall tracking on capture errors without false positives."""
        # Mark that this frame had no detections due to error; increase hysteresis counter.
        if self._loop_count > 5:
            self._no_detection_frames += 1
            # Keep last_good_frame_time somewhat recent so one-time glitches don't cause stall.
            self._last_good_frame_time = max(
                self._last_good_frame_time,
                frame_time - 0.25,
            )
    
    def _adaptive_performance(self, total_time_ms: float):
        """Adapt performance based on frame budget compliance."""
        budget = _FRAME_BUDGET_MS
        
        if total_time_ms > budget:
            self._budget_violations.append(1)
            self._adaptive_quality = max(0.5, self._adaptive_quality - 0.05)
        else:
            self._budget_violations.append(0)
            self._adaptive_quality = min(1.0, self._adaptive_quality + 0.01)
        
        # Check if we're consistently over budget — log once, not per-frame
        violation_ratio = sum(self._budget_violations) / len(self._budget_violations) if self._budget_violations else 0
        if violation_ratio > 0.5:
            # Reduce quality: lower confidence threshold
            vision_client.set_adaptive_confidence(CONFIDENCE_THRESHOLD * max(0.3, self._adaptive_quality))
        else:
            vision_client.set_adaptive_confidence(CONFIDENCE_THRESHOLD)
    
    def run_loop(
        self,
        stats_cb: Optional[StatsCallback] = None,
        log_cb: Optional[LogCallback] = None,
    ):
        if self.running:
            return
        self.running = True
        
        def _log(msg: str):
            logger.info(msg)
            if log_cb:
                log_cb(msg)
        
        # Reset tracking state
        reset_tracking()
        
        # Auto-detect game window if enabled
        if AUTO_DETECT_WINDOW:
            _detect_game_window()
        
        ai = BrawlAI(mode=self.mode, aggressive=self.aggressive)
        input_ctrl = InputController(dry_run=self.dry_run)
        
        fps_start = time.time()
        frame_idx = 0
        loop_start = time.time()
        
        _log(f"[START] Brawlhalla AI | mode={self.mode} dry_run={self.dry_run} aggression={self.aggressive}")
        _log(f"[INFO] Target FPS: {TARGET_FPS} | Async mode: {self.async_mode} | Budget: {_FRAME_BUDGET_MS:.1f}ms/frame")
        
        try:
            while self.running:
                frame_start = time.time()
                current_time = time.time()
                
                # ── STALL CHECK ──
                if self._check_stall(current_time):
                    self._recover_stall(ai, input_ctrl)
                    # Continue processing after recovery; no need to consume a separate var.

                # ── CAPTURE ──
                capture_start = time.time()
                frame = None
                try:
                    frame = capture_frame()
                except Exception as e:
                    logger.warning("[CAPTURE] Error (frame %d): %s", frame_idx, e)
                    # On capture failure, update stall guard to avoid false positives.
                    self._stale_or_skip_on_capture_error(frame_time=current_time - fps_start)
                    continue
                capture_time = (time.time() - capture_start) * 1000
                self._capture_times.append(capture_time)
                
                if capture_time > _CAPTURE_BUDGET_MS:
                    logger.debug(f"[PERF] Capture over budget: {capture_time:.1f}ms > {_CAPTURE_BUDGET_MS}ms")
                
                # ── INFERENCE ──
                if frame is None:
                    continue

                inference_start = time.time()
                img_h, img_w = frame.shape[:2]

                # Adaptive quality tuning for confidence threshold.
                conf_thresh = CONFIDENCE_THRESHOLD * max(0.3, self._adaptive_quality)
                detections = []
                try:
                    detections = vision_client.infer(frame, conf=conf_thresh)
                except Exception as e:
                    logger.error("[INFERENCE] Error (frame %d): %s", frame_idx, e)

                inference_time = (time.time() - inference_start) * 1000
                self._inference_times.append(inference_time)
                
                if inference_time > _INFERENCE_BUDGET_MS:
                    logger.debug(f"[PERF] Inference over budget: {inference_time:.1f}ms > {_INFERENCE_BUDGET_MS}ms")
                
                # Track good frame; use for stall hysteresis
                if detections:
                    self._last_good_frame_time = current_time
                    self._no_detection_frames = 0
                else:
                    self._no_detection_frames += 1
                
                # ── BUILD WORLD STATE ──
                state = build_world_state(detections, float(img_w), float(img_h))
                
                # ── DECISION ──
                decision_start = time.time()
                action = ai.choose_action(state)
                decision_time = (time.time() - decision_start) * 1000
                self._decision_times.append(decision_time)
                
                if decision_time > _DECISION_BUDGET_MS:
                    logger.debug(f"[PERF] Decision over budget: {decision_time:.1f}ms > {_DECISION_BUDGET_MS}ms")
                
                # ── APPLY INPUT ──
                input_start = time.time()
                if not self.dry_run and action:
                    keys = ai.resolve_action(action, state)
                    if keys:
                        input_ctrl.apply_keys(keys)
                        update_player_input(keys)
                        
                        for key_name, _ in keys:
                            self._action_counts[key_name] = self._action_counts.get(key_name, 0) + 1
                input_time = (time.time() - input_start) * 1000
                
                if input_time > _INPUT_BUDGET_MS:
                    logger.debug(f"[PERF] Input over budget: {input_time:.1f}ms > {_INPUT_BUDGET_MS}ms")
                
                # ── FRAME TIMING ──
                frame_time = time.time() - frame_start
                self._frame_times.append(frame_time)
                
                # Calculate FPS
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self._fps_history.append(current_fps)
                
                # Frame budget check
                total_time_ms = frame_time * 1000
                within_budget = total_time_ms <= _FRAME_BUDGET_MS
                
                # Adaptive performance
                self._adaptive_performance(total_time_ms)
                
                # Frame skipping if behind (but not too many)
                if frame_time < MIN_FRAME_TIME:
                    time.sleep(MIN_FRAME_TIME - frame_time)
                
                # Skip only if significantly behind (not just slightly slow) — avoid freezing during fights.
                skip_threshold = MIN_FRAME_TIME * 3.0
                if self._skip_count < MAX_FRAME_SKIP and frame_time > skip_threshold:
                    self._skip_count += 1
                    frame_idx += 1
                    self._loop_count += 1
                    continue
                
                self._skip_count = 0
                frame_idx += 1
                self._loop_count += 1
                
                # ── STATISTICS UPDATE (every ~1 second) ──
                if time.time() - self._last_stats_time >= 1.0:
                    avg_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
                    min_fps = min(self._fps_history) if self._fps_history else 0
                    max_fps = max(self._fps_history) if self._fps_history else 0
                    avg_capture = sum(self._capture_times) / len(self._capture_times) if self._capture_times else 0
                    avg_inference = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0
                    avg_decision = sum(self._decision_times) / len(self._decision_times) if self._decision_times else 0

                    # Update performance stats (safe even when deques empty).
                    self._perf_stats.avg_fps = avg_fps
                    self._perf_stats.min_fps = min_fps
                    self._perf_stats.max_fps = max_fps
                    self._perf_stats.avg_capture_ms = avg_capture
                    self._perf_stats.avg_inference_ms = avg_inference
                    self._perf_stats.avg_decision_ms = avg_decision
                    self._perf_stats.total_frames = frame_idx
                    self._perf_stats.action_distribution = self._action_counts.copy()
                    self._perf_stats.budget_violations = sum(self._budget_violations) if self._budget_violations else 0
                    self._perf_stats.adaptive_quality = self._adaptive_quality

                    # FPS alert (rate-limited to avoid spamming).
                    if avg_fps < FPS_ALERT_THRESHOLD:
                        logger.warning("[PERF] Low FPS: %.1f (target: %d)", avg_fps, TARGET_FPS)

                    player_ok = bool(state.get("player"))
                    enemies_count = len(state.get("enemies", []))

                    # Build extended stats.
                    ext_stats = {
                        **vision_client.get_stats(),
                        **get_tracker_stats(),
                        **ai.get_stats(),
                        "avg_capture_ms": avg_capture,
                        "avg_decision_ms": avg_decision,
                        "perf": self._perf_stats.__dict__,
                        "adaptive_quality": self._adaptive_quality,
                        "budget_violations": sum(self._budget_violations) if self._budget_violations else 0,
                        "stall_recoveries": self._stall.recovery_attempts,
                    }

                    if stats_cb:
                        try:
                            stats_cb(
                                avg_fps,
                                avg_capture + avg_inference + avg_decision,
                                enemies_count,
                                player_ok,
                                ext_stats,
                            )
                        except Exception as cb_err:
                            logger.debug("[STATS] Callback error: %s", cb_err)

                    self._last_stats_time = time.time()
                    self._fps_history.clear()
                    self._capture_times.clear()
                    self._inference_times.clear()
                    self._decision_times.clear()
        
        except KeyboardInterrupt:
            _log("[STOP] Interrupted by user.")
        except Exception as e:
            logger.error("AI loop crashed: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            release_all()
            _log("[STOP] Brawlhalla AI loop stopped.")
    
    def stop(self):
        self.running = False
        if self._pipeline:
            self._pipeline.stop()
    
    def get_loop_stats(self) -> dict:
        return {
            "loop_count": self._loop_count,
            "running": self.running,
            "adaptive_quality": self._adaptive_quality,
            "stall_recoveries": self._stall.recovery_attempts,
            "perf": self._perf_stats.__dict__,
        }
    
    def reset_stats(self):
        self._perf_stats = PerformanceStats()
        self._fps_history.clear()
        self._capture_times.clear()
        self._inference_times.clear()
        self._decision_times.clear()
        self._action_counts.clear()
        self._budget_violations.clear()


# ── Standalone functions for compatibility ─────────────────────────

_runner_instance: Optional[AIRunner] = None


def run_ai_loop(mode: str = "full", aggressive: float = 0.5, dry_run: bool = False,
                stats_cb: Optional[StatsCallback] = None, log_cb: Optional[LogCallback] = None):
    global _runner_instance
    _runner_instance = AIRunner(mode=mode, aggressive=aggressive, dry_run=dry_run)
    _runner_instance.run_loop(stats_cb=stats_cb, log_cb=log_cb)
    return _runner_instance


def stop_ai_loop():
    global _runner_instance
    if _runner_instance:
        _runner_instance.stop()


def get_ai_stats() -> dict:
    global _runner_instance
    if _runner_instance:
        return _runner_instance.get_loop_stats()
    return {}