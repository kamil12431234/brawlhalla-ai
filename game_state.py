__all__ = [
    "KalmanFilter1D",
    "AdaptiveKalmanFilter1D",
    "KalmanTracker2D",
    "TrackID",
    "WorldStateHistory",
    "BlastZoneInfo",
    "GameState",
    "reset_tracking",
    "update_player_input",
    "build_world_state",
    "get_tracker_stats",
    "is_player_tracked",
    "is_near_edge",
    "is_in_danger",
    "get_recovery_path",
    "get_recovery_urgency",
    "record_attack_result",
    "get_attack_hit_rate",
    "predict_object",
    "predict_nearest_enemy",
]
"""Game state builder with advanced tracking, Kalman filtering, blast zone awareness, and prediction quality scoring."""

import logging
from typing import Optional, TypedDict
from dataclasses import dataclass
from collections import deque

from config import (
    CONFIDENCE_THRESHOLD,
    STAGE_L as _STAGE_EDGE,        # align with ai_brain via shared config
    STAGE_BOT,                      # blast zone top boundary
    STAGE_TOP as _STAGE_TOP,        # top boundary reference
)

logger = logging.getLogger(__name__)

# Tracking constants (derived from shared stage geometry where applicable)
_STAGE_BOT = STAGE_BOT              # bottom blast-zone boundary
_MOVE_STEP = 0.012
_JUMP_STEP = 0.025
_DOWN_STEP = 0.010

_CHARACTER_CLASSES = {"player", "enemy"}
_ENEMY_MATCH_DIST = 0.12
_ENEMY_LIFETIME_LOSS = 5
_GADGET_CLASSES = {"gadget", "weapon"}

# Kalman filter parameters (adaptive)
_DEFAULT_Q = 0.001  # Process noise
_DEFAULT_R = 0.1    # Measurement noise

# Blast zone thresholds
_BLAST_ZONE_HORIZ = 0.02
_BLAST_ZONE_VERT = 0.05
_BLAST_ZONE_DANGER_MARGIN = 0.05


class KalmanFilter1D:
    """1D Kalman filter for smooth position tracking."""
    
    def __init__(self, q: float = _DEFAULT_Q, r: float = _DEFAULT_R):
        self._q = q
        self._r = r
        self._x = 0.0
        self._p = 1.0
        self._initialized = False
    
    def update(self, measurement: float) -> float:
        if not self._initialized:
            self._x = measurement
            self._initialized = True
            return self._x
        
        self._p += self._q
        k = self._p / (self._p + self._r)
        self._x += k * (measurement - self._x)
        self._p *= (1 - k)
        
        return self._x
    
    def get(self) -> float:
        return self._x
    
    def reset(self):
        self._x = 0.0
        self._p = 1.0
        self._initialized = False


class AdaptiveKalmanFilter1D(KalmanFilter1D):
    """Kalman filter that adapts noise parameters based on tracking quality."""
    
    def __init__(self, q: float = _DEFAULT_Q, r: float = _DEFAULT_R):
        super().__init__(q, r)
        self._q_base = q
        self._r_base = r
        self._tracking_quality: float = 1.0
        self._consecutive_updates: int = 0
    
    def update(self, measurement: float) -> float:
        self._consecutive_updates += 1
        
        if self._initialized:
            innovation = abs(measurement - self._x)
            
            if innovation > 0.05:
                self._q = min(0.1, self._q * 1.5)
            elif innovation < 0.01:
                self._q = max(self._q_base * 0.1, self._q * 0.9)
            
            # Update tracking quality
            self._tracking_quality = max(0.1, min(1.0, 1.0 - innovation * 5))
        
        return super().update(measurement)
    
    def get_tracking_quality(self) -> float:
        return self._tracking_quality
    
    def reset(self):
        super().reset()
        self._q = self._q_base
        self._r = self._r_base
        self._consecutive_updates = 0


class KalmanTracker2D:
    """2D Kalman filter with velocity estimation and prediction quality scoring."""
    
    def __init__(self, q: float = _DEFAULT_Q, r: float = _DEFAULT_R):
        self._x_filter = AdaptiveKalmanFilter1D(q, r)
        self._y_filter = AdaptiveKalmanFilter1D(q, r)
        self._vx = 0.0
        self._vy = 0.0
        self._prediction_quality: float = 1.0  # 0-1 confidence in predictions
        self._acceleration_x: float = 0.0
        self._acceleration_y: float = 0.0
        self._frame_count: int = 0
    
    def update(self, x: float, y: float) -> tuple[float, float]:
        """Update with measurement, return smoothed position."""
        smoothed_x = self._x_filter.update(x)
        smoothed_y = self._y_filter.update(y)
        
        prev_x = self._x_filter.get()
        prev_y = self._y_filter.get()
        
        # Velocity estimation with smoothing
        if hasattr(self, '_last_x'):
            dt = 1.0 / 60.0  # Assume 60 FPS
            raw_vx = (smoothed_x - self._last_x) / dt
            raw_vy = (smoothed_y - self._last_y) / dt
            
            # EMA smoothing for velocity
            self._vx = self._vx * 0.7 + raw_vx * 0.3
            self._vy = self._vy * 0.7 + raw_vy * 0.3
            
            # Acceleration estimation
            self._acceleration_x = raw_vx - self._vx
            self._acceleration_y = raw_vy - self._vy
        
        self._last_x = smoothed_x
        self._last_y = smoothed_y
        self._frame_count += 1
        
        # Update prediction quality
        x_quality = self._x_filter.get_tracking_quality()
        y_quality = self._y_filter.get_tracking_quality()
        self._prediction_quality = (x_quality + y_quality) / 2.0
        
        return (smoothed_x, smoothed_y)
    
    def predict(self, steps: int = 1) -> tuple[float, float]:
        """Predict position N steps ahead with velocity + acceleration."""
        base_x = self._x_filter.get()
        base_y = self._y_filter.get()
        
        # Simple: linear prediction
        pred_x = base_x + self._vx * steps
        pred_y = base_y + self._vy * steps
        
        # Blend with Kalman prediction for smoother output
        kalman_pred_x = self._x_filter.get() + self._vx * steps
        kalman_pred_y = self._y_filter.get() + self._vy * steps
        
        # Weight by prediction quality
        w = self._prediction_quality
        return (
            kalman_pred_x * w + pred_x * (1 - w),
            kalman_pred_y * w + pred_y * (1 - w),
        )
    
    def predict_safe(self, steps: int = 1, confidence_threshold: float = 0.3) -> tuple[float, float, float]:
        """
        Predict position with uncertainty bounds.
        
        Returns:
            (predicted_x, predicted_y, uncertainty)
        """
        px, py = self.predict(steps)
        
        # Uncertainty grows with steps and drops with quality
        uncertainty = steps * (1.0 - self._prediction_quality) * 0.02
        
        return (px, py, uncertainty)
    
    def get_velocity(self) -> tuple[float, float]:
        return (self._vx, self._vy)
    
    def get_acceleration(self) -> tuple[float, float]:
        return (self._acceleration_x, self._acceleration_y)
    
    def get_prediction_quality(self) -> float:
        """Return prediction confidence (0-1)."""
        return self._prediction_quality
    
    def get_speed(self) -> float:
        """Return total velocity magnitude."""
        return (self._vx ** 2 + self._vy ** 2) ** 0.5
    
    def is_stationary(self, threshold: float = 0.005) -> bool:
        return self.get_speed() < threshold
    
    def reset(self):
        self._x_filter.reset()
        self._y_filter.reset()
        self._vx = 0.0
        self._vy = 0.0
        self._prediction_quality = 1.0
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._frame_count = 0
        if hasattr(self, '_last_x'):
            del self._last_x
            del self._last_y


@dataclass
class TrackID:
    """Persistent tracker for a single detected object."""
    track_id: int
    label: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    conf: float = 0.0
    kalman: KalmanTracker2D = None
    age: int = 0
    hits: int = 0
    misses: int = 0
    first_seen: int = 0
    
    def __post_init__(self):
        if self.kalman is None:
            self.kalman = KalmanTracker2D()
    
    @property
    def confidence(self) -> float:
        """Track confidence based on hit/miss ratio."""
        total = self.hits + self.misses
        if total == 0:
            return 0.5
        return self.hits / total
    
    @property
    def is_stale(self) -> bool:
        return self.misses > 3
    
    @property
    def prediction_quality(self) -> float:
        return self.kalman.get_prediction_quality()


class WorldStateHistory(TypedDict):
    frame: int
    timestamp: float
    player: Optional[dict]
    enemies: list[dict]
    gadgets: list[dict]


@dataclass
class BlastZoneInfo:
    """Blast zone status for an object."""
    in_danger: bool
    danger_level: float  # 0-1
    nearest_edge: str  # "left", "right", "bottom", "safe"
    recovery_urgency: float  # 0-1
    recovery_direction: str  # recommended move


class GameState:
    """
    Advanced game state tracker with Kalman filtering, track management,
    blast zone awareness, and prediction quality scoring.
    """
    
    def __init__(self, max_history: int = 30):
        self._tracks: dict[int, TrackID] = {}
        self._next_id: int = 0
        self._frame_count: int = 0
        self._player: Optional[TrackID] = None
        
        # Stage awareness
        self._stage_edges: dict[str, float] = {
            "left": _STAGE_EDGE,
            "right": 1.0 - _STAGE_EDGE,
            "bottom": _STAGE_BOT,
            "top": _STAGE_TOP,
        }
        
        # Blast zone tracking
        self._blast_zones: dict[str, float] = {
            "left": _BLAST_ZONE_HORIZ,
            "right": 1.0 - _BLAST_ZONE_HORIZ,
            "bottom": 1.0 - _BLAST_ZONE_VERT,
        }
        
        # History for analysis
        self._history: deque[WorldStateHistory] = deque(maxlen=max_history)
        
        # Track statistics
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "player_tracked": 0,
            "enemies_tracked": 0,
            "avg_prediction_quality": 1.0,
            "total_frames": 0,
        }
        
        # Hit detection for feedback
        self._last_attack_result: bool = False
        self._attack_result_history: deque[bool] = deque(maxlen=20)
    
    def reset(self):
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        self._player = None
        self._history.clear()
    
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
    
    def _match_detection(self, det: dict) -> Optional[TrackID]:
        best_track = None
        best_iou = 0.3
        for track in self._tracks.values():
            if track.label != det.get("class_name", det.get("label", "")):
                continue
            cx1, cy1 = det.get("cx", det["x"] + det.get("w", 0) / 2), det.get("cy", det["y"] + det.get("h", 0) / 2)
            cx2, cy2 = track.x + track.w / 2, track.y + track.h / 2
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            # Convert TrackID to dict for IOU calculation
            track_dict = {"x": track.x, "y": track.y, "w": track.w, "h": track.h}
            iou = self._iou(det, track_dict)
            if iou > best_iou or dist < 0.03:
                if iou > best_iou:
                    best_iou = iou
                best_track = track
        return best_track
    
    def _update_track(self, track: TrackID, det: dict):
        cx = det.get("cx", det["x"] + det.get("w", 0) / 2)
        cy = det.get("cy", det["y"] + det.get("h", 0) / 2)
        
        smoothed_x, smoothed_y = track.kalman.update(cx, cy)
        
        track.x = det.get("x", track.x)
        track.y = det.get("y", track.y)
        track.w = det.get("w", det.get("width", track.w))
        track.h = det.get("h", det.get("height", track.h))
        track.conf = det.get("conf", track.conf)
        track.age += 1
        track.hits += 1
        track.misses = 0
        
        if track.label == "player":
            self._player = track
    
    def _create_track(self, det: dict) -> TrackID:
        track = TrackID(
            track_id=self._next_id,
            label=det.get("class_name", det.get("label", "unknown")),
            x=det.get("x", 0),
            y=det.get("y", 0),
            w=det.get("w", det.get("width", 0)),
            h=det.get("h", det.get("height", 0)),
            conf=det.get("conf", 0),
            first_seen=self._frame_count,
        )
        cx = track.x + track.w / 2
        cy = track.y + track.h / 2
        track.kalman.update(cx, cy)
        # BUG FIX: track was created but never added to tracks dict!
        self._tracks[track.track_id] = track
        self._next_id += 1
        self._stats["total_tracks"] += 1
        return track
    def _age_tracks(self):
        for track in list(self._tracks.values()):
            track.misses += 1
            if track.misses > 5:
                del self._tracks[track.track_id]
    
    def _get_blast_zone_info(self, obj: dict) -> BlastZoneInfo:
        """Calculate blast zone danger for an object."""
        cx = obj.get("cx", obj["x"] + obj.get("w", 0) / 2)
        cy = obj.get("cy", obj["y"] + obj.get("h", 0) / 2)
        vx = obj.get("vx", 0.0)
        vy = obj.get("vy", 0.0)
        
        danger_level = 0.0
        nearest_edge = "safe"
        recovery_direction = "none"
        recovery_urgency = 0.0
        
        # Check each edge
        left_danger = cx - self._blast_zones["left"]
        right_danger = self._blast_zones["right"] - cx
        bottom_danger = cy - self._blast_zones["bottom"]
        
        min_danger = min(left_danger, right_danger, bottom_danger)
        
        if cx < self._blast_zones["left"]:
            nearest_edge = "left"
            danger_level = 1.0 - (cx / self._blast_zones["left"])
            recovery_direction = "move_right"
            recovery_urgency = min(1.0, danger_level + abs(vx) * 2)
        elif cx > self._blast_zones["right"]:
            nearest_edge = "right"
            danger_level = 1.0 - ((1.0 - cx) / (1.0 - self._blast_zones["right"]))
            recovery_direction = "move_left"
            recovery_urgency = min(1.0, danger_level + abs(vx) * 2)
        elif cy > self._blast_zones["bottom"]:
            nearest_edge = "bottom"
            danger_level = 1.0 - ((1.0 - cy) / (1.0 - self._blast_zones["bottom"]))
            recovery_direction = "jump"
            recovery_urgency = min(1.0, danger_level + abs(vy) * 2)
        
        # Velocity augments danger
        if vx < -0.015 or vx > 0.015:
            recovery_urgency = min(1.0, recovery_urgency + 0.2)
        if vy > 0.01:
            recovery_urgency = min(1.0, recovery_urgency + 0.3)
        
        in_danger = danger_level > 0.3
        
        return BlastZoneInfo(
            in_danger=in_danger,
            danger_level=min(1.0, danger_level),
            nearest_edge=nearest_edge,
            recovery_urgency=min(1.0, recovery_urgency),
            recovery_direction=recovery_direction,
        )
    
    def _get_prediction_quality(self) -> float:
        """Calculate overall prediction quality across all tracks."""
        if not self._tracks:
            return 1.0
        qualities = [t.prediction_quality for t in self._tracks.values()]
        return sum(qualities) / len(qualities)
    
    def record_attack_result(self, hit: bool):
        """Record whether our last attack connected (for AI learning)."""
        self._attack_result_history.append(hit)
        self._last_attack_result = hit
    
    def get_attack_hit_rate(self) -> float:
        """Return recent attack hit rate (for AI calibration)."""
        if not self._attack_result_history:
            return 0.5
        return sum(1 for h in self._attack_result_history if h) / len(self._attack_result_history)
    
    def build_world_state(
        self, detections: list[dict], img_w: float, img_h: float
    ) -> dict:
        """Build complete world state from detections."""
        self._frame_count += 1
        self._stats["total_frames"] = self._frame_count
        for det in detections:
            if "cx" not in det:
                det["cx"] = det["x"] + det.get("w", det.get("width", 0)) / 2
                det["cy"] = det["y"] + det.get("h", det.get("height", 0)) / 2
        matched_tracks = set()
        for det in detections:
            track = self._match_detection(det)
            if track:
                self._update_track(track, det)
                matched_tracks.add(track.track_id)
            else:
                new_track = self._create_track(det)
                # Newly created tracks must be considered matched so they aren't penalized immediately
                if new_track is not None and hasattr(new_track, "track_id"):
                    matched_tracks.add(new_track.track_id)
        # Age and remove stale tracks (use list() to avoid dict mutation)
        for track in list(self._tracks.values()):
            if track.track_id not in matched_tracks:
                track.misses += 1
                if track.misses > 5:
                    del self._tracks[track.track_id]
        player = None
        final_enemies = []
        gadgets = []
        
        for track in self._tracks.values():
            obj = {
                "x": track.x,
                "y": track.y,
                "w": track.w,
                "h": track.h,
                "conf": track.conf,
                "cx": track.x + track.w / 2,
                "cy": track.y + track.h / 2,
                "track_id": track.track_id,
                "label": track.label,
            }
            
            # Kalman-smoothed position
            smoothed_x, smoothed_y = track.kalman.predict(0)
            obj["smooth_x"] = smoothed_x
            obj["smooth_y"] = smoothed_y
            
            # Velocity
            vx, vy = track.kalman.get_velocity()
            obj["vx"] = vx
            obj["vy"] = vy
            
            # Acceleration
            ax, ay = track.kalman.get_acceleration()
            obj["ax"] = ax
            obj["ay"] = ay
            
            # Prediction quality
            obj["pred_quality"] = track.prediction_quality
            
            # Blast zone info
            bz = self._get_blast_zone_info(obj)
            obj["blast_zone"] = {
                "in_danger": bz.in_danger,
                "danger_level": bz.danger_level,
                "nearest_edge": bz.nearest_edge,
                "recovery_direction": bz.recovery_direction,
                "recovery_urgency": bz.recovery_urgency,
            }
            
            # Speed
            obj["speed"] = track.kalman.get_speed()
            
            # Track confidence
            obj["track_confidence"] = track.confidence
            
            if track.label == "player":
                player = obj
            elif track.label == "enemy":
                if player:
                    dx = obj["cx"] - player["cx"]
                    dy = obj["cy"] - player["cy"]
                    obj["dist_to_player"] = (dx ** 2 + dy ** 2) ** 0.5
                final_enemies.append(obj)
            elif track.label in _GADGET_CLASSES:
                if player:
                    dx = obj["cx"] - player["cx"]
                    dy = obj["cy"] - player["cy"]
                    obj["dist_to_player"] = (dx ** 2 + dy ** 2) ** 0.5
                gadgets.append(obj)
        
        # Update statistics
        self._stats["active_tracks"] = len(self._tracks)
        self._stats["player_tracked"] = 1 if player else 0
        self._stats["enemies_tracked"] = len(final_enemies)
        self._stats["avg_prediction_quality"] = self._get_prediction_quality()
        
        # Sort enemies by distance to player
        final_enemies.sort(key=lambda e: e.get("dist_to_player", 999))
        
        state = {"player": player, "enemies": final_enemies, "gadgets": gadgets}
        
        self._history.append({
            "frame": self._frame_count,
            "timestamp": 0,
            "player": player,
            "enemies": final_enemies,
            "gadgets": gadgets,
        })
        return state
    def update_player_input(self, keys_pressed: list[tuple[str, int]]):
        if self._player is None:
            return
        
        for key_name, _ in keys_pressed:
            vx, vy = self._player.kalman.get_velocity()
            
            if key_name == "move_left":
                self._player.kalman.update(
                    self._player.x - _MOVE_STEP,
                    self._player.y
                )
            elif key_name == "move_right":
                self._player.kalman.update(
                    self._player.x + _MOVE_STEP,
                    self._player.y
                )
            elif key_name == "jump":
                self._player.kalman.update(
                    self._player.x,
                    self._player.y - _JUMP_STEP
                )
    
    def predict_object(self, track_id: int, steps: int = 5) -> tuple[float, float, float]:
        """Predict object position with uncertainty."""
        track = self._tracks.get(track_id)
        if not track:
            return (0.5, 0.5, 1.0)
        return track.kalman.predict_safe(steps)
    
    def predict_nearest_enemy(self, player_cx: float, player_cy: float,
                             steps: int = 5) -> tuple[float, float, float]:
        """Predict nearest enemy's position for attack planning."""
        if not self._tracks:
            return (0.5, 0.5, 1.0)
        
        enemy_tracks = [t for t in self._tracks.values() if t.label == "enemy"]
        if not enemy_tracks:
            return (0.5, 0.5, 1.0)
        
        # Find nearest enemy
        nearest = min(enemy_tracks, key=lambda t: abs(t.x - player_cx) + abs(t.y - player_cy))
        return nearest.kalman.predict_safe(steps)
    
    def is_near_edge(self, obj: dict) -> bool:
        cx = obj.get("cx", obj["x"] + obj.get("w", 0) / 2)
        cy = obj.get("cy", obj["y"] + obj.get("h", 0) / 2)
        
        return (cx < self._stage_edges["left"] or 
                cx > self._stage_edges["right"] or
                cy > self._stage_edges["bottom"])
    
    def is_in_danger(self, obj: dict) -> bool:
        bz = obj.get("blast_zone", {})
        return bz.get("in_danger", False)
    
    def get_recovery_path(self, obj: dict) -> str:
        if not self.is_near_edge(obj):
            return "none"
        
        bz = obj.get("blast_zone", {})
        return bz.get("recovery_direction", "jump")
    
    def get_recovery_urgency(self, obj: dict) -> float:
        """Get how urgent recovery is (0-1)."""
        bz = obj.get("blast_zone", {})
        return bz.get("recovery_urgency", 0.0)
    
    def is_player_tracked(self) -> bool:
        return self._player is not None
    
    def get_stats(self) -> dict:
        return {
            **self._stats,
            "history_size": len(self._history),
            "avg_tracks_per_frame": (
                sum(
                    1 + len(h.get("enemies", []))
                    for h in self._history
                )
                / len(self._history)
                if self._history
                else 0
            ),
            "attack_hit_rate": self.get_attack_hit_rate(),
        }


# ── Module-level singleton ───────────────────────────────────────
_state = GameState()


# ── Helper functions ─────────────────────────────────────────────

def _cx(obj: dict) -> float:
    w = obj.get("w") if "w" in obj else obj.get("width", 0.0)
    return float(obj["x"]) + float(w or 0.0) / 2.0


def _cy(obj: dict) -> float:
    h = obj.get("h") if "h" in obj else obj.get("height", 0.0)
    return float(obj["y"]) + float(h or 0.0) / 2.0




# ── Public API ───────────────────────────────────────────────────

def reset_tracking():
    """Reset all module-level tracking state."""
    _state.reset()


def update_player_input(keys_pressed: list[tuple[str, int]]):
    """Called after AI sends keys — moves the expected player position."""
    _state.update_player_input(keys_pressed)


def build_world_state(
    detections: list[dict], img_w: float, img_h: float
) -> dict:
    return _state.build_world_state(detections, img_w, img_h)


def get_tracker_stats() -> dict:
    """Return tracking statistics."""
    return _state.get_stats()


def is_player_tracked() -> bool:
    return _state._player is not None


def is_near_edge(obj: dict) -> bool:
    return _state.is_near_edge(obj)


def is_in_danger(obj: dict) -> bool:
    return _state.is_in_danger(obj)


def get_recovery_path(obj: dict) -> str:
    return _state.get_recovery_path(obj)


def get_recovery_urgency(obj: dict) -> float:
    """Get recovery urgency (0-1)."""
    return _state.get_recovery_urgency(obj)


def record_attack_result(hit: bool):
    """Record attack hit/miss for AI learning."""
    _state.record_attack_result(hit)


def get_attack_hit_rate() -> float:
    """Return recent attack hit rate."""
    return _state.get_attack_hit_rate()


def predict_object(track_id: int, steps: int = 5) -> tuple[float, float, float]:
    """Predict object position: (x, y, uncertainty)."""
    return _state.predict_object(track_id, steps)


def predict_nearest_enemy(player_cx: float, player_cy: float,
                          steps: int = 5) -> tuple[float, float, float]:
    """Predict nearest enemy position: (x, y, uncertainty)."""
    return _state.predict_nearest_enemy(player_cx, player_cy, steps)