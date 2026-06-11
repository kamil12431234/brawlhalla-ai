#!/usr/bin/env python3
__all__ = [
    "ActionScore",
    "TrajectoryPoint",
    "KalmanPredictor",
    "TrajectoryPredictor",
    "FrameHistory",
    "ActionCandidate",
    "StrategicActionScorer",
    "EnemyPatternLearner",
    "BrawlAI",
]
"""Brawlhalla AI — combat-grade opponent with adaptive learning and strategic decision-making."""

import math
import os
import logging
import random
from typing import Optional, TypedDict
from collections import deque
from dataclasses import dataclass, field

from config import (
    KEYS,
    STAGE_L, STAGE_R,
    STAGE_BOT as STAGE_B,
    STAGE_TOP,
)

logger = logging.getLogger(__name__)

CHARACTER = os.environ.get("BH_CHARACTER", "balanced")

# ── Hitbox zones (normalized) ──
HIT_CLOSE = 0.07
HIT_MEDIUM = 0.12
HIT_FAR = 0.18
HIT_SPECIAL = 0.22
DODGE_RANGE = 0.04
SHIELD_RANGE = 0.06

# ── Helper functions ──
def _cx(p): return float(p["x"]) + float(p.get("w", p.get("width", 0))) / 2.0
def _cy(p): return float(p["y"]) + float(p.get("h", p.get("height", 0))) / 2.0
def _dist(a, b): return math.hypot(_cx(a) - _cx(b), _cy(a) - _cy(b))
def _clamp(v, lo, hi): return max(lo, min(hi, v))


class ActionScore(TypedDict):
    action: str
    score: float
    risk: float
    reward: float


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    t: int


class KalmanPredictor:
    """Kalman filter for smooth trajectory prediction."""
    def __init__(self, process_noise: float = 0.001, measurement_noise: float = 0.1):
        self._q = process_noise
        self._r = measurement_noise
        self._x = 0.0
        self._v = 0.0
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
    
    def predict(self, steps: int = 1) -> float:
        return self._x + self._v * steps
    
    def get_velocity(self) -> float:
        return self._v


class TrajectoryPredictor:
    """Predicts enemy movement using Kalman filtering."""
    def __init__(self):
        self._kf_x = KalmanPredictor()
        self._kf_y = KalmanPredictor()
        self._last_t = 0
    
    def update(self, x: float, y: float, t: int):
        dt = max(1, t - self._last_t)
        if self._last_t > 0:
            vx = (x - self._kf_x._x) / dt
            vy = (y - self._kf_y._x) / dt
            self._kf_x._v = vx * 0.5
            self._kf_y._v = vy * 0.5
        self._kf_x.update(x)
        self._kf_y.update(y)
        self._last_t = t
    
    def predict(self, steps: int = 5) -> tuple[float, float]:
        return self._kf_x.predict(steps), self._kf_y.predict(steps)


@dataclass
class FrameHistory:
    ex: deque = field(default_factory=deque)
    ey: deque = field(default_factory=deque)
    px: deque = field(default_factory=deque)
    py: deque = field(default_factory=deque)
    frames: deque = field(default_factory=deque)
    
    def push(self, e, p, frame: int):
        self.ex.append(_cx(e))
        self.ey.append(_cy(e))
        if p:
            self.px.append(_cx(p))
            self.py.append(_cy(p))
        else:
            # Pad px/py with previous value to keep lengths aligned when player missing.
            last_px = self.px[-1] if self.px else 0.5
            last_py = self.py[-1] if self.py else 0.5
            self.px.append(last_px)
            self.py.append(last_py)
        self.frames.append(frame)

        # Trim to max length safely (all deques are kept in sync).
        maxlen = 60
        for q in (self.ex, self.ey, self.px, self.py, self.frames):
            while len(q) > maxlen:
                q.popleft()
    
    def approaching(self) -> bool:
        if len(self.ex) < 5:
            return True
        return self.ex[-1] > self.ex[-5]
    
    def predict_enemy_position(self, frames_ahead: int = 5) -> tuple[float, float]:
        if len(self.ex) < 2:
            return self.ex[-1] if self.ex else 0.5, self.ey[-1] if self.ey else 0.5
        vx = (self.ex[-1] - self.ex[-2]) * 0.8
        vy = (self.ey[-1] - self.ey[-2]) * 0.8
        pred_x = _clamp(self.ex[-1] + vx * frames_ahead, 0.0, 1.0)
        pred_y = _clamp(self.ey[-1] + vy * frames_ahead, 0.0, 1.0)
        return pred_x, pred_y


@dataclass
class ActionCandidate:
    action: str
    score: float = 0.0
    risk: float = 0.0
    reward: float = 0.0
    timing_bonus: float = 0.0
    
    def total_score(self) -> float:
        return self.score + self.timing_bonus - self.risk * 0.5


class StrategicActionScorer:
    """Scores actions based on strategic factors."""
    def __init__(self, aggression: float = 0.5):
        self._aggression = aggression
        self._hit_history = deque(maxlen=100)
        self._recent_accuracy = 0.5
    
    def record_hit_result(self, was_hit: bool):
        self._hit_history.append(1.0 if was_hit else 0.0)
        if len(self._hit_history) >= 20:
            self._recent_accuracy = sum(self._hit_history) / len(self._hit_history)
    
    def score_attack(self, attack_type: str, dist: float, approaching: bool,
                     enemy_vel: tuple, airborne: bool, pred_conf: float) -> ActionCandidate:
        base_score = 50.0
        risk = 0.3
        if dist < HIT_CLOSE:
            base_score += 30.0 * self._aggression
            risk = 0.2
        elif dist < HIT_MEDIUM:
            base_score += 15.0 * self._aggression
            risk = 0.35
        elif dist < HIT_FAR:
            base_score += 5.0
            risk = 0.5
        if approaching:
            base_score += 10.0
            risk -= 0.1
        if "heavy" in attack_type:
            base_score += 15.0 if self._aggression > 0.6 else 0.0
            risk += 0.1
        if airborne:
            base_score += 5.0
        if pred_conf > 0.6:
            base_score += 10.0 * pred_conf
        reward = base_score * self._recent_accuracy
        return ActionCandidate(action=attack_type, score=base_score, risk=risk, reward=reward)
    
    def score_movement(self, move_type: str, dist: float, edge_prox: float,
                       enemy_vel: tuple, blast_zone: float) -> ActionCandidate:
        score = 10.0
        risk = 0.1
        if dist > HIT_FAR:
            score += 15.0
        if blast_zone > 0.3 and "move" in move_type:
            score += 20.0
        if edge_prox < 0.05:
            risk += 0.2
        return ActionCandidate(action=move_type, score=score, risk=risk)
    
    def score_defense(self, defense_type: str, dist: float, approaching: bool,
                      enemy_vel: tuple, dodge_correct: str) -> ActionCandidate:
        score = 20.0 if approaching else 5.0
        risk = 0.2
        if dodge_correct == "correct":
            score += 15.0
            risk -= 0.1
        return ActionCandidate(action=defense_type, score=score, risk=risk)


class EnemyPatternLearner:
    """Learns enemy movement patterns for prediction."""
    def __init__(self):
        self._dodge_patterns: deque = deque(maxlen=100)
        self._attack_timing: deque = deque(maxlen=50)
        self._pattern_confidence = 0.3
    
    def record_dodge(self, direction: str):
        self._dodge_patterns.append(direction)
        if len(self._dodge_patterns) >= 10:
            last_10 = list(self._dodge_patterns)[-10:]
            most_common = max(set(last_10), key=last_10.count)
            consistency = last_10.count(most_common) / 10
            self._pattern_confidence = 0.3 + consistency * 0.4
    
    def record_attack_timing(self, timing: float):
        self._attack_timing.append(timing)
    
    def get_dodge_direction(self, player_x: float, enemy_x: float, enemy_vel_x: float) -> str:
        if len(self._dodge_patterns) < 5:
            return "away"
        last_dodges = list(self._dodge_patterns)[-10:]
        return max(set(last_dodges), key=last_dodges.count)
    
    def predict_attack_probability(self) -> float:
        if len(self._attack_timing) < 3:
            return 0.3
        recent = list(self._attack_timing)[-5:]
        avg_interval = sum(recent) / len(recent) if recent else 1.0
        if avg_interval < 0.5:
            return 0.8
        elif avg_interval < 1.0:
            return 0.5
        return 0.3


_ACTION_MAP = {
    "move_left": KEYS["left"],
    "move_right": KEYS["right"],
    "jump": KEYS["jump"],
    "light_attack": KEYS["light_attack"],
    "heavy_attack": KEYS["heavy_attack"],
    "special": KEYS["special"],
    "shield_back": KEYS["shield"],
    "dash": KEYS["dash"],
    "throw": KEYS["throw"],
}

# ── Combo patterns ──
GROUND_COMBO = ["light_attack", "light_attack", "light_attack"]
HEAVY_PRESSURE = ["heavy_attack", "light_attack", "heavy_attack"]
AIR_COMBO = ["light_attack", "jump", "light_attack"]
READ_DODGE = ["heavy_attack", "dash"]
EDGE_GUARD = ["heavy_attack", "heavy_attack"]

COMBO_PATTERNS = {
    "ground": GROUND_COMBO,
    "heavy_pressure": HEAVY_PRESSURE,
    "air": AIR_COMBO,
    "read_dodge": READ_DODGE,
    "edge_guard": EDGE_GUARD,
}

style_map = {
    "mixed": "ground",
    "heavy": "heavy_pressure",
}


class BrawlAI:
    def __init__(self, mode: str = "full", aggressive: float = 0.5):
        self.mode = mode
        self.aggression = aggressive
        
        self.hist = FrameHistory()
        self._scorer = StrategicActionScorer(aggressive)
        self._pattern_learner = EnemyPatternLearner()
        
        self.frame = 0
        self._stale = 0
        self._pr = 0
        self._last_special_frame = -1000
        self._recently_close = False
        
        self._enemy_vel_x = 0.0
        self._enemy_vel_y = 0.0
        self._player_vel_x = 0.0
        self._player_vel_y = 0.0
        self._last_enemy_x = 0.5
        self._last_enemy_y = 0.5
        self._last_player_x = 0.5
        self._last_player_y = 0.5
        
        self._near_edge = False
        self._player_airborne = False
        self._blast_zone_danger = 0.0
        self._pred_confidence = 0.5
        self._multi_enemy_mode = False
        self._primary_enemy = None
        
        self._danger_zone = 0.08
        self._smooth_decay = 0.7
        
        self._hit_streak = 0
        self._miss_streak = 0
        self._combo_step = 0
        self.cq: list[str] = []
        
        self._rng = random.Random()
        self._rng.seed()
    
    def choose_action(self, state: dict, now_ms=None) -> Optional[str]:
        self.frame += 1
        p = state.get("player")
        enemies = state.get("enemies") or []
        gs = state.get("gadgets") or []
        e = self._pick_enemy(p, enemies)
        self._multi_enemy_mode = len(enemies) > 1
        if self._multi_enemy_mode and e:
            self._primary_enemy = e
        # No player detected
        if not p:
            return self._no_player()
        # No enemy detected
        if not e:
            return self._no_enemy(p, gs)
        # Both player and enemy detected - update state and decide
        self.hist.push(e, p, self.frame)
        self._update_velocities(e, p)
        prox = self._proximity(p, e)
        if prox == "close":
            self._recently_close = True
        if self._pr > 0:
            self._pr -= 1
        player_y = _cy(p)
        self._player_airborne = abs(player_y - self._last_player_y) > 0.02
        self._last_player_y = player_y
        self._update_edge_awareness(p)
        self._blast_zone_danger = self._calc_blast_zone_danger(p)
        self._pred_confidence = self._calc_prediction_confidence(e, p)
        # Reset stale counter when we see both player and enemy
        self._stale = 0
        # Choose action based on mode
        if self.mode == "combat":
            return self._maybe_special(self._combat_strategic(p, e))
        if self.mode == "poke":
            return self._poke(p, e)
        if self.mode == "assist":
            return self._maybe_special(self._assist(p, e, prox))
        # Default (full / other modes).
        return self._maybe_special(self._assault_strategic(p, e, prox, gs))
    
    def _calc_blast_zone_danger(self, p) -> float:
        cx = _cx(p)
        danger = 0.0
        if cx < STAGE_L + 0.05:
            danger = max(danger, 1.0 - (cx / (STAGE_L + 0.05)))
        if cx > STAGE_R - 0.05:
            danger = max(danger, 1.0 - ((1.0 - cx) / (STAGE_R - 0.05)))
        cy = _cy(p)
        if cy > STAGE_B - 0.05:
            danger = max(danger, 1.0 - ((1.0 - cy) / (STAGE_B - 0.05)))
        return min(1.0, danger)
    
    def _calc_prediction_confidence(self, e, p) -> float:
        conf = 0.5
        if len(self.hist.ex) >= 5:
            vx_vals = []
            for i in range(1, min(len(self.hist.ex), 5)):
                vx_vals.append(self.hist.ex[-i] - self.hist.ex[-i-1])
            if vx_vals:
                vx_std = self._std(vx_vals)
                conf += (0.3 - min(0.3, vx_std * 10))
        conf += self._pattern_learner._pattern_confidence * 0.3
        return max(0.1, min(0.95, conf))
    
    def _std(self, vals: list[float]) -> float:
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    
    def _update_velocities(self, e, p):
        ex, ey = _cx(e), _cy(e)
        px, py = _cx(p), _cy(p)
        self._enemy_vel_x = self._smooth_decay * self._enemy_vel_x + (1 - self._smooth_decay) * (ex - self._last_enemy_x)
        self._enemy_vel_y = self._smooth_decay * self._enemy_vel_y + (1 - self._smooth_decay) * (ey - self._last_enemy_y)
        self._player_vel_x = self._smooth_decay * self._player_vel_x + (1 - self._smooth_decay) * (px - self._last_player_x)
        self._player_vel_y = self._smooth_decay * self._player_vel_y + (1 - self._smooth_decay) * (py - self._last_player_y)
        self._last_enemy_x = ex
        self._last_enemy_y = ey
        self._last_player_x = px
        self._last_player_y = py
    
    def _update_edge_awareness(self, p):
        cx = _cx(p)
        self._near_edge = cx < STAGE_L + self._danger_zone or cx > STAGE_R - self._danger_zone
    
    def _pick_enemy(self, p, enemies):
        """Pick target enemy with FALLBACK for missing detections."""
        if not enemies:
            # FALLBACK: Create heuristic enemy when model doesn't detect.
            if p:
                cx = _cx(p)
                fake_enemy_x = 1.0 - cx + 0.1 if cx > 0.5 else cx + 0.1
                return {
                    "x": _clamp(fake_enemy_x, 0.2, 0.8),
                    "y": p.get("y", 0.5),
                    "w": 0.08,
                    "h": 0.18,
                    "conf": 0.3,
                    "class_name": "enemy",
                }
            return None
        if p is None:
            # If player not detected, pick highest-confidence enemy.
            return max(enemies, key=lambda x: x.get("conf", 0))
        # Normal case: nearest enemy to player.
        return min(enemies, key=lambda x: _dist(p, x))
    
    def _proximity(self, p, e):
        d = _dist(p, e)
        if d < HIT_CLOSE:
            return "close"
        if d < HIT_MEDIUM:
            return "medium"
        return "far"
    
    def _combat_strategic(self, p, e) -> Optional[str]:
        d = _dist(p, e)
        appr = self.hist.approaching()
        
        if self._blast_zone_danger > 0.7:
            return self._get_blast_zone_recovery(p)
        if d < SHIELD_RANGE and appr:
            return "shield_back"
        if d < DODGE_RANGE and appr:
            return self._evade_directional("jump", e, p)
        if d < HIT_CLOSE:
            if self.cq:
                return self.cq.pop(0)
            return self._combo("ground")
        if d < HIT_MEDIUM and self.aggression > 0.5:
            return "light_attack"
        return self._move_to(e, p)
    
    def _poke(self, p, e) -> Optional[str]:
        d = _dist(p, e)
        if d < HIT_CLOSE:
            return "light_attack"
        if d < HIT_MEDIUM and self._rng.random() < 0.7:
            return "light_attack"
        return self._move_to(e, p)
    
    def _assault_strategic(self, p, e, prox, gs) -> str:
        if self._blast_zone_danger > 0.6:
            recovery = self._get_blast_zone_recovery(p)
            if recovery:
                return recovery
        
        d = _dist(p, e)
        appr = self.hist.approaching()
        e_above = _cy(e) < _cy(p) - 0.05
        
        enemy_attack_prob = self._pattern_learner.predict_attack_probability()
        
        pred_x, pred_y = self.hist.predict_enemy_position(frames_ahead=5)
        pred_dist = math.hypot(pred_x - _cx(p), pred_y - _cy(p))
        effective_dist = min(d, pred_dist)
        
        candidates: list[ActionCandidate] = []
        pred_conf = self._pred_confidence
        
        # ATTACK OPTIONS
        if effective_dist < HIT_CLOSE:
            candidates.append(self._scorer.score_attack("light_attack", effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
            candidates.append(self._scorer.score_attack(self._combo("ground"), effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
            if self.aggression > 0.6:
                candidates.append(self._scorer.score_attack(self._combo("heavy_pressure"), effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
        
        if effective_dist < HIT_MEDIUM:
            candidates.append(self._scorer.score_attack("light_attack", effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
            if self.aggression > 0.5:
                candidates.append(self._scorer.score_attack("heavy_attack", effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
        
        if effective_dist < HIT_SPECIAL:
            candidates.append(self._scorer.score_attack("special", effective_dist, appr, (self._enemy_vel_x, self._enemy_vel_y), self._player_airborne, pred_conf))
        
        # MOVEMENT
        move_action = self._move_to(e, p)
        edge_proximity = min(_cx(p) - STAGE_L, STAGE_R - _cx(p))
        candidates.append(self._scorer.score_movement(move_action, d, edge_proximity, (self._enemy_vel_x, self._enemy_vel_y), self._blast_zone_danger))
        
        # DEFENSE
        dodge_correct = "correct" if (enemy_attack_prob > 0.5) else "away"
        if d < SHIELD_RANGE and appr:
            candidates.append(self._scorer.score_defense("shield_back", d, appr, (self._enemy_vel_x, self._enemy_vel_y), dodge_correct))
        if d < DODGE_RANGE:
            candidates.append(self._scorer.score_defense("jump", d, appr, (self._enemy_vel_x, self._enemy_vel_y), dodge_correct))
        
        if not candidates:
            return move_action
        
        if self.aggression < 0.4:
            candidates = [c for c in candidates if c.risk < 0.3] or candidates
        
        # If still empty (e.g., all filtered out), fall back to move-only
        if not candidates:
            return move_action

        for c in candidates:
            if "attack" in c.action and enemy_attack_prob > 0.6:
                c.timing_bonus = 15.0 * enemy_attack_prob
        
        best = max(candidates, key=lambda c: c.total_score())
        
        # Human-like randomization
        if self._rng.random() < 0.15:
            valid_candidates = [c for c in candidates if c.risk < 0.4]
            if valid_candidates:
                best = self._rng.choice(valid_candidates)
        
        # Combined attack+movement
        if best.action in ("move_left", "move_right"):
            if self.aggression > 0.5 and self._rng.random() < 0.4:
                attack = "light_attack" if d < HIT_MEDIUM else "heavy_attack"
                return f"{best.action}+{attack}"
            if d > 0.20 and self._rng.random() < 0.3:
                return f"{best.action}+jump"
        
        return best.action
    
    def _get_blast_zone_recovery(self, p) -> Optional[str]:
        cx = _cx(p)
        cy = _cy(p)
        vert_dist = STAGE_B - cy
        
        if vert_dist > 0.1:
            return "jump"
        
        stage_center = 0.5
        if abs(cx - stage_center) > abs(cy - STAGE_B):
            return "move_right" if cx < stage_center else "move_left"
        return "jump"
    
    def _assist(self, p, e, prox):
        if self.cq:
            return self.cq.pop(0)
        
        d = _dist(p, e)
        appr = self.hist.approaching()
        
        if d < SHIELD_RANGE and appr:
            return "shield_back"
        if d < DODGE_RANGE:
            return self._evade_directional("jump", e, p)
        if d < HIT_CLOSE:
            return self._combo("ground")
        if d < HIT_MEDIUM and self.aggression > 0.4:
            return "light_attack"
        return None
    
    def _maybe_special(self, token):
        if not token or "special" not in str(token):
            return token
        
        # Cooldown guard for special move
        cooldown = 40
        if (self.frame - self._last_special_frame) < cooldown:
            # If compound action (e.g. "move_right+special"), keep non-special part
            parts = [p.strip() for p in str(token).split("+")]
            non_special_parts = [p for p in parts if "special" not in p]
            return "+".join(non_special_parts) or None
        self._last_special_frame = self.frame
        return token
    
    def _combo(self, t: str) -> str:
        """Return next action in current combo sequence.

        If the queue is empty or we're switching styles, initialize a new combo.
        This prevents resetting mid-combo and allows multi-hit combos to execute.
        """
        if not self.cq:
            key = t if t in COMBO_PATTERNS else style_map.get(t, "ground")
            combo = COMBO_PATTERNS.get(key, ["light_attack"])
            # Queue the remaining steps after the first action.
            self.cq.extend(combo[1:])
            return combo[0]

        # Continue existing combo by popping next step.
        return self.cq.pop(0)
    
    def _move_to(self, e, p):
        dx = _cx(e) - _cx(p)
        if abs(dx) < 0.015:
            return "idle"
        return "move_right" if dx > 0 else "move_left"
    
    def _no_player(self):
        if self.mode in ("assist", "combat", "poke"):
            return None
        # Training mode fallback: patrol movement when player not detected
        # This keeps AI useful even when player detection fails
        if self._stale > 60:
            # Every 60 frames, move in a pattern
            pattern = self._stale // 60 % 4
            if pattern == 0:
                return "move_right"
            elif pattern == 1:
                return "move_left"
            elif pattern == 2:
                return "jump"
            else:
                return None
        return None  # Stay idle when recently saw detections
    
    def _no_enemy(self, p, gs):
        self._stale += 1
        if gs and self.mode in ("full", "assist"):
            g = min(gs, key=lambda x: _dist(p, x))
            gd = _dist(p, g)
            if gd < 0.15 or (self._stale > 30 and self.frame % 8 == 0):
                return self._move_to(g, p)
        return "idle"
    
    def _evade_directional(self, token, e, p):
        base = "move_left"
        if e and p:
            dx = _cx(e) - _cx(p)
            base = "move_left" if dx > 0 else "move_right"
        if self.frame % 3 == 0:
            return f"{base}+jump"
        return base
    
    def record_hit(self, was_hit: bool):
        self._scorer.record_hit_result(was_hit)
        if was_hit:
            self._hit_streak += 1
            self._miss_streak = 0
        else:
            self._miss_streak += 1
            self._hit_streak = 0
            self.cq.clear()
        self._hit_streak = min(self._hit_streak, 10)
        self._miss_streak = min(self._miss_streak, 5)
    
    def resolve_action(self, token: Optional[str], state=None) -> list[tuple[str, int]]:
        """Translate a high-level action token into key presses.

        Returns list of (key_code, semantic_name).
        """
        if not token or token == "idle":
            return []

        keys_to_press: list[tuple[str, int]] = []

        parts = token.split("+")

        for key in parts:
            key_lower = key.lower().strip()
            if key_lower in _ACTION_MAP:
                keys_to_press.append((_ACTION_MAP[key_lower], key_lower))
            elif key_lower == "evade":
                # Evade away from last known enemy position.
                move_key = (
                    _ACTION_MAP["move_left"]
                    if self._last_enemy_x > 0.5
                    else _ACTION_MAP["move_right"]
                )
                keys_to_press.append((move_key, "move_toward_center"))
                keys_to_press.append((_ACTION_MAP["dash"], "dash"))
                keys_to_press.append((_ACTION_MAP["jump"], "jump"))
            else:
                # Log once per unknown action (at DEBUG level to avoid spam).
                logger.debug("[AI] Unknown action token part: %s", key_lower)

        return keys_to_press
    
    def get_stats(self) -> dict:
        return {
            "frame": self.frame,
            "aggression": self.aggression,
            "hit_streak": self._hit_streak,
            "miss_streak": self._miss_streak,
            "near_edge": self._near_edge,
            "blast_zone_danger": self._blast_zone_danger,
        }