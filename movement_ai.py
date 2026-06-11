__all__ = [
    "MovementPattern",
    "MovementDirection",
    "MovementState",
    "MovementConfig",
    "MovementResult",
    "StrafeController",
    "DashDanceController",
    "EdgeRecoveryController",
    "PlatformController",
    "DIController",
    "WeaponController",
    "ComboMovementController",
    "MovementAI",
    "create_movement_ai",
    "get_default_config",
]
#!/usr/bin/env python3
"""Advanced Movement AI — strafing, dash-dancing, edge recovery, platform awareness, DI prediction, and ledge options."""

import math
import random
from typing import Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

import logging

logger = logging.getLogger(__name__)


class MovementPattern(Enum):
    """Available movement patterns."""
    IDLE = auto()
    APPROACH = auto()
    RETREAT = auto()
    STRAFE_LEFT = auto()
    STRAFE_RIGHT = auto()
    OSCILLATE = auto()
    DASH_DANCE = auto()
    JUMP_MIXUP = auto()
    PLATFORM_DROP = auto()
    EDGE_RECOVERY = auto()
    WALL_AVOID = auto()
    # ── New patterns ──
    LEDGE_DASH = auto()
    DI_ESCAPE = auto()
    COMBO_FOLLOW = auto()
    WEAPON_RUSH = auto()
    FAST_FALL = auto()
    WALL_TECH = auto()
    HITSTUN_ESCAPE = auto()

class MovementDirection(Enum):
    """Movement direction."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    IDLE = "idle"


@dataclass
class MovementState:
    """Current movement state snapshot."""
    pattern: MovementPattern = MovementPattern.IDLE
    direction: MovementDirection = MovementDirection.IDLE
    is_airborne: bool = False
    is_strafing: bool = False
    strafe_phase: float = 0.0
    dash_count: int = 0
    last_dash_time: float = 0.0
    edge_proximity: float = 1.0  # 0 = at edge, 1 = safe
    platform_below: bool = False
    # ── New state fields ──
    hitstun_frames: int = 0
    di_active: bool = False
    at_ledge: bool = False
    weapon_active: bool = False
    combo_hit_confirm: bool = False


@dataclass
class MovementConfig:
    """Configuration for movement AI."""
    # Strafe settings
    strafe_speed: float = 0.8
    strafe_frequency: float = 2.0  # Hz
    strafe_amplitude: float = 0.05  # normalized screen units
    
    # Dash-dance settings
    dash_cooldown_ms: float = 200.0
    dash_distance: float = 0.08
    dash_random_chance: float = 0.3
    
    # Jump mixup settings
    jump_frequency: float = 0.15  # probability per frame when in range
    drop_through_chance: float = 0.2
    
    # Edge recovery
    edge_threshold: float = 0.06
    recovery_urgency: float = 0.8
    safe_zone_buffer: float = 0.10
    
    # Platform awareness
    platform_height_threshold: float = 0.15
    drop_height_threshold: float = 0.20
    
    # ── New config fields ──
    # DI settings
    di_strength: float = 0.7  # how strong DI input is
    di_predict_window: int = 5  # frames to predict hit trajectory
    fast_fall_threshold: float = 0.08  # Y velocity threshold for fast-fall
    
    # Ledge settings
    ledge_dash_chance: float = 0.5
    ledge_wakeup_mixup: bool = True
    ledge_jump_delay: int = 3  # frames before jump from ledge
    
    # Hitstun settings
    hitstun_escape_odds: float = 0.4
    hitstun_di_window: int = 8
    
    # Combo movement
    combo_follow_chance: float = 0.7
    combo_walk_speed: float = 0.6


@dataclass
class MovementResult:
    """Result of movement decision."""
    primary_action: Optional[str] = None
    secondary_action: Optional[str] = None
    pattern: MovementPattern = MovementPattern.IDLE
    confidence: float = 0.0
    reasoning: str = ""
    # ── New result fields ──
    di_direction: Optional[str] = None  # "left", "right", "up", "down"
    ledge_option: Optional[str] = None  # "ledgedash", "ledge_jump", "neutral_getup"
    fast_fall: bool = False
    wall_tech: bool = False


class StrafeController:
    """Controls strafing patterns to confuse enemy and maintain optimal spacing."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._phase: float = 0.0
        self._direction: float = 1.0  # 1 or -1
        self._mode: str = "random"  # random, oscillate, defensive
        self._switch_history: deque[bool] = deque(maxlen=8)
    
    def update(self, dt: float) -> tuple[Optional[str], float]:
        self._phase += dt * self._config.strafe_frequency * self._direction
        
        if self._mode == "oscillate":
            if abs(self._phase - 0.5) > 0.45:
                self._direction *= -1
            action = "move_right" if self._direction > 0 else "move_left"
        
        elif self._mode == "defensive":
            if abs(self._phase) < 0.1:
                return None, self._phase
            action = "move_right" if self._phase > 0 else "move_left"
        
        else:  # random
            if random.random() < 0.02:
                self._direction *= -1
                self._switch_history.append(True)
            else:
                self._switch_history.append(False)
            action = "move_right" if self._direction > 0 else "move_left"
        
        return action, self._phase
    
    def set_mode(self, mode: str):
        if mode in ("random", "oscillate", "defensive"):
            self._mode = mode
    
    def get_switch_rate(self) -> float:
        """Return how often we switch direction (for unpredictability tuning)."""
        if not self._switch_history:
            return 0.0
        return sum(1 for s in self._switch_history if s) / len(self._switch_history)
    
    def reset(self):
        self._phase = 0.0
        self._direction = random.choice([-1, 1])
        self._switch_history.clear()


class DashDanceController:
    """Controls dash-dancing for spacing and baiting."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._last_dash_time: float = 0.0
        self._dash_direction: int = 1  # 1 = toward enemy, -1 = away
        self._consecutive_dashes: int = 0
        self._last_action_time: float = 0.0
        self._dash_intent_history: deque[bool] = deque(maxlen=5)  # toward vs away
    
    def can_dash(self, current_time: float) -> bool:
        return (current_time - self._last_dash_time) >= (self._config.dash_cooldown_ms / 1000.0)
    
    def should_dash(self, enemy_dist: float, enemy_approaching: bool, 
                   player_airborne: bool, enemy_velocity_x: float = 0.0,
                   player_at_ledge: bool = False) -> bool:
        """Decide whether to dash with enhanced awareness."""
        if player_airborne:
            return False
        
        # Ledge dash opportunity
        if player_at_ledge:
            if random.random() < self._config.ledge_dash_chance:
                return True
        
        # Dash when enemy is in punish range
        if enemy_dist < 0.08:
            return random.random() < 0.7
        
        # Dash to bait punishment when enemy approaching
        if enemy_approaching and enemy_dist < 0.15:
            return random.random() < 0.4
        
        # Whiff-punish bait: dash after enemy attacks
        if abs(enemy_velocity_x) > 0.015 and not enemy_approaching:
            return random.random() < 0.5
        
        return False
    
    def get_dash_action(self, toward_enemy: bool, player_at_ledge: bool = False) -> tuple[str, Optional[str]]:
        """
        Get dash action direction and option.
        
        Returns:
            (movement_action, ledge_option or None)
        """
        if player_at_ledge:
            # Ledge dash is special
            self._dash_intent_history.append(True)
            return "move_left", "ledgedash"  # direction based on ledge position
        
        direction = 1 if toward_enemy else -1
        self._dash_intent_history.append(toward_enemy)
        return "move_right" if direction > 0 else "move_left", None
    
    def record_dash(self, current_time: float, toward: bool):
        self._last_dash_time = current_time
        self._consecutive_dashes += 1
    
    def get_bait_pattern(self) -> str:
        """Analyze dash pattern to predict next intent."""
        if not self._dash_intent_history:
            return "random"
        toward_count = sum(1 for t in self._dash_intent_history if t)
        total = len(self._dash_intent_history)
        ratio = toward_count / total
        if ratio > 0.7:
            return "aggressive"
        elif ratio < 0.3:
            return "defensive"
        return "mixed"
    
    def reset(self):
        self._last_dash_time = 0.0
        self._consecutive_dashes = 0
        self._dash_intent_history.clear()


class EdgeRecoveryController:
    """Handles edge/blast zone awareness and recovery with ledge options."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._stage_left: float = 0.04
        self._stage_right: float = 0.96
        self._stage_bottom: float = 0.92
        self._recovery_path: Optional[str] = None
        self._in_danger: bool = False
        self._ledge_frame_count: int = 0
        self._last_was_ledge: bool = False
    
    def check_edge_proximity(self, player_x: float, player_y: float,
                            player_vx: float, player_vy: float) -> float:
        """Calculate edge proximity (0 = at edge, 1 = safe)."""
        left_dist = player_x - self._stage_left
        right_dist = self._stage_right - player_x
        
        horiz_proximity = min(left_dist, right_dist) / self._config.edge_threshold
        
        vert_proximity = max(0, player_y - self._stage_bottom) / 0.1
        
        if player_vx < -0.01:
            horiz_proximity *= 1.2
        if player_vx > 0.01:
            horiz_proximity *= 1.2
        
        return min(1.0, horiz_proximity + vert_proximity * 0.5)
    
    def is_at_ledge(self, player_x: float, player_y: float) -> bool:
        """Check if player is at a ledge (for ledge options)."""
        near_left = player_x < self._stage_left + self._config.safe_zone_buffer * 1.5
        near_right = player_x > self._stage_right - self._config.safe_zone_buffer * 1.5
        at_y = player_y < self._stage_bottom - 0.05  # Not at very bottom
        return (near_left or near_right) and at_y
    
    def get_recovery_action(self, player_x: float, player_y: float,
                           player_vx: float = 0.0, player_vy: float = 0.0) -> tuple[Optional[str], Optional[str]]:
        """
        Get recommended recovery action with ledge options.
        
        Returns:
            (movement_action, ledge_option or None)
        """
        # Check if at ledge
        at_ledge = self.is_at_ledge(player_x, player_y)
        if at_ledge:
            self._ledge_frame_count += 1
        else:
            self._ledge_frame_count = 0
        
        # Left edge recovery
        if player_x < self._stage_left + self._config.safe_zone_buffer:
            if player_vy > 0.01:  # Moving down (falling off)
                return "jump", "ledge_jump"
            return "move_right", None
        
        # Right edge recovery
        if player_x > self._stage_right - self._config.safe_zone_buffer:
            if player_vy > 0.01:
                return "jump", "ledge_jump"
            return "move_left", None
        
        # Falling recovery
        if player_y > self._stage_bottom:
            return "jump", None
        
        return None, None
    
    def get_ledge_option(self, player_x: float, player_y: float,
                        enemy_x: float, enemy_y: float,
                        frame_count: int) -> str:
        """
        Choose ledge option (ledgedash, ledge jump, neutral getup).
        Mix up to avoid being read.
        """
        dist_to_enemy = math.hypot(enemy_x - player_x, enemy_y - player_y)
        
        # Short ledge hang before options
        if frame_count < self._config.ledge_jump_delay:
            return "wait"
        
        # Enemy far — more freedom
        if dist_to_enemy > 0.30:
            if random.random() < 0.4:
                return "ledgedash"
            elif random.random() < 0.6:
                return "ledge_jump"
            return "neutral_getup"
        
        # Enemy close — ledgedash to punish their edge guard
        if dist_to_enemy < 0.15:
            if random.random() < 0.7:
                return "ledgedash"
            return "neutral_getup"
        
        # Medium range — mix it up
        roll = random.random()
        if roll < 0.3:
            return "ledgedash"
        elif roll < 0.6:
            return "ledge_jump"
        return "neutral_getup"
    
    def is_dangerous_position(self, player_x: float, player_y: float) -> bool:
        return (player_x < self._stage_left or 
                player_x > self._stage_right or 
                player_y > self._stage_bottom)
    
    def get_blast_zone_danger(self, player_x: float, player_y: float,
                             player_vx: float, player_vy: float) -> float:
        """Calculate danger level (0-1) of current position."""
        danger = 0.0
        
        # Horizontal blast zone
        if player_x < self._stage_left:
            danger = max(danger, 1.0 - (player_x / self._stage_left))
        if player_x > self._stage_right:
            danger = max(danger, 1.0 - ((1.0 - player_x) / (1.0 - self._stage_right)))
        
        # Vertical blast zone
        if player_y > self._stage_bottom:
            danger = max(danger, 1.0 - ((1.0 - player_y) / (1.0 - self._stage_bottom)))
        
        # Velocity augments danger
        if player_vx < -0.015:
            danger = max(danger, 0.7)
        if player_vx > 0.015:
            danger = max(danger, 0.7)
        if player_vy > 0.01:
            danger = max(danger, 0.6)
        
        return min(1.0, danger)


class PlatformController:
    """Handles platform awareness and drop-through decisions."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._platform_y: float = 0.5
        self._platform_height: float = 0.02
        self._platform_x: float = 0.5
        self._platform_width: float = 0.4
        self._last_drop_frame: int = 0
        self._drop_pattern: deque[bool] = deque(maxlen=6)
    
    def update_platform(self, y: float, height: float, x: float = 0.5, width: float = 0.4):
        self._platform_y = y
        self._platform_height = height
        self._platform_x = x
        self._platform_width = width
    
    def should_drop_through(self, player_y: float, player_x: float,
                           player_above_platform: bool, below_enemy: bool,
                           enemy_dist: float, frame: int) -> bool:
        """Decide whether to drop through platform with pattern awareness."""
        if not player_above_platform:
            return False
        
        if below_enemy:
            return False
        
        # Check drop pattern to avoid predictability
        recent_drops = sum(1 for d in self._drop_pattern if d)
        if recent_drops >= 3:
            return random.random() < 0.2  # Rare drop if we've been dropping
        
        # Enemy approaching from below platform — stay up
        if enemy_dist < 0.12:
            return False
        
        return random.random() < self._config.drop_through_chance
    
    def record_drop(self, did_drop: bool, frame: int):
        self._drop_pattern.append(did_drop)
        self._last_drop_frame = frame
    
    def get_drop_action(self) -> tuple[str, str]:
        return ("move_down", "jump")
    
    def is_on_platform(self, player_y: float, player_x: float) -> bool:
        """Check if player is currently on a platform."""
        y_match = abs(player_y - self._platform_y) < self._platform_height * 2
        x_match = self._platform_x - self._platform_width/2 < player_x < self._platform_x + self._platform_width/2
        return y_match and x_match


class DIController:
    """Directional Influence controller for hitstun escape and combo break."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._di_active: bool = False
        self._di_direction: str = "none"
        self._hitstun_frames: int = 0
        self._last_hit_frame: int = 0
        self._di_window_remaining: int = 0
    
    def detect_hitstun(self, player_vx: float, player_vy: float,
                      player_y: float, frame: int,
                      knockback_threshold: float = 0.02) -> bool:
        """Detect if player is in hitstun (high velocity after being hit)."""
        speed = math.hypot(player_vx, player_vy)
        if speed > knockback_threshold and player_vy > 0:
            self._hitstun_frames += 1
            self._last_hit_frame = frame
            self._di_window_remaining = self._config.hitstun_di_window
            return True
        return False
    
    def calculate_di(self, player_x: float, player_y: float,
                    knockback_angle: float, enemy_x: float,
                    enemy_y: float) -> str:
        """
        Calculate optimal DI direction to escape combo or reach stage.
        
        Args:
            knockback_angle: estimated angle of knockback (radians)
            enemy_x, enemy_y: enemy position for DI toward/away decision
        
        Returns:
            DI direction: "up", "down", "left", "right", or combined like "up+left"
        """
        # Determine if we want to DI toward stage or away from enemy
        toward_stage = player_y > 0.7  # Lower = toward blast zone, higher = toward stage
        
        # DI away from enemy to reduce follow-up damage
        away_from_enemy = player_x < enemy_x  # enemy is right of us
        
        # Mix with pure survival DI
        di_options = []
        
        if knockback_angle < -0.5:  # Strong upward knockback
            di_options.append("down")  # DI down to land faster
        elif knockback_angle > 0.5:  # Downward knockback
            di_options.append("up")  # DI up to survive longer
        
        if away_from_enemy:
            di_options.append("right")
        else:
            di_options.append("left")
        
        # Return combined DI
        if len(di_options) == 2:
            return f"{di_options[0]}+{di_options[1]}"
        return di_options[0] if di_options else "up"
    
    def should_fast_fall(self, player_vy: float, player_y: float,
                        stage_bottom: float = 0.92) -> bool:
        """Decide whether to fast-fall (quick down input) to land faster."""
        if player_vy < self._config.fast_fall_threshold:
            return True
        # Also fast-fall when high above stage
        if player_y < stage_bottom - 0.15 and player_vy > 0.005:
            return True
        return False
    
    def get_di_action(self) -> Optional[str]:
        """Get DI input action if active."""
        if self._di_window_remaining > 0:
            self._di_window_remaining -= 1
            return self._di_direction
        return None
    
    def reset(self):
        self._di_active = False
        self._di_direction = "none"
        self._hitstun_frames = 0
        self._di_window_remaining = 0


class WeaponController:
    """Handles weapon/gadget pickup and usage awareness."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._has_weapon: bool = False
        self._weapon_type: str = "none"
        self._weapon_timer: int = 0
        self._pickup_cooldown: int = 0
    
    def update(self, gadgets: list[dict], player_x: float, player_y: float):
        """Check for weapon pickup."""
        if self._pickup_cooldown > 0:
            self._pickup_cooldown -= 1
        
        for g in gadgets:
            if g.get("label") == "weapon":
                gx = g.get("cx", 0.5)
                gy = g.get("cy", 0.5)
                dist = math.hypot(gx - player_x, gy - player_y)
                if dist < 0.08 and not self._has_weapon and self._pickup_cooldown == 0:
                    self._has_weapon = True
                    self._weapon_type = "weapon"
                    self._weapon_timer = 300  # ~5 seconds
                    self._pickup_cooldown = 60
    
    def use_weapon(self) -> bool:
        """Decide if we should use the weapon."""
        if self._weapon_timer > 0:
            self._weapon_timer -= 1
            return self._weapon_timer > 60  # Use it in last 4 seconds
        return False
    
    def has_weapon(self) -> bool:
        return self._has_weapon
    
    def reset(self):
        self._has_weapon = False
        self._weapon_type = "none"
        self._weapon_timer = 0


class ComboMovementController:
    """Handles movement during combo strings (walk forward to chase, etc.)."""
    
    def __init__(self, config: MovementConfig):
        self._config = config
        self._in_combo: bool = False
        self._combo_step: int = 0
        self._walk_direction: int = 0
    
    def start_combo(self, enemy_x: float, player_x: float):
        self._in_combo = True
        self._combo_step = 0
        self._walk_direction = 1 if enemy_x > player_x else -1
    
    def step_combo(self, enemy_x: float, player_x: float) -> Optional[str]:
        """Get movement during combo step."""
        if not self._in_combo:
            return None
        
        self._combo_step += 1
        dx = enemy_x - player_x
        
        # Walk toward enemy during combo
        if abs(dx) > 0.05:
            walk = "move_right" if dx > 0 else "move_left"
            return walk
        
        return None
    
    def end_combo(self, hit_confirm: bool):
        self._in_combo = False
        self._combo_step = 0
    
    def is_chasing(self) -> bool:
        return self._in_combo and self._combo_step > 0


class MovementAI:
    """Advanced movement AI combining all movement controllers."""
    
    def __init__(self, config: Optional[MovementConfig] = None):
        self._config = config or MovementConfig()
        
        # Initialize controllers
        self._strafe = StrafeController(self._config)
        self._dash_dance = DashDanceController(self._config)
        self._edge_recovery = EdgeRecoveryController(self._config)
        self._platform = PlatformController(self._config)
        self._di = DIController(self._config)
        self._weapon = WeaponController(self._config)
        self._combo = ComboMovementController(self._config)
        
        # State tracking
        self._state = MovementState()
        self._frame: int = 0
        self._last_update_time: float = 0.0
        self._rng = random.Random()
        
        # Action history for pattern detection
        self._recent_actions: deque[str] = deque(maxlen=10)
        
        # Blast zone tracking
        self._blast_zone_danger: float = 0.0
    
    def update(self, state: dict, frame: int, current_time: float,
              combo_hit_confirm: bool = False) -> MovementResult:
        """
        Update movement AI and return recommended action.
        
        Args:
            state: Game state dict with player, enemies, gadgets
            frame: Current frame number
            current_time: Current timestamp
            combo_hit_confirm: Whether last action was a confirmed hit
        
        Returns:
            MovementResult with recommended action
        """
        self._frame = frame
        self._last_update_time = current_time
        
        player = state.get("player")
        enemies = state.get("enemies", [])
        gadgets = state.get("gadgets", [])
        
        if not player:
            return MovementResult(pattern=MovementPattern.IDLE, confidence=0.0)
        
        # Get player position and velocity
        px = player.get("cx", 0.5)
        py = player.get("cy", 0.5)
        pvx = player.get("vx", 0.0)
        pvy = player.get("vy", 0.0)
        
        # Update state
        self._state.is_airborne = abs(pvy) > 0.005 or py < 0.7
        self._state.edge_proximity = self._edge_recovery.check_edge_proximity(px, py, pvx, pvy)
        self._state.at_ledge = self._edge_recovery.is_at_ledge(px, py)
        self._blast_zone_danger = self._edge_recovery.get_blast_zone_danger(px, py, pvx, pvy)
        
        # Update weapon awareness
        self._weapon.update(gadgets, px, py)
        self._state.weapon_active = self._weapon.has_weapon()
        
        # Update combo movement
        if combo_hit_confirm and not self._combo.is_chasing():
            enemy = enemies[0] if enemies else None
            if enemy:
                self._combo.start_combo(enemy.get("cx", 0.5), px)
        self._state.combo_hit_confirm = combo_hit_confirm
        
        # ── PRIORITY 1: BLAST ZONE RECOVERY ──
        if self._blast_zone_danger > 0.7:
            recovery_move, ledge_opt = self._edge_recovery.get_recovery_action(px, py, pvx, pvy)
            if recovery_move:
                return MovementResult(
                    primary_action=recovery_move,
                    secondary_action="jump" if ledge_opt == "ledge_jump" else None,
                    pattern=MovementPattern.EDGE_RECOVERY,
                    confidence=0.95,
                    reasoning=f"Blast zone escape ({self._blast_zone_danger:.1f})",
                    ledge_option=ledge_opt,
                )
        
        # ── PRIORITY 2: DI / HITSTUN ESCAPE ──
        in_hitstun = self._di.detect_hitstun(pvx, pvy, py, frame)
        if in_hitstun and self._di._di_window_remaining > 0:
            di_action = self._di.get_di_action()
            return MovementResult(
                primary_action=di_action,
                pattern=MovementPattern.HITSTUN_ESCAPE,
                confidence=0.8,
                reasoning="DI escape from hitstun",
            )
        
        # ── PRIORITY 3: FAST FALL ──
        if self._state.is_airborne and self._di.should_fast_fall(pvy, py):
            return MovementResult(
                primary_action=None,
                secondary_action="move_down",
                pattern=MovementPattern.FAST_FALL,
                confidence=0.7,
                reasoning="Fast-fall to land faster",
                fast_fall=True,
            )
        
        # ── PRIORITY 4: LEDGE OPTIONS ──
        if self._state.at_ledge:
            enemy = enemies[0] if enemies else None
            ex, ey = (enemy.get("cx", 0.5), enemy.get("cy", 0.5)) if enemy else (0.5, 0.5)
            ledge_opt = self._edge_recovery.get_ledge_option(px, py, ex, ey, self._edge_recovery._ledge_frame_count)
            
            if ledge_opt == "ledgedash":
                return MovementResult(
                    primary_action="move_left" if px < 0.5 else "move_right",
                    secondary_action="dash",
                    pattern=MovementPattern.LEDGE_DASH,
                    confidence=0.75,
                    reasoning="Ledge dash off stage",
                    ledge_option="ledgedash",
                )
            elif ledge_opt == "ledge_jump":
                return MovementResult(
                    primary_action="jump",
                    secondary_action=None,
                    pattern=MovementPattern.JUMP_MIXUP,
                    confidence=0.7,
                    reasoning="Ledge jump",
                    ledge_option="ledge_jump",
                )
            elif ledge_opt == "neutral_getup":
                return MovementResult(
                    primary_action="idle",
                    pattern=MovementPattern.IDLE,
                    confidence=0.6,
                    reasoning="Neutral getup from ledge",
                    ledge_option="neutral_getup",
                )
        
        # ── PRIORITY 5: COMBO CHASE MOVEMENT ──
        if self._combo.is_chasing():
            chase_move = self._combo.step_combo(
                enemies[0].get("cx", 0.5) if enemies else 0.5,
                px
            )
            if chase_move:
                return MovementResult(
                    primary_action=chase_move,
                    pattern=MovementPattern.COMBO_FOLLOW,
                    confidence=0.8,
                    reasoning="Chasing during combo",
                )
        
        # ── PRIORITY 6: DANGER RECOVERY ──
        if self._edge_recovery.is_dangerous_position(px, py):
            recovery_action, ledge_opt = self._edge_recovery.get_recovery_action(px, py, pvx, pvy)
            if recovery_action:
                return MovementResult(
                    primary_action=recovery_action,
                    secondary_action="jump",
                    pattern=MovementPattern.EDGE_RECOVERY,
                    confidence=0.9,
                    reasoning="Emergency recovery from edge",
                    ledge_option=ledge_opt,
                )
        
        # Get enemy info
        enemy = enemies[0] if enemies else None
        if not enemy:
            return MovementResult(pattern=MovementPattern.IDLE, confidence=0.0)
        
        ex = enemy.get("cx", 0.5)
        ey = enemy.get("cy", 0.5)
        dist = math.hypot(ex - px, ey - py)
        enemy_approaching = enemy.get("vx", 0) * (ex - px) < 0
        evx = enemy.get("vx", 0.0)
        
        # ── RANGE-BASED MOVEMENT ──
        if dist > 0.25:  # Far — approach
            return self._approach_movement(px, py, ex, ey, evx, enemy_approaching, current_time)
        
        elif dist > 0.12:  # Medium — mixups
            return self._mixup_movement(px, py, ex, ey, evx, enemy_approaching, current_time)
        
        else:  # Close — control space
            return self._close_range_movement(px, py, ex, ey, evx, enemy_approaching, current_time)
    
    def _approach_movement(self, px: float, py: float, ex: float, ey: float,
                          evx: float, enemy_approaching: bool, current_time: float) -> MovementResult:
        """Handle approaching movement with dash-dance bait."""
        dx = ex - px
        move = "move_right" if dx > 0 else "move_left"
        
        # Try dash-dance approach if enemy is moving
        if self._dash_dance.should_dash(dist=0.25, enemy_approaching=enemy_approaching,
                                         player_airborne=self._state.is_airborne,
                                         enemy_velocity_x=evx,
                                         player_at_ledge=self._state.at_ledge):
            if self._dash_dance.can_dash(current_time):
                dash_action, ledge_opt = self._dash_dance.get_dash_action(toward_enemy=True)
                self._dash_dance.record_dash(current_time, toward=True)
                
                return MovementResult(
                    primary_action=dash_action,
                    secondary_action="dash",
                    pattern=MovementPattern.DASH_DANCE,
                    confidence=0.7,
                    reasoning="Dash-dance approach",
                    ledge_option=ledge_opt,
                )
        
        # Occasional jump approach
        if self._frame % 7 == 0 and not self._state.is_airborne:
            self._state.pattern = MovementPattern.APPROACH
            return MovementResult(
                primary_action=move,
                secondary_action="jump",
                pattern=MovementPattern.APPROACH,
                confidence=0.7,
                reasoning="Approaching with jump"
            )
        
        self._state.pattern = MovementPattern.APPROACH
        return MovementResult(
            primary_action=move,
            pattern=MovementPattern.APPROACH,
            confidence=0.6,
            reasoning="Approaching enemy"
        )
    
    def _mixup_movement(self, px: float, py: float, ex: float, ey: float,
                       evx: float, enemy_approaching: bool, current_time: float) -> MovementResult:
        """Handle mixed movement options with strafing and dash-dance."""
        dx = ex - px
        
        # Strafe while closing distance
        strafe_action, self._state.strafe_phase = self._strafe.update(1.0 / 60.0)
        self._state.is_strafing = True
        
        # Dash-dance occasionally
        if self._dash_dance.should_dash(0.15, enemy_approaching, self._state.is_airborne, evx):
            if self._dash_dance.can_dash(current_time):
                toward = dx < 0
                dash_action, ledge_opt = self._dash_dance.get_dash_action(not toward)
                self._dash_dance.record_dash(current_time, not toward)
                
                self._state.pattern = MovementPattern.DASH_DANCE
                return MovementResult(
                    primary_action=dash_action,
                    secondary_action="dash",
                    pattern=MovementPattern.DASH_DANCE,
                    confidence=0.65,
                    reasoning="Dash-dance to bait",
                    ledge_option=ledge_opt,
                )
        
        # Jump mixups
        if not self._state.is_airborne and self._rng.random() < self._config.jump_frequency:
            self._state.pattern = MovementPattern.JUMP_MIXUP
            move = "move_right" if dx > 0 else "move_left"
            return MovementResult(
                primary_action=move,
                secondary_action="jump",
                pattern=MovementPattern.JUMP_MIXUP,
                confidence=0.55,
                reasoning="Jump mixup"
            )
        
        # Platform drop mixup
        if not self._state.is_airborne and self._rng.random() < 0.05:
            if self._platform.should_drop_through(py, px, True, False, dist, self._frame):
                self._platform.record_drop(True, self._frame)
                self._state.pattern = MovementPattern.PLATFORM_DROP
                action, down = self._platform.get_drop_action()
                return MovementResult(
                    primary_action=action,
                    secondary_action=down,
                    pattern=MovementPattern.PLATFORM_DROP,
                    confidence=0.5,
                    reasoning="Platform drop mixup"
                )
        
        # Regular strafing approach
        self._state.pattern = MovementPattern.STRAFE_LEFT if self._state.strafe_phase < 0 else MovementPattern.STRAFE_RIGHT
        return MovementResult(
            primary_action=strafe_action,
            pattern=self._state.pattern,
            confidence=0.5,
            reasoning="Strafing approach"
        )
    
    def _close_range_movement(self, px: float, py: float, ex: float, ey: float,
                             evx: float, enemy_approaching: bool, current_time: float) -> MovementResult:
        """Handle close range spacing with DI awareness."""
        dx = ex - px
        
        # High enemy approach = maintain distance
        if enemy_approaching:
            # Dash away to bait
            if self._dash_dance.can_dash(current_time) and self._rng.random() < 0.3:
                dash_action, ledge_opt = self._dash_dance.get_dash_action(False)
                self._dash_dance.record_dash(current_time, False)
                
                self._state.pattern = MovementPattern.DASH_DANCE
                return MovementResult(
                    primary_action=dash_action,
                    secondary_action="dash",
                    pattern=MovementPattern.DASH_DANCE,
                    confidence=0.75,
                    reasoning="Escape pressure with dash",
                    ledge_option=ledge_opt,
                )
            
            # Oscillating retreat
            self._strafe.set_mode("defensive")
            strafe_action, self._state.strafe_phase = self._strafe.update(1.0 / 60.0)
            
            self._state.pattern = MovementPattern.RETREAT
            return MovementResult(
                primary_action=strafe_action,
                pattern=MovementPattern.RETREAT,
                confidence=0.65,
                reasoning="Retreating from pressure"
            )
        
        # Enemy not approaching — offensive movement
        self._strafe.set_mode("oscillate")
        strafe_action, self._state.strafe_phase = self._strafe.update(1.0 / 60.0)
        
        # Jump pressure
        if not self._state.is_airborne and self._frame % 5 == 0:
            self._state.pattern = MovementPattern.JUMP_MIXUP
            return MovementResult(
                primary_action=strafe_action,
                secondary_action="jump",
                pattern=MovementPattern.JUMP_MIXUP,
                confidence=0.55,
                reasoning="Pressure with jump"
            )
        
        self._state.pattern = MovementPattern.OSCILLATE
        return MovementResult(
            primary_action=strafe_action,
            pattern=self._state.pattern,
            confidence=0.5,
            reasoning="Controlling space"
        )
    
    def get_state(self) -> MovementState:
        """Get current movement state."""
        return self._state
    
    def get_blast_zone_danger(self) -> float:
        return self._blast_zone_danger
    
    def reset(self):
        """Reset all movement controllers."""
        self._strafe.reset()
        self._dash_dance.reset()
        self._di.reset()
        self._frame = 0
        self._recent_actions.clear()


# Convenience function for integration with ai_brain
def create_movement_ai(config: Optional[MovementConfig] = None) -> MovementAI:
    """Create a new MovementAI instance."""
    return MovementAI(config)


def get_default_config() -> MovementConfig:
    """Get default movement configuration."""
    return MovementConfig()