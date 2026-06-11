__all__ = [
    "CharacterProfile",
    "MatchupProfile",
    "InputMap",
    "TrainingConfig",
    "get_config_summary",
    "get_matchup_profile",
    "validate_config",
    "get_training_parser",
]
"""Brawlhalla AI — configuration, character profiles, matchup profiles, and learning parameters."""

import os
import logging
from typing import Final, TypedDict
from dataclasses import dataclass
logger = logging.getLogger(__name__)

# ── Validation helpers ────────────────────────────────────────────

def _validate_positive(value: float, name: str, min_val: float = 0.0) -> float:
    if value < min_val:
        raise ValueError(f"Invalid {name}: {value} < {min_val}")
    return float(value)


def _validate_range(value: float, name: str, min_val: float, max_val: float) -> float:
    val = _validate_positive(value, name, min_val)
    if val > max_val:
        raise ValueError(f"Invalid {name}: {value} > {max_val}")
    return val


# ── Roboflow ─────────────────────────────────────────────────────
ROBOFLOW_API_KEY: Final[str] = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE: Final[str] = "rasheds-workspace"
ROBOFLOW_PROJECT_NAME: Final[str] = "brawlhalla-vision"
ROBOFLOW_VERSION: Final[int] = int(os.getenv("ROBOFLOW_VERSION", "5"))

# ── Model ────────────────────────────────────────────────────────
LOCAL_MODEL_ONNX_PATH: Final[str] = os.getenv("BH_MODEL_PATH", "./model/brawlhalla_vision.onnx")
# Auto-detect GPU availability
def _auto_detect_device() -> str:
    """Auto-detect best available device."""
    requested = os.getenv("BH_DEVICE", "cuda")
    if requested == "cpu":
        return "cpu"
    # Check CUDA availability
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "cpu"
DEVICE: Final[str] = _auto_detect_device()
CONFIDENCE_THRESHOLD: Final[float] = float(os.getenv("BH_CONF_THRESH", "0.15"))

# ── Stage geometry (normalized) — shared across modules ───────────
STAGE_L: Final[float] = 0.03         # left edge of stage
STAGE_R: Final[float] = 0.97         # right edge of stage
STAGE_BOT: Final[float] = 0.92       # bottom (blast zone top)
STAGE_TOP: Final[float] = 0.08       # top boundary reference
BLAST_ZONE_HORIZ: Final[float] = 0.02   # horizontal danger margin
BLAST_ZONE_VERT: Final[float] = 0.05    # vertical danger margin

# ── Game window capture ──────────────────────────────────────────
CAPTURE_WIDTH: Final[int] = int(os.getenv("BH_WIDTH", "1280"))
CAPTURE_HEIGHT: Final[int] = int(os.getenv("BH_HEIGHT", "720"))
REGION_LEFT: Final[int] = int(os.getenv("BH_LEFT", "0"))
REGION_TOP: Final[int] = int(os.getenv("BH_TOP", "0"))

AUTO_DETECT_WINDOW: Final[bool] = os.getenv("BH_AUTO_DETECT", "true").lower() == "true"


def _detect_game_window():
    """Auto-detect whether a Brawlhalla window exists (for logging).

    NOTE: Geometry parsing is deferred; this function currently only reports
    presence of the game window.
    """
    try:
        import subprocess as sp
        result = sp.run(
            ["xdotool", "search", "--name", "Brawlhalla"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False

        window_id = result.stdout.strip().split()[0]
        win_result = sp.run(
            ["xdotool", "getwindowgeometry", window_id],
            capture_output=True, text=True, timeout=3,
        )
        if win_result.returncode == 0:
            logger.info("[CONFIG] Detected Brawlhalla window")
            return True

        # Window found but geometry read failed – still consider it detected.
        logger.debug("[CONFIG] Found Brawlhalla window, geometry unavailable.")
        return False

    except (FileNotFoundError, sp.TimeoutExpired):
        logger.debug("[CONFIG] xdotool unavailable or timed out; skipping auto-detect.")
        return False
    except Exception:
        logger.debug("[CONFIG] Error during auto-detect; falling back to manual config.")
        return False


# ── AI behavior ──────────────────────────────────────────────────
ACTION_COOLDOWN_MS: Final[int] = int(os.getenv("BH_COOLDOWN", "120"))
ENEMY_PROXIMITY_AGGRESSIVE: Final[int] = 200
ENEMY_PROXIMITY_DEFENSIVE: Final[int] = 400

# ── Character profiles ──────────────────────────────────────────

class CharacterProfile(TypedDict):
    """Character profile configuration."""
    aggression: float
    combo_style: str
    sweet_spot: float
    reaction_delay: int
    dodge_aggression: float
    shield_timing: float
    # ── New fields ──
    preferred_range: str  # "close", "medium", "far"
    recovery_priority: float  # 0-1 how much to prioritize recovery over offense
    combo_length: int  # max combo steps before resetting
    ledge_options: list[str]  # preferred ledge options
    di_preference: str  # "away", "toward", "up", "down"


CHARACTER_PROFILES: dict[str, CharacterProfile] = {
    "balanced": {
        "aggression": 0.5,
        "combo_style": "mixed",
        "sweet_spot": 0.10,
        "reaction_delay": 3,
        "dodge_aggression": 0.5,
        "shield_timing": 0.15,
        "preferred_range": "close",
        "recovery_priority": 0.7,
        "combo_length": 3,
        "ledge_options": ["ledgedash", "ledge_jump", "neutral_getup"],
        "di_preference": "away",
    },
    "aggressive": {
        "aggression": 0.8,
        "combo_style": "heavy",
        "sweet_spot": 0.08,
        "reaction_delay": 2,
        "dodge_aggression": 0.3,
        "shield_timing": 0.10,
        "preferred_range": "close",
        "recovery_priority": 0.4,
        "combo_length": 5,
        "ledge_options": ["ledgedash", "ledge_jump"],
        "di_preference": "toward",
    },
    "defensive": {
        "aggression": 0.3,
        "combo_style": "mixed",
        "sweet_spot": 0.12,
        "reaction_delay": 4,
        "dodge_aggression": 0.7,
        "shield_timing": 0.20,
        "preferred_range": "medium",
        "recovery_priority": 0.9,
        "combo_length": 2,
        "ledge_options": ["neutral_getup", "ledge_jump"],
        "di_preference": "up",
    },
    "tournament": {
        "aggression": 0.6,
        "combo_style": "mixed",
        "sweet_spot": 0.09,
        "reaction_delay": 2,
        "dodge_aggression": 0.5,
        "shield_timing": 0.15,
        "preferred_range": "close",
        "recovery_priority": 0.8,
        "combo_length": 4,
        "ledge_options": ["ledgedash", "neutral_getup"],
        "di_preference": "away",
    },
    # ── New character profiles ──
    "zuko": {
        "aggression": 0.55,
        "combo_style": "heavy",
        "sweet_spot": 0.11,
        "reaction_delay": 3,
        "dodge_aggression": 0.45,
        "shield_timing": 0.15,
        "preferred_range": "close",
        "recovery_priority": 0.6,
        "combo_length": 4,
        "ledge_options": ["ledgedash", "ledge_jump"],
        "di_preference": "away",
    },
    "bodvar": {
        "aggression": 0.7,
        "combo_style": "heavy",
        "sweet_spot": 0.09,
        "reaction_delay": 2,
        "dodge_aggression": 0.35,
        "shield_timing": 0.12,
        "preferred_range": "close",
        "recovery_priority": 0.5,
        "combo_length": 5,
        "ledge_options": ["ledgedash"],
        "di_preference": "toward",
    },
    "orion": {
        "aggression": 0.5,
        "combo_style": "mixed",
        "sweet_spot": 0.10,
        "reaction_delay": 3,
        "dodge_aggression": 0.55,
        "shield_timing": 0.16,
        "preferred_range": "medium",
        "recovery_priority": 0.75,
        "combo_length": 3,
        "ledge_options": ["neutral_getup", "ledgedash"],
        "di_preference": "away",
    },
    "ember": {
        "aggression": 0.65,
        "combo_style": "heavy",
        "sweet_spot": 0.10,
        "reaction_delay": 3,
        "dodge_aggression": 0.4,
        "shield_timing": 0.13,
        "preferred_range": "close",
        "recovery_priority": 0.55,
        "combo_length": 4,
        "ledge_options": ["ledgedash", "ledge_jump"],
        "di_preference": "toward",
    },
}


# ── Matchup profiles ─────────────────────────────────────────────
class MatchupProfile(TypedDict):
    """Matchup-specific adjustments."""
    aggression_modifier: float  # +/- from base aggression
    preferred_spacing: float  # preferred distance to maintain
    anti_aggressive: bool  # play more defensively against this character
    ledge_preference: str  # how aggressively to use ledge options
    dodge_bias: str  # "away", "toward", "none"
    punish_window: int  # frames to look for punish opportunity


MATCHUP_PROFILES: dict[str, dict[str, MatchupProfile]] = {
    # vs aggressive characters — play defensive, bait, punish
    "aggressive": {
        "bodvar": {"aggression_modifier": -0.2, "preferred_spacing": 0.15, "anti_aggressive": True,
                   "ledge_preference": "neutral", "dodge_bias": "away", "punish_window": 15},
        "zuko": {"aggression_modifier": -0.15, "preferred_spacing": 0.14, "anti_aggressive": True,
                 "ledge_preference": "neutral", "dodge_bias": "away", "punish_window": 14},
        "ember": {"aggression_modifier": -0.1, "preferred_spacing": 0.13, "anti_aggressive": True,
                  "ledge_preference": "passive", "dodge_bias": "away", "punish_window": 12},
    },
    # vs defensive characters — pressure, force mistakes
    "defensive": {
        "orion": {"aggression_modifier": 0.2, "preferred_spacing": 0.08, "anti_aggressive": False,
                  "ledge_preference": "aggressive", "dodge_bias": "away", "punish_window": 10},
    },
    # vs balanced — play your game
    "balanced": {
        "tournament": {"aggression_modifier": 0.0, "preferred_spacing": 0.10, "anti_aggressive": False,
                       "ledge_preference": "mixed", "dodge_bias": "away", "punish_window": 12},
    },
}


# Validate character is in profiles
_valid_chars = list(CHARACTER_PROFILES.keys())
DEFAULT_CHARACTER = "balanced"

def _get_character() -> str:
    char = os.environ.get("BH_CHARACTER", DEFAULT_CHARACTER)
    if char not in _valid_chars:
        logger.warning("[CONFIG] Unknown character '%s', using 'balanced'", char)
        return DEFAULT_CHARACTER
    return char

CHARACTER: Final[str] = _get_character()

# Apply character profile
_profile: CharacterProfile = CHARACTER_PROFILES[CHARACTER]
AGGRESSION: Final[float] = float(os.getenv("BH_AGGRESSION", str(_profile["aggression"])))
COMBO_STYLE: Final[str] = _profile["combo_style"]
SWEET_SPOT: Final[float] = _profile["sweet_spot"]
REACTION_DELAY: Final[int] = _profile["reaction_delay"]
DODGE_AGGRESSION: Final[float] = _profile["dodge_aggression"]
SHIELD_TIMING: Final[float] = _profile["shield_timing"]

# ── Input mapping ────────────────────────────────────────────────

class InputMap(TypedDict):
    left: str
    right: str
    jump: str
    light_attack: str
    heavy_attack: str
    special: str
    shield: str
    dash: str
    throw: str


KEYS: InputMap = {
    "left": "a",
    "right": "d",
    "jump": "w",
    "light_attack": "j",
    "heavy_attack": "k",
    "special": "l",
    "shield": "s",
    "dash": "Shift_L",
    "throw": "e",
}


# ── Learning parameters ──────────────────────────────────────────
LEARNING_ENABLED: Final[bool] = os.getenv("BH_LEARNING", "false").lower() == "true"
LEARNING_RATE: Final[float] = float(os.getenv("BH_LR", "0.001"))
ADAPTIVE_AGGRESSION: Final[bool] = os.getenv("BH_ADAPTIVE", "true").lower() == "true"

# Reinforcement learning parameters
RL_GAMMA: Final[float] = float(os.getenv("BH_RL_GAMMA", "0.95"))  # discount factor
RL_EPSILON: Final[float] = float(os.getenv("BH_RL_EPSILON", "0.1"))  # exploration rate
RL_EPSILON_DECAY: Final[float] = float(os.getenv("BH_RL_EPSILON_DECAY", "0.995"))
RL_BATCH_SIZE: Final[int] = int(os.getenv("BH_RL_BATCH", "32"))
RL_MEMORY_SIZE: Final[int] = int(os.getenv("BH_RL_MEMORY", "10000"))
RL_TARGET_UPDATE: Final[int] = int(os.getenv("BH_RL_TARGET_UPDATE", "100"))

# Pattern learning
PATTERN_MEMORY: Final[int] = int(os.getenv("BH_PATTERN_MEMORY", "50"))
ATTACK_PREDICTION_WINDOW: Final[int] = int(os.getenv("BH_ATTACK_WINDOW", "60"))


# ── Performance settings ─────────────────────────────────────────
TARGET_FPS: Final[int] = int(os.getenv("BH_TARGET_FPS", "45"))
ENABLE_TEMPORAL_SMOOTHING: Final[bool] = os.getenv("BH_TEMPORAL", "true").lower() == "true"
ENABLE_GPU_ACCELERATION: Final[bool] = os.getenv("BH_GPU", "true").lower() == "true"

# Adaptive quality
ADAPTIVE_QUALITY: Final[bool] = os.getenv("BH_ADAPTIVE_QUALITY", "true").lower() == "true"
QUALITY_DROP_THRESHOLD: Final[float] = float(os.getenv("BH_QUALITY_THRESHOLD", "45"))  # FPS threshold


# ── Training configuration ───────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = DEVICE  # Use auto-detected device instead of hard-coded "cuda"
    patience: int = 10
    imgsz: int = 640
    half: bool = True
    workers: int = 4
    cos_lr: bool = False
    close_mosaic: int = 10
    
    def __post_init__(self):
        self.epochs = max(1, min(500, int(self.epochs)))
        self.batch_size = max(1, min(64, int(self.batch_size)))
        self.learning_rate = max(1e-6, min(0.1, float(self.learning_rate)))
        self.patience = max(5, min(100, int(self.patience)))


def get_training_parser() -> argparse.ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(description="Brawlhalla AI Training")
    parser.add_argument("--model", default="yolov8m.pt", help="Model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--project", default="runs/train", help="Project path")
    parser.add_argument("--name", default="brawlhalla", help="Run name")
    parser.add_argument("--half", action="store_true", help="FP16 training")
    parser.add_argument("--workers", type=int, default=4, help="Data workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--cos_lr", action="store_true", help="Cosine LR schedule")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Mosaic disable epoch")
    return parser


# ── Helper functions ─────────────────────────────────────────────

def get_config_summary() -> str:
    return (
        f"Character: {CHARACTER}\n"
        f"Aggression: {AGGRESSION:.2f} | Style: {COMBO_STYLE}\n"
        f"Target FPS: {TARGET_FPS} | GPU: {ENABLE_GPU_ACCELERATION}\n"
        f"Learning: {LEARNING_ENABLED} | Adaptive: {ADAPTIVE_AGGRESSION}\n"
        f"RL: gamma={RL_GAMMA} epsilon={RL_EPSILON}"
    )


def get_matchup_profile(enemy_type: str) -> Optional[MatchupProfile]:
    """Get matchup-specific profile for enemy type."""
    for category, profiles in MATCHUP_PROFILES.items():
        if enemy_type in profiles:
            return profiles[enemy_type]
    return None


def validate_config() -> list[str]:
    warnings = []
    if not os.path.exists(LOCAL_MODEL_ONNX_PATH):
        warnings.append(f"Model not found at {LOCAL_MODEL_ONNX_PATH}")
    if ACTION_COOLDOWN_MS < 50:
        warnings.append("Very low cooldown may cause input conflicts")
    if AGGRESSION < 0 or AGGRESSION > 1:
        warnings.append(f"Aggression {AGGRESSION} out of [0,1] range")
    return warnings


# Run validation on import
_config_warnings = validate_config()
for w in _config_warnings:
    logger.warning(f"[CONFIG] {w}")