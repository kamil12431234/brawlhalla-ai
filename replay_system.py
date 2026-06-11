__all__ = [
    "ActionRecord",
    "DetectionRecord",
    "MetricsSummary",
    "ReplayData",
    "TrainingSample",
    "AutoLabeler",
    "TrainingExporter",
    "ReplayRecorder",
    "ReplayAnalyzer",
    "main",
]
#!/usr/bin/env python3
"""
Brawlhalla AI — Replay System with auto-labeling and training data export.

Records match data for later analysis, auto-labels actions based on outcomes,
and exports training data for vision model improvement.
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, TypedDict
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────

@dataclass
class ActionRecord:
    """Single action record with outcome."""
    frame: int
    timestamp: float
    action: str
    keys: list[str]
    hit_result: str = ""  # hit, miss, blocked, none
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    combo_count: int = 0
    labeled: bool = False


@dataclass
class DetectionRecord:
    """Detection state at a frame."""
    frame: int
    timestamp: float
    player: Optional[dict] = None
    enemies: list[dict] = field(default_factory=list)
    gadgets: list[dict] = field(default_factory=list)


@dataclass
class MetricsSummary:
    """Aggregated metrics for a match."""
    total_frames: int = 0
    actions_taken: int = 0
    hits_landed: int = 0
    misses: int = 0
    blocks: int = 0
    combos_completed: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    edge_guards: int = 0
    recoveries: int = 0
    ledge_options_used: int = 0
    di_escapes: int = 0
    hitstun_escapes: int = 0


@dataclass
class ReplayData:
    """Complete replay data structure."""
    match_id: str
    timestamp: str
    character: str
    mode: str
    outcome: str = ""
    actions: list[ActionRecord] = field(default_factory=list)
    detections: list[DetectionRecord] = field(default_factory=list)
    metrics: Optional[MetricsSummary] = None


@dataclass
class TrainingSample:
    """Training sample for vision model improvement."""
    frame: int
    image_path: str
    detections: list[dict]
    labels: list[dict]  # auto-labeled based on action outcomes
    quality_score: float  # 0-1 confidence in labels


# ── Auto-labeling engine ─────────────────────────────────────────

class AutoLabeler:
    """Automatically labels actions based on outcomes and game state."""
    
    def __init__(self):
        self._label_cache: deque[tuple[int, str, str]] = deque(maxlen=100)  # (frame, action, label)
        self._hit_sequences: deque[int] = deque(maxlen=20)  # frames where hits occurred
        self._miss_sequences: deque[int] = deque(maxlen=20)  # frames where misses occurred
    
    def label_action(self, action_record: ActionRecord, game_state: dict,
                    player: Optional[dict], enemies: list[dict]) -> str:
        """
        Label an action based on context and outcome.
        
        Returns:
            Label string: "hit_confirm", "whiff_punish", "block", "miss", "neutral", "recovery", "edge_guard"
        """
        # Check recent frames for hit confirmation
        hit_confirm = self._check_hit_confirm(action_record.frame, enemies)
        
        # Analyze action type
        action_type = self._categorize_action(action_record.action)
        
        # Check if in combo
        in_combo = self._is_in_combo(action_record.frame)
        
        # Check position (edge guard?)
        at_edge = self._is_at_edge(player)
        
        # Check blast zone danger (recovery?)
        in_danger = self._is_in_danger(player)
        
        # Assign label based on combination
        if in_danger and action_type in ("movement", "jump"):
            label = "recovery"
        elif at_edge and action_type == "attack":
            label = "edge_guard"
        elif hit_confirm and action_type == "attack":
            label = "hit_confirm" if in_combo else "combo_start"
        elif not hit_confirm and action_type == "attack":
            label = "whiff" if self._was_whiff(action_record.frame) else "neutral"
        elif self._was_blocked(action_record.frame):
            label = "blocked"
        else:
            label = "neutral"
        
        self._label_cache.append((action_record.frame, action_record.action, label))
        return label
    
    def _check_hit_confirm(self, frame: int, enemies: list[dict]) -> bool:
        """Check if a hit was confirmed this frame."""
        # Look for rapid distance change toward enemy (hit confirm)
        for e in enemies:
            if e.get("dist_to_player", 999) < 0.05:
                return True
        return False
    
    def _categorize_action(self, action: str) -> str:
        if "attack" in action or "light" in action or "heavy" in action or "special" in action:
            return "attack"
        if "jump" in action:
            return "jump"
        if "move" in action or "strafe" in action:
            return "movement"
        if "shield" in action:
            return "defense"
        if "dash" in action:
            return "dash"
        return "other"
    
    def _is_in_combo(self, frame: int) -> bool:
        """Check if current frame is part of a combo."""
        # Combo = multiple attacks within ~20 frames
        recent_hits = [f for f in self._hit_sequences if frame - 30 < f < frame]
        return len(recent_hits) >= 2
    
    def _is_at_edge(self, player: Optional[dict]) -> bool:
        if not player:
            return False
        cx = player.get("cx", 0.5)
        return cx < 0.06 or cx > 0.94
    
    def _is_in_danger(self, player: Optional[dict]) -> bool:
        if not player:
            return False
        bz = player.get("blast_zone", {})
        return bz.get("in_danger", False) or bz.get("danger_level", 0) > 0.5
    
    def _was_whiff(self, frame: int) -> bool:
        """Check if this was a whiffed attack (attack but no hit)."""
        recent = [f for f in self._miss_sequences if frame - 10 < f < frame]
        return len(recent) > 0
    
    def _was_blocked(self, frame: int) -> bool:
        """Check if attack was blocked."""
        # For now, approximate blocked = enemy very close but no hit
        return False  # Would need actual blocked detection
    
    def record_hit(self, frame: int):
        self._hit_sequences.append(frame)
    
    def record_miss(self, frame: int):
        self._miss_sequences.append(frame)
    
    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of labels for analysis."""
        counts = {}
        for _, _, label in self._label_cache:
            counts[label] = counts.get(label, 0) + 1
        return counts


# ── Training data export ─────────────────────────────────────────

class TrainingExporter:
    """Exports replay data as training samples for vision model."""
    
    def __init__(self, output_dir: str = "./data/training"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_count: int = 0
    
    def add_sample(self, sample: TrainingSample) -> str:
        """Add a training sample and return its path."""
        filename = f"sample_{sample.frame:06d}.json"
        filepath = self._output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump({
                "frame": sample.frame,
                "image_path": sample.image_path,
                "detections": sample.detections,
                "labels": sample.labels,
                "quality_score": sample.quality_score,
            }, f, indent=2)
        
        self._sample_count += 1
        return str(filepath)
    
    def export_dataset(self, replay: ReplayData, output_file: Optional[str] = None) -> str:
        """Export complete dataset as YOLO-compatible JSON."""
        if output_file is None:
            output_file = self._output_dir / f"dataset_{replay.match_id}.json"
        
        dataset = {
            "match_id": replay.match_id,
            "timestamp": replay.timestamp,
            "character": replay.character,
            "samples": [],
        }
        
        for action in replay.actions:
            if action.labeled:
                sample = {
                    "frame": action.frame,
                    "action": action.action,
                    "hit_result": action.hit_result,
                    "label": self._infer_label(action),
                }
                dataset["samples"].append(sample)
        
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        return str(output_file)
    
    def _infer_label(self, action: ActionRecord) -> str:
        """Infer label from action record."""
        if action.hit_result == "hit":
            return "hit_confirm"
        if action.hit_result == "miss":
            return "whiff"
        if action.combo_count > 0:
            return "combo_continuation"
        return "neutral"


# ── Replay recorder ──────────────────────────────────────────────

class ReplayRecorder:
    """Records match data for later analysis."""
    
    def __init__(self, match_id: Optional[str] = None, character: str = "balanced",
                 mode: str = "full", capture_images: bool = False):
        self._match_id = match_id or self._generate_match_id()
        self._character = character
        self._mode = mode
        self._capture_images = capture_images
        
        self._actions: deque[ActionRecord] = deque(maxlen=1000)
        self._detections: deque[DetectionRecord] = deque(maxlen=500)
        
        self._recording: bool = False
        self._start_time: float = 0.0
        self._frame_count: int = 0
        
        self._labeler = AutoLabeler()
        self._exporter = TrainingExporter()
        
        self._combo_count: int = 0
        self._last_action_was_hit: bool = False
    
    def _generate_match_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def start(self):
        self._recording = True
        self._start_time = datetime.now().isoformat()
        self._frame_count = 0
        logger.info("[REPLAY] Recording started: %s", self._match_id)
    
    def stop(self) -> ReplayData:
        self._recording = False
        replay = ReplayData(
            match_id=self._match_id,
            timestamp=self._start_time,
            character=self._character,
            mode=self._mode,
            actions=list(self._actions),
            detections=list(self._detections),
            metrics=self._compute_metrics(),
        )
        logger.info("[REPLAY] Recording stopped: %s | %d actions", self._match_id, len(self._actions))
        return replay
    
    def record_action(self, action: str, keys: list[str],
                     game_state: dict, player: Optional[dict],
                     enemies: list[dict], gadgets: list[dict],
                     hit_result: str = "none", damage_dealt: float = 0.0):
        """Record an action with automatic labeling."""
        if not self._recording:
            return
        
        self._frame_count += 1
        record = ActionRecord(
            frame=self._frame_count,
            timestamp=datetime.now().timestamp(),
            action=action,
            keys=keys,
            hit_result=hit_result,
            damage_dealt=damage_dealt,
        )
        
        # Auto-label
        label = self._labeler.label_action(record, game_state, player, enemies)
        record.hit_result = hit_result
        record.combo_count = self._combo_count
        record.labeled = True
        
        # Track combo state
        if hit_result == "hit" and not self._last_action_was_hit:
            self._combo_count += 1
            self._labeler.record_hit(self._frame_count)
        elif hit_result == "miss":
            self._combo_count = 0
            self._labeler.record_miss(self._frame_count)
        
        self._last_action_was_hit = (hit_result == "hit")
        
        self._actions.append(record)
        
        # Record detection state
        det_record = DetectionRecord(
            frame=self._frame_count,
            timestamp=datetime.now().timestamp(),
            player=dict(player) if player else None,
            enemies=[dict(e) for e in enemies],
            gadgets=[dict(g) for g in gadgets],
        )
        self._detections.append(det_record)
        
        # Export training sample if high quality
        if self._should_export_sample(record, enemies):
            self._export_training_sample(record, enemies)
    
    def _should_export_sample(self, action: ActionRecord, enemies: list[dict]) -> bool:
        """Decide if this sample is worth exporting for training."""
        if not enemies:
            return False
        
        # Only export attack actions with clear outcomes
        if "attack" in action.action or "light" in action.action or "heavy" in action.action:
            return True
        
        return False
    
    def _export_training_sample(self, action: ActionRecord, enemies: list[dict]):
        """Export a training sample."""
        sample = TrainingSample(
            frame=action.frame,
            image_path=f"replays/{self._match_id}/frame_{action.frame:06d}.png",
            detections=[{"label": "enemy", "x": e.get("x", 0), "y": e.get("y", 0),
                        "w": e.get("w", 0), "h": e.get("h", 0)} for e in enemies],
            labels=[{"action": action.action, "hit_result": action.hit_result}],
            quality_score=0.8 if action.hit_result in ("hit", "miss") else 0.5,
        )
        self._exporter.add_sample(sample)
    
    def _compute_metrics(self) -> MetricsSummary:
        """Compute aggregate metrics from recorded actions."""
        m = MetricsSummary()
        m.total_frames = self._frame_count
        m.actions_taken = len(self._actions)
        
        for a in self._actions:
            if a.hit_result == "hit":
                m.hits_landed += 1
                m.damage_dealt += a.damage_dealt
            elif a.hit_result == "miss":
                m.misses += 1
            elif a.hit_result == "blocked":
                m.blocks += 1
        
        m.combos_completed = max(0, self._combo_count)
        
        return m
    
    def get_replay(self) -> ReplayData:
        return ReplayData(
            match_id=self._match_id,
            timestamp=self._start_time,
            character=self._character,
            mode=self._mode,
            actions=list(self._actions),
            detections=list(self._detections),
            metrics=self._compute_metrics(),
        )
    
    def save(self, path: str):
        replay = self.get_replay()
        with open(path, "w") as f:
            json.dump({
                "match_id": replay.match_id,
                "timestamp": replay.timestamp,
                "character": replay.character,
                "mode": replay.mode,
                "actions": [
                    {"frame": a.frame, "action": a.action, "keys": a.keys,
                     "hit_result": a.hit_result, "damage_dealt": a.damage_dealt}
                    for a in replay.actions
                ],
                "metrics": replay.metrics.__dict__ if replay.metrics else None,
            }, f, indent=2)
        logger.info("[REPLAY] Saved to: %s", path)
    
    def is_recording(self) -> bool:
        return self._recording


# ── Replay analyzer ──────────────────────────────────────────────

class ReplayAnalyzer:
    """Analyzes saved replays to extract insights."""
    
    def __init__(self):
        self._loaded_replays: dict[str, ReplayData] = {}
    
    def load_replay(self, path: str) -> Optional[ReplayData]:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            replay = ReplayData(
                match_id=data.get("match_id", "unknown"),
                timestamp=data.get("timestamp", ""),
                character=data.get("character", "unknown"),
                mode=data.get("mode", "unknown"),
                actions=[ActionRecord(
                    frame=a["frame"], timestamp=0, action=a["action"],
                    keys=a.get("keys", []), hit_result=a.get("hit_result", "none"),
                    damage_dealt=a.get("damage_dealt", 0)
                ) for a in data.get("actions", [])],
            )
            
            if data.get("metrics"):
                replay.metrics = MetricsSummary(**data["metrics"])
            
            self._loaded_replays[replay.match_id] = replay
            return replay
        except Exception as e:
            logger.error("[ANALYZER] Failed to load replay: %s", e)
            return None
    
    def analyze(self, replay: ReplayData) -> dict:
        """Generate analysis report for a replay."""
        if not replay.metrics:
            return {}
        
        m = replay.metrics
        
        hit_rate = m.hits_landed / max(1, m.hits_landed + m.misses)
        
        return {
            "match_id": replay.match_id,
            "character": replay.character,
            "outcome": replay.outcome,
            "total_frames": m.total_frames,
            "actions_taken": m.actions_taken,
            "hit_rate": f"{hit_rate:.1%}",
            "damage_dealt": m.damage_dealt,
            "damage_taken": m.damage_taken,
            "avg_action_frequency": m.actions_taken / max(1, m.total_frames / 60),
            "combo_efficiency": m.combos_completed / max(1, m.hits_landed),
        }
    
    def compare_replays(self, replay_a: str, replay_b: str) -> dict:
        """Compare two replays."""
        a = self._loaded_replays.get(replay_a)
        b = self._loaded_replays.get(replay_b)
        
        if not a or not b:
            return {"error": "One or both replays not loaded"}
        
        return {
            "hit_rate_diff": (a.metrics.hits_landed / max(1, a.metrics.hits_landed + a.metrics.misses)) -
                            (b.metrics.hits_landed / max(1, b.metrics.hits_landed + b.metrics.misses)),
            "damage_diff": a.metrics.damage_dealt - b.metrics.damage_dealt,
            "aggression_diff": (a.metrics.actions_taken / max(1, a.metrics.total_frames / 60)) -
                              (b.metrics.actions_taken / max(1, b.metrics.total_frames / 60)),
        }


def main():
    parser = argparse.ArgumentParser(description="Brawlhalla AI Replay System")
    parser.add_argument("--mode", choices=["record", "analyze", "export"], default="record",
                       help="Mode: record, analyze, or export")
    parser.add_argument("--character", default="balanced", help="Character for recording")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--input", default=None, help="Input replay to analyze")
    
    args = parser.parse_args()
    
    if args.mode == "record":
        recorder = ReplayRecorder(character=args.character)
        recorder.start()
        
        try:
            while True:
                # Placeholder — integrate with game loop
                pass
        except KeyboardInterrupt:
            replay = recorder.stop()
            if args.output:
                recorder.save(args.output)
    
    elif args.mode == "analyze":
        if not args.input:
            logger.error("--input required for analyze mode")
            return 1
        
        analyzer = ReplayAnalyzer()
        replay = analyzer.load_replay(args.input)
        if replay:
            report = analyzer.analyze(replay)
            print(json.dumps(report, indent=2))
    
    elif args.mode == "export":
        if not args.input or not args.output:
            logger.error("--input and --output required for export mode")
            return 1
        
        analyzer = ReplayAnalyzer()
        replay = analyzer.load_replay(args.input)
        if replay:
            exporter = TrainingExporter()
            path = exporter.export_dataset(replay, args.output)
            print(f"Exported to: {path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())