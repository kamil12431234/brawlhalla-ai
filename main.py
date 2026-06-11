#!/usr/bin/env python3
"""
Brawlhalla AI (external vision-based) — CLI entry point.

Modes:
  - Full auto:      python main.py
  - Assist mode:    python main.py --assist
  - Dry-run test:   python main.py --dry-run

Flow:
  capture_frame -> local_infer(brawlhalla-vision ONNX) -> build_world_state
    -> ai.choose_action(state) -> resolve_action -> input_controller.apply_actions
"""

import argparse
import logging
from config import CHARACTER, COMBO_STYLE, CAPTURE_WIDTH, CAPTURE_HEIGHT, AGGRESSION
from ai_runner import AIRunner
from input_controller import release_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(dry_run=False, assist_mode=False, combat_mode=False, poke_mode=False, aggressive=None):
    if aggressive is None:
        aggressive = AGGRESSION
    if poke_mode:
        mode = "poke"
    elif combat_mode:
        mode = "combat"
    elif assist_mode:
        mode = "assist"
    else:
        mode = "full"

    runner = AIRunner(mode=mode, aggressive=aggressive, dry_run=dry_run)

    def _stats(avg_fps, dt_ms, enemies, player_ok, ext_stats=None):
        log_msg = (
            f"[STATS] fps={avg_fps:.1f} dt={dt_ms:.0f}ms "
            f"enemies={enemies} player={player_ok}"
        )
        if ext_stats:
            log_msg += (
                f" | inf={ext_stats.get('inferences', 0)} "
                f"trackers={ext_stats.get('active_tracks', 0)}"
            )
        logger.info(log_msg)

    logger.info(f"[INFO] Character: {CHARACTER} | Style: {COMBO_STYLE} | Aggression: {aggressive:.2f}")
    logger.info(f"[INFO] Window: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} | Press Ctrl+C to stop.")

    try:
        runner.run_loop(stats_cb=_stats)
    finally:
        release_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brawlhalla vision-based AI")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions instead of sending keys")
    parser.add_argument("--assist", action="store_true", help="Assist mode: you control movement; AI assists")
    parser.add_argument("--combat", action="store_true", help="Combat mode: attacks/shields only, you move")
    parser.add_argument("--poke", action="store_true", help="Poke mode: light attacks only, in-range only, you move")
    parser.add_argument("--aggressive", type=float, default=None, help="Aggression 0..1 (default: character profile)")
    parser.add_argument("--character", type=str, default=None, help="Override character (e.g. bodvar, orion, ember)")
    args = parser.parse_args()

    if args.character:
        import os
        os.environ["BH_CHARACTER"] = args.character

    main(dry_run=args.dry_run, assist_mode=args.assist, combat_mode=args.combat, poke_mode=args.poke, aggressive=args.aggressive)