#!/usr/bin/env python3
"""Input controller — non-blocking async key execution with key buffering and emergency rollback."""

import subprocess
import threading
import time
import logging
from typing import Optional
from dataclasses import dataclass

from config import ACTION_COOLDOWN_MS

logger = logging.getLogger(__name__)

# ── Key buffering ──
_MAX_BUFFER_SIZE = 8
_BUFFER_FLUSH_INTERVAL_MS = 60


@dataclass
class BufferedKeyEvent:
    """A buffered key event with timing information."""
    keys: list[str]
    duration_ms: int
    timestamp: float
    priority: int = 0


@dataclass
class RollbackEvent:
    """An event that can be rolled back."""
    keys: list[str]
    timestamp: float
    applied: bool = False


class KeyBuffer:
    """Ring buffer for key events to ensure no inputs are dropped."""
    
    def __init__(self, max_size: int = _MAX_BUFFER_SIZE):
        self._buffer: list[BufferedKeyEvent] = []
        self._max_size = max_size
    
    def push(self, event: BufferedKeyEvent):
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(event)
    
    def pop(self) -> Optional[BufferedKeyEvent]:
        if self._buffer:
            return self._buffer.pop(0)
        return None
    
    def clear(self):
        self._buffer.clear()
    
    def __len__(self):
        return len(self._buffer)


class RollbackManager:
    """Manages rollback capability for key presses."""
    
    def __init__(self):
        self._events: list[RollbackEvent] = []
        self._rollback_count: int = 0
    
    def add_event(self, event: RollbackEvent):
        self._events.append(event)
        if len(self._events) > 100:
            self._events.pop(0)
    
    def rollback_last(self):
        if self._events:
            event = self._events.pop()
            if event.applied:
                for key in event.keys:
                    try:
                        subprocess.run(
                            ["xdotool", "keyup", key],
                            capture_output=True, timeout=0.5
                        )
                    except Exception:
                        pass
                self._rollback_count += 1
    
    def get_rollback_count(self) -> int:
        return self._rollback_count


# Global state
_lock = threading.Lock()
_active: set[str] = set()
_active_keys: dict[str, float] = {}
_pressed_keys: set[str] = set()

_key_buffer = KeyBuffer()
_rollback_mgr = RollbackManager()
_emergency_stop: bool = False


def _hold_keys_batch_async(keys: list[str], duration_ms: int, callback=None):
    """Press keys asynchronously with proper timing."""
    def _execute():
        global _active, _active_keys, _pressed_keys
        
        # Filter and validate keys
        valid_keys = [str(k) for k in keys if k and str(k).strip()]
        if not valid_keys:
            if callback:
                callback(False)
            return
        
        success = True
        pressed = []
        
        try:
            for key in valid_keys:
                with _lock:
                    _active.add(key)
                    _active_keys[key] = time.time()
                    _pressed_keys.add(key)
                
                result = subprocess.run(
                    ["xdotool", "keydown", key],
                    capture_output=True, text=True, timeout=0.5
                )
                if result.returncode != 0:
                    logger.warning("[INPUT] keydown failed: %s %s", key, result.stderr)
                    success = False
                else:
                    pressed.append(key)
            
            time.sleep(max(duration_ms, 30) / 1000.0)
            
            for key in pressed:
                with _lock:
                    _active.discard(key)
                    _active_keys.pop(key, None)
                    _pressed_keys.discard(key)
                
                subprocess.run(
                    ["xdotool", "keyup", key],
                    capture_output=True, timeout=0.5
                )
        
        except subprocess.TimeoutExpired:
            logger.warning("[INPUT] xdotool timeout on keys: %s", valid_keys)
            success = False
        except Exception as e:
            logger.warning("[INPUT] key press error: %s", e)
            success = False
        finally:
            with _lock:
                for key in pressed:
                    _active.discard(key)
                    _active_keys.pop(key, None)
                    _pressed_keys.discard(key)
            if callback:
                callback(success)
    
    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    return thread


def _release_stuck_keys(keys: list[str]):
    """Emergency key release if xdotool times out."""
    for key in keys:
        try:
            subprocess.run(
                ["xdotool", "keyup", key],
                capture_output=True, timeout=1.0
            )
            with _lock:
                _active.discard(key)
                _pressed_keys.discard(key)
        except Exception:
            pass


def _hold_keys_batch_sync(keys: list[str], duration_ms: int) -> bool:
    """Press keys synchronously (blocking)."""
    valid_keys = [str(k) for k in keys if k and str(k).strip()]
    if not valid_keys:
        return False
    
    success = True
    pressed = []
    
    try:
        for key in valid_keys:
            with _lock:
                _active.add(key)
                _active_keys[key] = time.time()
                _pressed_keys.add(key)
            
            result = subprocess.run(
                ["xdotool", "keydown", key],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode != 0:
                success = False
            else:
                pressed.append(key)
        
        time.sleep(max(duration_ms, 30) / 1000.0)
        
        for key in pressed:
            with _lock:
                _active.discard(key)
                _active_keys.pop(key, None)
                _pressed_keys.discard(key)
            
            subprocess.run(
                ["xdotool", "keyup", key],
                capture_output=True, timeout=0.5
            )
    
    except Exception as e:
        logger.warning("[INPUT] sync key press error: %s", e)
        success = False
    
    return success


def press_keys(keys: list[str], duration_ms: int = 80, async_mode: bool = True):
    """Press keys either asynchronously (default) or synchronously."""
    if _emergency_stop:
        return
    
    if async_mode:
        _hold_keys_batch_async(keys, duration_ms)
    else:
        _hold_keys_batch_sync(keys, duration_ms)


def release_all():
    """Release all currently active keys (emergency stop)."""
    global _emergency_stop, _active, _pressed_keys, _active_keys
    
    try:
        with _lock:
            keys_to_release = list(_pressed_keys)
            _active.clear()
            _active_keys.clear()
            _pressed_keys.clear()
        
        for key in keys_to_release:
            try:
                subprocess.run(
                    ["xdotool", "keyup", key],
                    capture_output=True, timeout=0.5
                )
            except Exception:
                pass
        
        _emergency_stop = False
    except Exception as e:
        logger.warning("release_all failed: %s", e)


def reset_emergency():
    """Reset emergency stop flag."""
    global _emergency_stop
    _emergency_stop = False


def get_active_keys() -> set[str]:
    with _lock:
        return set(_active)


class InputController:
    """Manages AI input with key buffering and emergency rollback."""
    
    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._last_flush = time.time()
        self._last_keys: list[str] = []
        self._cooldown_until: float = 0.0
    
    def apply_keys(self, keys: list[tuple[str, int]]):
        """Apply keys with cooldown management."""
        if self._dry_run:
            return
        
        if time.time() < self._cooldown_until:
            return
        
        key_names = [k for k, _ in keys]
        
        if not key_names:
            return
        
        press_keys(key_names, ACTION_COOLDOWN_MS)
        self._last_keys = key_names
        self._cooldown_until = time.time() + (ACTION_COOLDOWN_MS / 1000.0)
    
    def flush_buffer(self):
        """Flush pending buffered inputs."""
        while len(_key_buffer) > 0:
            event = _key_buffer.pop()
            if event:
                self.apply_keys([(k, event.duration_ms) for k in event.keys])
    
    def emergency_stop(self):
        """Emergency stop - release all keys."""
        global _emergency_stop
        _emergency_stop = True
        release_all()
    
    def get_stats(self) -> dict:
        """Return input controller statistics."""
        return {
            "active_keys": len(_active),
            "buffer_size": len(_key_buffer),
            "rollbacks": _rollback_mgr.get_rollback_count(),
            "emergency": _emergency_stop,
        }