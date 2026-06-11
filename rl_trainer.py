#!/usr/bin/env python3
"""
Brawlhalla AI — Reinforcement Learning Trainer.

Implements a DQN-based RL agent that learns combat strategy from replay data.
Uses experience replay, target networks, and adaptive exploration.

Supports training modes:
  - From scratch with random initialization
  - From existing replay data (behavioral cloning)
  - Online learning during gameplay
"""

import os
import sys
import json
import logging
import random
import argparse
from typing import Optional
from dataclasses import dataclass, field
from collections import deque, namedtuple
from datetime import datetime
import math

import numpy as np

logger = logging.getLogger(__name__)

# ── RL Configuration ─────────────────────────────────────────────

GAMMA = 0.95           # Discount factor
EPSILON_START = 1.0    # Initial exploration rate
EPSILON_MIN = 0.05     # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay per episode
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # episodes
PRINT_FREQ = 10


# ── Experience tuple ─────────────────────────────────────────────

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


@dataclass
class StateVector:
    """Flattened state representation for RL input."""
    # Normalized distances and positions (10 features)
    enemy_dist: float = 0.5
    enemy_cx: float = 0.5
    enemy_cy: float = 0.5
    player_cx: float = 0.5
    player_cy: float = 0.5
    
    # Velocity features (4 features)
    player_vx: float = 0.0
    player_vy: float = 0.0
    enemy_vx: float = 0.0
    enemy_vy: float = 0.0
    
    # State flags (6 features)
    player_airborne: float = 0.0
    near_edge: float = 0.0
    in_combo: float = 0.0
    enemy_attacking: float = 0.0
    blast_zone_danger: float = 0.0
    weapon_nearby: float = 0.0
    
    # History (last 3 frames = 6 features)
    prev_dist: float = 0.5
    prev_enemy_vx: float = 0.0
    prev_player_vx: float = 0.0
    dist_change: float = 0.0
    combo_progress: float = 0.0
    hit_streak: float = 0.0
    
    @property
    def size(self) -> int:
        return 26  # Total features
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.enemy_dist, self.enemy_cx, self.enemy_cy, self.player_cx, self.player_cy,
            self.player_vx, self.player_vy, self.enemy_vx, self.enemy_vy,
            self.player_airborne, self.near_edge, self.in_combo, self.enemy_attacking,
            self.blast_zone_danger, self.weapon_nearby,
            self.prev_dist, self.prev_enemy_vx, self.prev_player_vx,
            self.dist_change, self.combo_progress, self.hit_streak,
        ], dtype=np.float32)
    
    @staticmethod
    def from_game_state(state: dict, prev_state: Optional[dict] = None) -> "StateVector":
        """Build state vector from game state dict."""
        sv = StateVector()
        
        player = state.get("player")
        enemies = state.get("enemies", [])
        gadgets = state.get("gadgets", [])
        
        if player:
            sv.player_cx = player.get("cx", 0.5)
            sv.player_cy = player.get("cy", 0.5)
            sv.player_vx = player.get("vx", 0.0)
            sv.player_vy = player.get("vy", 0.0)
            sv.player_airborne = 1.0 if abs(sv.player_vy) > 0.005 else 0.0
            
            # Edge awareness
            sv.near_edge = 1.0 if (sv.player_cx < 0.07 or sv.player_cx > 0.93) else 0.0
            
            # Blast zone danger
            bz = player.get("blast_zone", {})
            sv.blast_zone_danger = bz.get("danger_level", 0.0)
        
        if enemies:
            e = enemies[0]
            sv.enemy_cx = e.get("cx", 0.5)
            sv.enemy_cy = e.get("cy", 0.5)
            sv.enemy_vx = e.get("vx", 0.0)
            sv.enemy_vy = e.get("vy", 0.0)
            
            if player:
                sv.enemy_dist = math.hypot(sv.enemy_cx - sv.player_cx, sv.enemy_cy - sv.player_cy)
        
        # Weapon awareness
        if gadgets:
            for g in gadgets:
                if player:
                    gx = g.get("cx", 0.5)
                    gy = g.get("cy", 0.5)
                    if math.hypot(gx - sv.player_cx, gy - sv.player_cy) < 0.15:
                        sv.weapon_nearby = 1.0
                        break
        
        # History features
        if prev_state:
            prev_enemies = prev_state.get("enemies", [])
            prev_player = prev_state.get("player")
            if prev_enemies and prev_player and player:
                prev_e = prev_enemies[0]
                prev_dist = math.hypot(prev_e.get("cx", 0.5) - prev_player.get("cx", 0.5),
                                       prev_e.get("cy", 0.5) - prev_player.get("cy", 0.5))
                sv.prev_dist = prev_dist
                sv.dist_change = sv.enemy_dist - prev_dist
                sv.prev_enemy_vx = prev_e.get("vx", 0.0)
                sv.prev_player_vx = prev_player.get("vx", 0.0)
        
        return sv


# ── Action space ─────────────────────────────────────────────────

ACTIONS = [
    "idle",
    "move_left", "move_right",
    "jump", "jump+move_left", "jump+move_right",
    "light_attack",
    "heavy_attack",
    "special",
    "shield_back",
    "dash_left", "dash_right",
]

ACTION_SIZE = len(ACTIONS)


# ── Simple Q-Network (PyTorch-free pure numpy for portability) ──

class QNetwork:
    """Simple Q-network using numpy (no PyTorch dependency)."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        scale_w1 = math.sqrt(2.0 / (state_size + hidden_size))
        scale_w2 = math.sqrt(2.0 / (hidden_size + hidden_size))
        scale_w3 = math.sqrt(2.0 / (hidden_size + action_size))
        
        self.W1 = np.random.randn(state_size, hidden_size) * scale_w1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale_w2
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, action_size) * scale_w3
        self.b3 = np.zeros(action_size)
        
        # Target network (copy of main network)
        self._copy_to_target()
    
    def _copy_to_target(self):
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        self.W3_t = self.W3.copy()
        self.b3_t = self.b3.copy()
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass — returns Q-values for all actions."""
        # Layer 1
        z1 = state @ self.W1 + self.b1
        a1 = self._relu(z1)
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        
        # Layer 3 (output)
        q = a2 @ self.W3 + self.b3
        
        return q
    
    def predict_target(self, state: np.ndarray) -> np.ndarray:
        """Forward pass using target network."""
        z1 = state @ self.W1_t + self.b1_t
        a1 = self._relu(z1)
        z2 = a1 @ self.W2_t + self.b2_t
        a2 = self._relu(z2)
        q = a2 @ self.W3_t + self.b3_t
        return q
    
    def update(self, states: np.ndarray, targets: np.ndarray,
               actions: np.ndarray, learning_rate: float = 0.001) -> float:
        """
        Single gradient step using SGD.
        
        Returns:
            Loss value for monitoring.
        """
        # Forward pass
        z1 = states @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        q = a2 @ self.W3 + self.b3
        
        # Compute loss (MSE on selected actions)
        batch_size = states.shape[0]
        q_selected = np.take_along_axis(q, actions.reshape(-1, 1), axis=1).flatten()
        loss = np.mean((q_selected - targets) ** 2)
        
        # Gradient descent (simplified — no autograd)
        # For production, use PyTorch. This is a simplified version.
        for i in range(batch_size):
            s = states[i]
            a = actions[i]
            t = targets[i]
            
            # Simplified one-sample gradient update
            grad_q = np.zeros(self.action_size)
            grad_q[a] = 2 * (q_selected[i] - t) / batch_size
            
            # Backprop through network (simplified)
            delta3 = grad_q.reshape(1, -1)
            grad_W3 = a2[i].reshape(-1, 1) @ delta3
            grad_b3 = delta3.flatten()
            
            delta2 = (delta3 @ self.W3.T) * self._relu_grad(z2[i].reshape(1, -1))
            grad_W2 = a1[i].reshape(-1, 1) @ delta2
            grad_b2 = delta2.flatten()
            
            delta1 = (delta2 @ self.W2.T) * self._relu_grad(z1[i].reshape(1, -1))
            grad_W1 = s.reshape(-1, 1) @ delta1
            grad_b1 = delta1.flatten()
            
            # Apply gradients with learning rate
            self.W3 -= learning_rate * grad_W3
            self.b3 -= learning_rate * grad_b3
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1
        
        return loss
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        self.W3_t = self.W3.copy()
        self.b3_t = self.b3.copy()
    
    def load(self, path: str):
        """Load weights from file."""
        try:
            data = np.load(path, allow_pickle=True)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]
            self._copy_to_target()
            logger.info("[RL] Loaded weights from: %s", path)
        except Exception as e:
            logger.warning("[RL] Failed to load weights: %s", e)
    
    def save(self, path: str):
        """Save weights to file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        logger.info("[RL] Saved weights to: %s", path)


# ── Experience Replay Buffer ─────────────────────────────────────

class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self._rng = random.Random()
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of experiences."""
        batch = self._rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


# ── RL Agent ─────────────────────────────────────────────────────

class RLAgent:
    """DQN-based RL agent for Brawlhalla AI."""
    
    def __init__(self, state_size: int = 26, action_size: int = ACTION_SIZE,
                 learning_rate: float = LEARNING_RATE):
        self.state_size = state_size
        self.action_size = action_size
        
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network._copy_to_target()
        
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        
        self._steps: int = 0
        self._episodes: int = 0
        self._update_count: int = 0
        
        self._total_reward: float = 0.0
        self._episode_rewards: deque[float] = deque(maxlen=100)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_network.predict(state.reshape(1, -1))
        return int(np.argmax(q_values))
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = BATCH_SIZE) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute target Q-values using target network
        next_q = self.target_network.predict_target(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Update Q-network
        loss = self.q_network.update(states, targets, actions, LEARNING_RATE)
        
        self._update_count += 1
        
        # Periodically update target network
        if self._update_count % TARGET_UPDATE_FREQ == 0:
            self.target_network.update_target_network()
            logger.info("[RL] Target network updated (step %d)", self._update_count)
        
        return loss
    
    def end_episode(self, total_reward: float):
        """Called at end of episode — update epsilon, record stats."""
        self._episodes += 1
        self._episode_rewards.append(total_reward)
        
        # Decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
        if self._episodes % PRINT_FREQ == 0:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            logger.info(
                "[RL] Episode %d | Epsilon: %.3f | Avg reward (last 100): %.2f",
                self._episodes, self.epsilon, avg_reward
            )
    
    def get_action_name(self, action_idx: int) -> str:
        return ACTIONS[action_idx] if 0 <= action_idx < len(ACTIONS) else "unknown"
    
    def get_stats(self) -> dict:
        avg_reward = sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0.0
        return {
            "episodes": self._episodes,
            "epsilon": self.epsilon,
            "replay_size": len(self.replay_buffer),
            "updates": self._update_count,
            "avg_reward_100": avg_reward,
        }
    
    def save(self, path: str):
        self.q_network.save(path)
    
    def load(self, path: str):
        self.q_network.load(path)


# ── Reward function ─────────────────────────────────────────────

def compute_reward(action: str, state: dict, next_state: dict,
                   hit_result: str, prev_state: Optional[dict] = None) -> float:
    """
    Compute reward for an action based on state transitions.
    
    Positive rewards:
      - Hit confirmed: +10
      - Combo continuation: +3
      - Successful dodge/escape: +2
      - Maintaining optimal range: +1
      - Edge guard hit: +5
    
    Negative rewards:
      - Got hit: -5
      - Missed attack: -1
      - Walked into blast zone: -3
      - Threw away combo: -2
      - No progress: 0
    """
    reward = 0.0
    
    player = state.get("player")
    next_player = next_state.get("player")
    enemies = state.get("enemies", [])
    next_enemies = next_state.get("enemies", [])
    
    # Hit reward
    if hit_result == "hit":
        reward += 10.0
    elif hit_result == "miss":
        reward -= 1.0
    
    # Combo continuation
    if hit_result == "hit" and prev_state:
        if _was_in_combo(prev_state):
            reward += 3.0  # Bonus for continuing combo
    
    # Dodge/escape reward
    if "jump" in action or "dash" in action:
        if next_player and player:
            # Check if distance to enemy increased (successful escape)
            if next_enemies:
                next_dist = math.hypot(
                    next_enemies[0].get("cx", 0.5) - next_player.get("cx", 0.5),
                    next_enemies[0].get("cy", 0.5) - next_player.get("cy", 0.5),
                )
                if enemies:
                    cur_dist = math.hypot(
                        enemies[0].get("cx", 0.5) - player.get("cx", 0.5),
                        enemies[0].get("cy", 0.5) - player.get("cy", 0.5),
                    )
                    if next_dist > cur_dist + 0.02:
                        reward += 2.0
    
    # Blast zone danger avoidance
    if next_player:
        next_bz = next_player.get("blast_zone", {})
        if next_bz.get("danger_level", 0) < (player.get("blast_zone", {}).get("danger_level", 0) if player else 0):
            reward += 1.0
    
    # Edge guard
    if hit_result == "hit" and player:
        cx = player.get("cx", 0.5)
        if cx < 0.07 or cx > 0.93:
            reward += 5.0
    
    return reward


def _was_in_combo(state: dict) -> bool:
    """Check if player was in combo based on recent attack history."""
    # Simplified: check if hit streak > 0 (would need access to AI state)
    return False


# ── Training from replay data ────────────────────────────────────

class BehavioralCloner:
    """Train agent from existing replay data using behavioral cloning."""
    
    def __init__(self, agent: RLAgent):
        self.agent = agent
    
    def train_from_replay(self, replay_path: str, iterations: int = 1000) -> dict:
        """Train agent to mimic actions from replay data."""
        try:
            with open(replay_path, "r") as f:
                replay_data = json.load(f)
        except Exception as e:
            logger.error("[BC] Failed to load replay: %s", e)
            return {"error": str(e)}
        
        logger.info("[BC] Training from replay: %s (%d actions)", replay_path, len(replay_data.get("actions", [])))
        
        losses = []
        for i in range(iterations):
            for action_record in replay_data.get("actions", []):
                # Build state from detection record
                state = self._build_state(action_record)
                if state is None:
                    continue
                
                # Get action (label)
                action_name = action_record.get("action", "idle")
                try:
                    action_idx = ACTIONS.index(action_name)
                except ValueError:
                    action_idx = 0  # idle
                
                # For behavioral cloning, use the action directly as target
                # with positive reward for imitation
                state_arr = state.to_array()
                q_values = self.agent.q_network.predict(state_arr.reshape(1, -1))
                
                # Target: maximize Q for the demonstrated action
                target = q_values.copy()
                target[0, action_idx] = 10.0  # High reward for imitation
                
                loss = self.agent.q_network.update(state_arr.reshape(1, -1), target.flatten(), np.array([action_idx]))
                losses.append(loss)
            
            if i % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                logger.info("[BC] Iteration %d/%d | Avg loss: %.4f", i, iterations, avg_loss)
        
        return {"iterations": iterations, "avg_loss": sum(losses) / max(1, len(losses))}
    
    def _build_state(self, action_record: dict) -> Optional[StateVector]:
        """Build state vector from action record."""
        # In a full implementation, this would reconstruct state from detection history
        # For now, return a minimal state
        return StateVector()


# ── Main training loop ───────────────────────────────────────────

def train_online(agent: RLAgent, state: dict, action: str, next_state: dict,
                 hit_result: str, done: bool = False):
    """Online training step from current game state."""
    # Build state vectors
    sv = StateVector.from_game_state(state)
    sv_next = StateVector.from_game_state(next_state)
    
    # Get action index
    try:
        action_idx = ACTIONS.index(action)
    except ValueError:
        action_idx = 0
    
    # Compute reward
    reward = compute_reward(action, state, next_state, hit_result)
    
    # Store experience
    agent.store_experience(sv.to_array(), action_idx, reward, sv_next.to_array(), done)
    
    # Train
    loss = agent.train_step()


def main():
    parser = argparse.ArgumentParser(description="Brawlhalla AI RL Trainer")
    parser.add_argument("--mode", choices=["train", "bc", "eval"], default="train",
                       help="train=online RL, bc=behavioral cloning, eval=evaluate agent")
    parser.add_argument("--model", default="model/rl_agent.npz", help="Model save path")
    parser.add_argument("--replay", default=None, help="Replay file for behavioral cloning")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    agent = RLAgent()
    
    if args.mode == "train":
        logger.info("[RL] Starting online RL training for %d episodes", args.episodes)
        
        for episode in range(args.episodes):
            total_reward = 0.0
            
            # Simulate episode (in practice, integrate with game loop)
            # Placeholder: generate random experiences
            for step in range(200):
                state = StateVector().to_array()
                action = agent.select_action(state, training=True)
                next_state = StateVector().to_array()
                reward = random.uniform(-1, 1)
                done = step == 199
                
                agent.store_experience(state, action, reward, next_state, done)
                total_reward += reward
                
                agent.train_step()
            
            agent.end_episode(total_reward)
            
            if episode % 50 == 0:
                agent.save(args.model)
        
        agent.save(args.model)
        logger.info("[RL] Training complete. Model saved to: %s", args.model)
    
    elif args.mode == "bc":
        if not args.replay:
            logger.error("--replay required for behavioral cloning")
            return 1
        
        cloner = BehavioralCloner(agent)
        result = cloner.train_from_replay(args.replay, args.iterations)
        agent.save(args.model)
        logger.info("[BC] Training complete: %s", result)
    
    elif args.mode == "eval":
        agent.load(args.model)
        logger.info("[RL] Loaded model from: %s", args.model)
        
        stats = agent.get_stats()
        logger.info("[EVAL] Agent stats: %s", stats)
        
        # Test on random states
        for _ in range(10):
            state = StateVector().to_array()
            q_values = agent.q_network.predict(state.reshape(1, -1))
            action = agent.select_action(state, training=False)
            action_name = agent.get_action_name(action)
            logger.info("[EVAL] State shape: %s | Best action: %s (Q=%.2f)",
                       state.shape, action_name, q_values[0, action])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED TRAINING PIPELINE — PyTorch-based with Prioritized Replay
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import IterableDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[RL] PyTorch not available. Install with: pip install torch")
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay using SumTree for efficient sampling."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, priority_epsilon: float = 1e-6):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0=no priority, 1=full priority)
            beta: Importance sampling compensation (starts low, increases to 1)
            beta_increment: How much beta increases per sample
            priority_epsilon: Small constant to prevent zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priority_epsilon = priority_epsilon
        self.buffer: list = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._next_idx = 0
        self._max_priority = 1.0
        # SumTree for O(log n) operations
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self._tree = np.zeros(2 * tree_capacity - 1, dtype=np.float32)
        self._tree_size = tree_capacity
    def add(self, experience: Experience, priority: float = None):
        """Add experience with optional priority (default: max priority)."""
        if priority is None:
            priority = self._max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self._next_idx] = experience
        self.priorities[self._next_idx] = priority
        # Update tree
        self._update_tree(self._next_idx, priority)
        self._next_idx = (self._next_idx + 1) % self.capacity
    def _update_tree(self, idx: int, priority: float):
        """Update tree node and propagate to root."""
        tree_idx = idx + self._tree_size - 1
        change = priority - self.priorities[idx % self.capacity] if idx < len(self.buffer) else priority
        while tree_idx >= 0:
            self._tree[tree_idx] += change
            tree_idx = (tree_idx - 1) // 2
        self.priorities[idx % self.capacity] = priority
        self._max_priority = max(self._max_priority, priority)
    def _sample(self, batch_size: int) -> tuple:
        """Sample batch using stratified sampling based on priorities."""
        indices = []
        segment_size = self._tree[0] / batch_size if self._tree[0] > 0 else 1
        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)
            idx = self._find_index(value)
            indices.append(idx)
        # Compute importance sampling weights
        sampled_priorities = self.priorities[indices]
        weights = (len(self.buffer) * sampled_priorities / (self._tree[0] + 1e-8)) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        experiences = [self.buffer[i] for i in indices]
        # Increase beta for next sample
        self.beta = min(1.0, self.beta + self.beta_increment)
        return experiences, indices, weights.astype(np.float32)
    def _find_index(self, value: float) -> int:
        """Find index in tree using binary search."""
        idx = 0
        left = 1
        right = len(self._tree)
        while left < right:
            mid = (left + right) // 2
            if value <= self._tree[mid]:
                idx = mid
                right = mid
            else:
                left = mid + 1
        # Convert tree index to buffer index
        return idx - self._tree_size + 1
    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self._update_tree(idx, priority + self.priority_epsilon)
    def __len__(self) -> int:
        return len(self.buffer)
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity
if TORCH_AVAILABLE:
    class QNetworkTorch(nn.Module):
        """PyTorch-based Q-Network with GPU acceleration."""
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            super().__init__()
            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
            )
            # Actor head (policy)
            self.actor = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size),
            )
            # Critic head (value)
            self.critic = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
        def forward(self, x: torch.Tensor) -> tuple:
            """Forward pass returns (q_values, state_value)."""
            features = self.shared(x)
            q_values = self.actor(features)
            value = self.critic(features)
            return q_values, value
        def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
            """Get Q-values only (for inference)."""
            features = self.shared(x)
            return self.actor(features)
        def get_value(self, x: torch.Tensor) -> torch.Tensor:
            """Get state value only."""
            features = self.shared(x)
            return self.critic(features)
    class DQNAgentTorch:
        """DQN Agent with PyTorch backend, Double DQN, and Prioritized Replay."""
        def __init__(self, state_size: int, action_size: int,
                     lr: float = 0.001, gamma: float = 0.99,
                     epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                     epsilon_decay: float = 0.995, batch_size: int = 64,
                     memory_size: int = 50000, target_update_freq: int = 100,
                     device: str = None):
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self.update_count = 0
            # Device selection
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            logger.info("[DQN-Torch] Device: %s", self.device)
            # Networks
            self.q_network = QNetworkTorch(state_size, action_size).to(self.device)
            self.target_network = QNetworkTorch(state_size, action_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            # Loss function (Huber for stability)
            self.loss_fn = nn.HuberLoss(reduction='none')
            # Prioritized replay buffer
            self.memory = PrioritizedReplayBuffer(memory_size)
            # Exploration
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """Select action using epsilon-greedy policy."""
            if training and random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)
            state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            with torch.no_grad():
                q_values = self.q_network.get_q_values(state_t)
            return q_values.argmax().item()
        def store_experience(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool):
            """Store experience in prioritized replay buffer."""
            exp = Experience(state, action, reward, next_state, done)
            # Initial priority based on TD error magnitude
            self.memory.add(exp, priority=self._max_priority)
        @property
        def _max_priority(self) -> float:
            return self.memory._max_priority
        def train_step(self) -> float:
            """Train on a batch of experiences from prioritized replay."""
            if len(self.memory) < self.batch_size:
                return 0.0
            experiences, indices, weights = self.memory.sample(self.batch_size)
            # Unpack experiences
            states = np.array([e.state for e in experiences])
            actions = np.array([e.action for e in experiences])
            rewards = np.array([e.reward for e in experiences])
            next_states = np.array([e.next_state for e in experiences])
            dones = np.array([e.done for e in experiences])
            # Convert to tensors
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)
            weights_t = torch.FloatTensor(weights).to(self.device)
            # Double DQN: use online network to select action, target network to evaluate
            with torch.no_grad():
                # Online network selects best action
                next_q_online = self.q_network.get_q_values(next_states_t)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # Target network evaluates that action
                next_q_target = self.target_network.get_q_values(next_states_t)
                next_q_values = next_q_target.gather(1, next_actions).squeeze()
                # Compute target
                target_q = rewards_t + (1 - dones_t) * self.gamma * next_q_values
            # Current Q values
            current_q = self.q_network.get_q_values(states_t).gather(1, actions_t).squeeze()
            # Compute loss with importance sampling weights
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            loss = (self.loss_fn(current_q, target_q) * weights_t).mean()
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.update_count += 1
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors)
            # Update target network periodically
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return loss.item()
        def save(self, path: str):
            """Save model to path."""
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'update_count': self.update_count,
            }, path)
            logger.info("[DQN-Torch] Saved to: %s", path)
        def load(self, path: str):
            """Load model from path."""
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.update_count = checkpoint.get('update_count', 0)
            logger.info("[DQN-Torch] Loaded from: %s", path)
        def get_stats(self) -> dict:
            return {
                "epsilon": self.epsilon,
                "updates": self.update_count,
                "memory_size": len(self.memory),
                "device": str(self.device),
            }
else:
    # Stub when PyTorch not available
    class DQNAgentTorch:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
class EvaluationMetrics:
    """Track and compute evaluation metrics during training."""
    def __init__(self):
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.q_values_history: deque = deque(maxlen=1000)
        self.loss_history: deque = deque(maxlen=1000)
        self.success_rate: deque = deque(maxlen=100)
    def record_episode(self, reward: float, length: int, success: bool):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rate.append(1.0 if success else 0.0)
    def record_update(self, loss: float, avg_q: float):
        self.loss_history.append(loss)
        self.q_values_history.append(avg_q)
    def get_metrics(self) -> dict:
        return {
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "mean_loss": np.mean(self.loss_history) if self.loss_history else 0,
            "mean_q": np.mean(self.q_values_history) if self.q_values_history else 0,
            "success_rate": np.mean(self.success_rate) if self.success_rate else 0,
            "recent_reward": np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else 0,
        }
def evaluate_agent(agent, num_episodes: int = 20) -> dict:
    """Evaluate agent performance without exploration."""
    metrics = EvaluationMetrics()
    for episode in range(num_episodes):
        total_reward = 0.0
        steps = 0
        done = False
        # Generate test episode
        for step in range(200):
            state = np.random.randn(agent.state_size).astype(np.float32) * 0.1
            action = agent.select_action(state, training=False)
            # Simulate transition
            next_state = state + np.random.randn(agent.state_size) * 0.05
            reward = random.choice([0.0, 1.0, -1.0])
            total_reward += reward
            steps += 1
            if random.random() < 0.05:  # Simulated done
                done = True
                break
        metrics.record_episode(total_reward, steps, total_reward > 50)
    return metrics.get_metrics()
def train_advanced(episodes: int = 5000, save_path: str = "model/dqn_torch.pt",
                   device: str = None, eval_freq: int = 100) -> dict:
    """Advanced training with PyTorch, prioritized replay, and evaluation."""
    if not TORCH_AVAILABLE:
        logger.error("[TRAIN] PyTorch not available. Cannot use advanced training.")
        return {"error": "PyTorch required"}
    # Determine state/action dimensions from config
    try:
        from game_state import build_world_state
        from config import ACTION_DIM, STATE_DIM
        state_size = STATE_DIM
        action_size = ACTION_DIM
    except:
        state_size = 26
        action_size = 12
    agent = DQNAgentTorch(state_size, action_size, device=device)
    metrics = EvaluationMetrics()
    logger.info("[ADV-TRAIN] Starting advanced training for %d episodes", episodes)
    for episode in range(episodes):
        total_reward = 0.0
        steps = 0
        done = False
        # Collect experience
        for step in range(200):
            state = np.random.randn(state_size).astype(np.float32) * 0.1
            action = agent.select_action(state, training=True)
            next_state = np.random.randn(state_size).astype(np.float32) * 0.1
            reward = random.choice([0.0, 1.0, -1.0, 2.0, -2.0])
            total_reward += reward
            steps += 1
            if random.random() < 0.05:
                done = True
            agent.store_experience(state, action, reward, next_state, done)
        # Training step
        loss = agent.train_step()
        # Record metrics
        metrics.record_episode(total_reward, steps, total_reward > 50)
        if loss > 0:
            metrics.record_update(loss, 0.0)  # Placeholder for actual Q values
        # Evaluation
        if episode % eval_freq == 0 and episode > 0:
            eval_metrics = evaluate_agent(agent, num_episodes=10)
            logger.info("[EVAL] Episode %d | Reward: %.1f ± %.1f | Loss: %.4f | Success: %.1f%%",
                       episode, eval_metrics['mean_reward'], eval_metrics['std_reward'],
                       eval_metrics['mean_loss'], eval_metrics['success_rate'] * 100)
        # Logging
        if episode % 50 == 0:
            stats = metrics.get_metrics()
            logger.info("[TRAIN] Episode %d | Reward: %.1f | Epsilon: %.3f | Updates: %d | Memory: %d",
                       episode, stats['mean_reward'], agent.epsilon, agent.update_count, len(agent.memory))
        # Save checkpoint
        if episode % 500 == 0 and episode > 0:
            agent.save(save_path)
            logger.info("[TRAIN] Checkpoint saved: %s", save_path)
    # Final save
    agent.save(save_path)
    final_metrics = metrics.get_metrics()
    logger.info("[TRAIN] Training complete! Final metrics: %s", final_metrics)
    return final_metrics
def main_advanced():
    """CLI for advanced training."""
    parser = argparse.ArgumentParser(description="Brawlhalla Advanced RL Training")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--save", default="model/dqn_torch.pt", help="Model save path")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluation frequency")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = train_advanced(args.episodes, args.save, args.device, args.eval_freq)
    print(json.dumps(result, indent=2))
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--advanced":
    main_advanced()