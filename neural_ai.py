__all__ = [
    # Core neural network
    "xavier_init", "he_init", "orthogonal_init",
    "NeuralNetwork",
    # PPO components (PPOPolicyNetwork methods: compute_gae, train)
    "PPOPolicyNetwork",
    # Self-play engine
    "SelfPlayEngine", "RegretMinimizer", "NashOpponentSelector", "SelfPlayEngineNash",
    # MCTS engine
    "MCTSNode", "MCTSEngine",
    # Opponent modeling
    "OpponentModel", "OpponentFingerprint",
    # Curriculum learning
    "CurriculumStage", "CURRICULUM", "CurriculumManager",
    # Main AI interface
    "NeuralAI",
    # Constants
    "ACTIONS", "ACTION_DIM", "STATE_DIM",
    "STAGE_L", "STAGE_R", "STAGE_B",
    "HIT_CLOSE", "HIT_MEDIUM", "HIT_FAR", "HIT_SPECIAL",
]
#!/usr/bin/env python3
"""
Brawlhalla AI — Neural AI Engine (Wave 1-5: Full Intelligence Overhaul)

Implements:
  - Wave 1: PPO Policy Network + Value Network (pure numpy, no PyTorch)
  - Wave 2: Self-Play Engine with ELO evolution
  - Wave 3: MCTS Engine for critical decision lookahead
  - Wave 4: Opponent Modeling with behavior fingerprinting
  - Wave 5: Curriculum Learning with progressive difficulty

No external ML dependencies — pure NumPy implementation.
"""

import os
import sys
import json
import math
import copy
import random
import logging
import argparse
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Neural Network Foundation
    "NeuralNetwork", "xavier_init", "he_init", "orthogonal_init",
    # PPO Policy Network
    "PPOPolicyNetwork", "PPOAgent",
    # Opponent Modeling
    "OpponentFingerprint", "OpponentModel", "KMeansClusterer", "OpponentClusterManager",
    # MCTS Engine
    "MCTSNode", "MCTSEngine",
    # Self-Play Engine
    "SelfPlayEngine",
    # Curriculum Learning
    "CurriculumStage", "CURRICULUM", "CurriculumManager",
    # Neural AI — Full Integration
    "NeuralAI",
    # Training Loop
    "train_neural_ai", "main",
]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0: CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Action space (12 actions)
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
ACTION_DIM = len(ACTIONS)

# State space: 26 features
STATE_DIM = 26

# Stage boundaries
STAGE_L, STAGE_R = 0.03, 0.97
STAGE_B = 0.93
HIT_CLOSE, HIT_MEDIUM, HIT_FAR, HIT_SPECIAL = 0.07, 0.12, 0.18, 0.22

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: NEURAL NETWORK FOUNDATION (Pure NumPy, Python 3.14 Compatible)
# ══════════════════════════════════════════════════════════════════════════════

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot initialization for stable training."""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def orthogonal_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Orthogonal initialization for recurrent networks."""
    a = math.sqrt(1.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * a


class NeuralNetwork:
    """
    Multi-layer neural network with Xavier/He initialization.
    Supports: dense layers, tanh/sigmoid/relu/softmax activations, L2 regularization.
    No external dependencies — pure NumPy.
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        activations: List[str] = None,
        learning_rate: float = 0.0003,
        l2_reg: float = 1e-4,
        dropout: float = 0.0,
        optimizer: str = "adam",
    ):
        assert len(layer_dims) >= 2, "Need at least input + output layer"
        self.layer_dims = layer_dims
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.dropout = dropout
        
        # Initialize weights
        self.weights = []
        self.biases = []
        self._init_weights(optimizer)
        
        # Set activations
        self.activations = activations or ["relu"] * (len(layer_dims) - 2) + ["linear"]
        
        # Adam optimizer state
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t = 0
        
        # Gradient buffers
        self._grad_w = [np.zeros_like(w) for w in self.weights]
        self._grad_b = [np.zeros_like(b) for b in self.biases]
    
    def _init_weights(self, optimizer: str):
        for i in range(len(self.layer_dims) - 1):
            if optimizer == "he":
                W = he_init(self.layer_dims[i], self.layer_dims[i + 1])
            elif optimizer == "orthogonal":
                W = orthogonal_init(self.layer_dims[i], self.layer_dims[i + 1])
            else:  # xavier (default, good for tanh)
                W = xavier_init(self.layer_dims[i], self.layer_dims[i + 1])
            b = np.zeros(self.layer_dims[i + 1])
            self.weights.append(W)
            self.biases.append(b)
    
    @staticmethod
    def _activate(x: np.ndarray, name: str) -> np.ndarray:
        if name == "tanh":
            return np.tanh(x)
        elif name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif name == "relu":
            return np.maximum(0, x)
        elif name == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif name == "elu":
            return np.where(x > 0, x, np.exp(x) - 1)
        elif name == "softmax":
            e = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e / (np.sum(e, axis=-1, keepdims=True) + 1e-8)
        else:  # linear
            return x
    
    @staticmethod
    def _activate_grad(x: np.ndarray, name: str) -> np.ndarray:
        if name == "tanh":
            return 1.0 - np.tanh(x) ** 2
        elif name == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        elif name == "relu":
            return (x > 0).astype(float)
        elif name == "leaky_relu":
            return np.where(x > 0, 1.0, 0.01)
        elif name == "elu":
            return np.where(x > 0, 1.0, np.exp(x))
        else:
            return np.ones_like(x)
    
    def forward(self, x: np.ndarray, training: bool = False) -> List[np.ndarray]:
        """Forward pass, returns list of (pre-activation, post-activation) for each layer."""
        self._cache = [(x, x)]
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            pre = self._cache[-1][1] @ W + b
            post = self._activate(pre, self.activations[i])
            
            # Dropout during training
            if training and self.dropout > 0 and i < len(self.weights) - 1:
                mask = (np.random.rand(*post.shape) > self.dropout).astype(float)
                post *= mask / (1.0 - self.dropout)
            
            self._cache.append((pre, post))
        
        return self._cache
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Single forward pass, returns output."""
        out = self.forward(x, training=False)
        return out[-1][1]
    
    def backward(self, grad_output: np.ndarray):
        """Backpropagation. Computes gradients and applies Adam update."""
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Clip gradients for stability
        grad_output = np.clip(grad_output, -10.0, 10.0)
        
        # Backprop through layers
        delta = grad_output
        for i in range(len(self.weights) - 1, -1, -1):
            pre, post = self._cache[i + 1]
            act_grad = self._activate_grad(pre, self.activations[i])
            delta = delta * act_grad
            
            # Gradient w.r.t. weights (L2 regularization)
            input_act = self._cache[i][1]
            grad_w = input_act.T @ delta + self.l2_reg * self.weights[i]
            grad_b = np.sum(delta, axis=0)
            
            # Accumulate gradients
            self._grad_w[i] += grad_w
            self._grad_b[i] += grad_b
            
            # Backprop delta to previous layer
            if i > 0:
                delta = delta @ self.weights[i].T
        
        return self._grad_w, self._grad_b
    
    def apply_gradients(self, batch_size: int):
        """Apply accumulated gradients with Adam optimizer."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for i in range(len(self.weights)):
            # Normalize by batch size
            g_w = self._grad_w[i] / batch_size
            g_b = self._grad_b[i] / batch_size
            
            # Adam update for weights
            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * g_w
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * g_w ** 2
            m_hat = self._m_w[i] / (1 - beta1 ** self._t)
            v_hat = self._v_w[i] / (1 - beta2 ** self._t)
            self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Adam update for biases
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * g_b
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * g_b ** 2
            m_hat_b = self._m_b[i] / (1 - beta1 ** self._t)
            v_hat_b = self._v_b[i] / (1 - beta2 ** self._t)
            self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
            
            # Clear gradients
            self._grad_w[i] = np.zeros_like(self._grad_w[i])
            self._grad_b[i] = np.zeros_like(self._grad_b[i])
    
    def copy_from(self, other: "NeuralNetwork"):
        """Copy weights from another network (for target network updates)."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()
    
    def mutate(self, rate: float = 0.02, strength: float = 0.1):
        """Apply random mutations for evolution (self-play)."""
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) < rate
            noise = np.random.randn(*self.weights[i].shape) * strength
            self.weights[i] += noise * mask
    
    def load(self, path: str):
        """Load weights from NPZ file."""
        try:
            data = np.load(path, allow_pickle=True)
            for i, key in enumerate([f"w{i}" for i in range(len(self.weights))]):
                self.weights[i] = data[key]
            for i, key in enumerate([f"b{i}" for i in range(len(self.biases))]):
                self.biases[i] = data[key]
            logger.info("[NN] Loaded weights from: %s", path)
        except Exception as e:
            logger.warning("[NN] Failed to load: %s", e)
    
    def save(self, path: str):
        """Save weights to NPZ file."""
        data = {}
        for i, w in enumerate(self.weights):
            data[f"w{i}"] = w
        for i, b in enumerate(self.biases):
            data[f"b{i}"] = b
        np.savez(path, **data)
        logger.info("[NN] Saved weights to: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PPO POLICY NETWORK (Wave 1)
# ══════════════════════════════════════════════════════════════════════════════

class PPOPolicyNetwork:
    """
    PPO (Proximal Policy Optimization) Policy + Value Network.
    
    - Actor: outputs action probabilities (policy)
    - Critic: outputs state value estimate
    
    Uses clip surrogate loss for stable training.
    Generalized Advantage Estimation (GAE) for credit assignment.
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,  # GAE lambda
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        target_kl: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        
        # Shared feature extractor
        shared_dims = [state_dim, hidden, hidden, hidden // 2]
        self.shared = NeuralNetwork(
            shared_dims,
            activations=["leaky_relu", "leaky_relu", "leaky_relu"],
            learning_rate=lr,
            l2_reg=1e-5,
            dropout=0.05,
            optimizer="xavier",
        )
        
        # Actor head (policy)
        actor_dims = [hidden // 2, hidden // 2, action_dim]
        self.actor = NeuralNetwork(
            actor_dims,
            activations=["relu", "relu", "softmax"],
            learning_rate=lr,
            l2_reg=1e-5,
            dropout=0.0,
            optimizer="he",
        )
        
        # Critic head (value function)
        critic_dims = [hidden // 2, hidden // 2, 1]
        self.critic = NeuralNetwork(
            critic_dims,
            activations=["relu", "relu", "linear"],
            learning_rate=lr,
            l2_reg=1e-5,
            dropout=0.0,
            optimizer="he",
        )
        
        # Target critic for stable training
        self.target_critic = copy.deepcopy(self.critic)
        
        # Memory for PPO updates
        self.memory: deque = deque(maxlen=10000)
        self._log_probs_history: deque = deque(maxlen=10000)
        
        # Training stats
        self._update_count = 0
        self._avg_loss = 0.0
    
    def forward_shared(self, state: np.ndarray) -> np.ndarray:
        """Shared feature extraction."""
        return self.shared.predict(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float, float]:
        """
        Get action with epsilon-greedy exploration.
        
        Returns:
            (action_index, log_prob, value_estimate)
        """
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
            log_prob = math.log(1.0 / self.action_dim)
        else:
            features = self.forward_shared(state.reshape(1, -1))
            probs = self.actor.predict(features).flatten()
            probs = np.clip(probs, 1e-8, 1.0)
            probs = probs / probs.sum()
            action = int(np.random.choice(self.action_dim, p=probs))
            log_prob = math.log(probs[action])
        # Use shared features for critic too
        features = self.forward_shared(state.reshape(1, -1))
        value = self.critic.predict(features).item()
        return action, log_prob, value
    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate."""
        features = self.forward_shared(state.reshape(1, -1))
        return float(self.critic.predict(features).item())
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in replay memory."""
        self.memory.append({
            "state": state, "action": action, "reward": reward,
            "next_state": next_state, "done": done,
            "log_prob": log_prob, "value": value,
            "advantage": 0.0, "return": 0.0,
        })
    
    def compute_gae(self, terminal_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation + returns for all transitions.
        Uses TD(λ) for smooth advantage estimation.
        """
        if len(self.memory) == 0:
            return
        
        rewards = []
        values = []
        
        for i, transition in enumerate(self.memory):
            rewards.append(transition["reward"])
            values.append(transition["value"])
        
        # Add terminal value
        values.append(terminal_value)
        
        advantages = []
        gae = 0.0
        
        # Backward pass for GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        
        # Compute returns (advantage + value baseline)
        returns = [advantages[i] + values[i] for i in range(len(rewards))]
        
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = [(a - adv_mean) / adv_std for a in advantages]
        
        # Store computed advantages and returns
        for i, transition in enumerate(self.memory):
            transition["advantage"] = advantages[i]
            transition["return"] = returns[i]
    
    def update(self, batch_size: int = 64, epochs: int = 10) -> dict:
        """
        PPO update: maximize clip surrogate objective.
        L = E[min(r * A, clip(r, 1-ε, 1+ε) * A) - c1 * (V - V_target)^2 + c2 * H]
        where r = π_new / π_old (importance ratio)
        """
        if len(self.memory) < batch_size:
            return {"skipped": True}
        # Compute GAE for all transitions
        self.compute_gae()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0
        for epoch in range(epochs):
            # Shuffle memory
            indices = list(range(len(self.memory)))
            random.shuffle(indices)
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                # Collect batch data
                states = np.array([self.memory[i]["state"] for i in batch_idx])
                actions = np.array([self.memory[i]["action"] for i in batch_idx])
                old_log_probs = np.array([self.memory[i]["log_prob"] for i in batch_idx])
                advantages = np.array([self.memory[i]["advantage"] for i in batch_idx])
                returns = np.array([self.memory[i]["return"] for i in batch_idx])
                # Forward pass
                features = self.shared.predict(states)
                probs = self.actor.predict(features)
                values = self.critic.predict(features).flatten()
                # Compute entropy (encourages exploration)
                probs_clipped = np.clip(probs, 1e-8, 1.0)
                entropy = -np.sum(probs * np.log(probs_clipped), axis=1).mean()
                total_entropy += entropy
                # Compute log probabilities for current policy
                new_log_probs = np.sum(np.log(probs + 1e-8) * np.eye(self.action_dim)[actions], axis=1)
                # Importance ratio
                ratios = np.exp(new_log_probs - old_log_probs)
                # PPO clipped surrogate loss
                surr1 = ratios * advantages
                surr2 = np.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -np.mean(np.minimum(surr1, surr2))
                # Value function loss with clipping (PPO-style)
                clipped_values = values + np.clip(values - returns, -self.clip_epsilon, self.clip_epsilon)
                value_loss = np.mean((values - returns) ** 2)
                value_clipped_loss = np.mean((clipped_values - returns) ** 2)
                value_loss = np.maximum(value_loss, value_clipped_loss)
                # KL divergence (for early stopping)
                kl = np.mean(old_log_probs - new_log_probs)
                total_kl += kl
                # Backpropagate actor + shared
                self._backprop_policy(states, actions, advantages, old_log_probs, probs)
                # Backpropagate critic
                self._backprop_critic(states, returns)
                # Apply gradients with gradient clipping
                self.actor.apply_gradients(batch_size)
                self.shared.apply_gradients(batch_size)
                self.critic.apply_gradients(batch_size)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                n_updates += 1
                # Early stopping if KL divergence too large
                if kl > self.target_kl * 1.5:
                    logger.info("[PPO] Early stopping: KL=%.4f > %.4f", kl, self.target_kl)
                    break
            # Update target critic (polyak averaging)
            self._update_target_critic(tau=0.005)
        self._update_count += 1
        if n_updates > 0:
            self._avg_loss = (total_policy_loss + total_value_loss) / n_updates
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "kl": total_kl / max(n_updates, 1),
            "n_updates": n_updates,
            "memory_size": len(self.memory),
        }
    
    def _backprop_policy(self, states, actions, advantages, old_log_probs, probs):
        """Backprop through actor + shared for policy loss."""
        batch_size = len(states)
        
        # Forward pass through shared layers
        features = self.shared.predict(states)
        
        # Forward pass through actor
        probs_out = self.actor.predict(features)
        
        # Compute gradient of policy loss w.r.t. action probabilities
        # Using cross-entropy loss gradient: d(CE)/d(probs) = probs - one_hot(action)
        # But for the gradient flowing through, we use advantage-weighted gradient
        grad_output = np.zeros((batch_size, self.action_dim))
        for i, (action, adv) in enumerate(zip(actions, advantages)):
            grad_output[i, action] = adv / batch_size
        
        # Backprop through actor (policy head)
        self.actor.backward(grad_output)
        
        # Get gradient from actor output to shared features
        actor_grad = self.actor.grads[0]  # gradient w.r.t. actor input (shared output)
        
        # Backprop through shared layers with actor gradient
        self.shared.backward(actor_grad)
        
        # Clip gradients for stability
        self._clip_gradients()
    def _backprop_critic(self, states, returns):
        """Backprop through critic for value loss."""
        batch_size = len(states)
        features = self.shared.predict(states)
        values = self.critic.predict(features).flatten()
        grad_output = 2.0 * (values - returns).reshape(-1, 1) / batch_size
        self.critic.backward(grad_output)
        # Backprop shared with ones gradient
        critic_input_grad = np.ones((batch_size, self.shared.layer_dims[-1]))
        self.shared.backward(critic_input_grad)
    
    def _update_target_critic(self, tau: float = 0.005):
        """Polyak averaging for target network update."""
        for i in range(len(self.critic.weights)):
            self.target_critic.weights[i] = (
                tau * self.critic.weights[i] +
                (1 - tau) * self.target_critic.weights[i]
            )
            self.target_critic.biases[i] = (
                tau * self.critic.biases[i] +
                (1 - tau) * self.target_critic.biases[i]
            )
    def _clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients to prevent exploding gradients."""
        for network in [self.shared, self.actor, self.critic]:
            for i in range(len(network.grads)):
                grad = network.grads[i]
                norm = np.linalg.norm(grad)
                if norm > max_norm:
                    network.grads[i] = grad * (max_norm / norm)
    
    def compute_value_loss_clipped(self, states: np.ndarray, returns: np.ndarray) -> float:
        """Compute clipped value function loss for PPO."""
        features = self.shared.predict(states)
        values = self.critic.predict(features).flatten()
        
        # Clipped value loss as per Schulman et al. 2017
        unclipped_loss = (values - returns) ** 2
        clipped_values = np.clip(values, returns - self.clip_epsilon, returns + self.clip_epsilon)
        clipped_loss = (clipped_values - returns) ** 2
        
        return float(np.mean(np.maximum(unclipped_loss, clipped_loss)))

    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.shared.save(path.replace(".npz", "_shared.npz"))
        self.actor.save(path.replace(".npz", "_actor.npz"))
        self.critic.save(path.replace(".npz", "_critic.npz"))
    
    def load(self, path: str):
        self.shared.load(path.replace(".npz", "_shared.npz"))
        self.actor.load(path.replace(".npz", "_actor.npz"))
        self.critic.load(path.replace(".npz", "_critic.npz"))
    def mutate(self, rate: float = 0.02, strength: float = 0.1):
        """Apply random mutations to all sub-networks for evolution."""
        self.shared.mutate(rate=rate, strength=strength)
        self.actor.mutate(rate=rate, strength=strength)
        self.critic.mutate(rate=rate, strength=strength)
    def copy_from(self, other: "PPOPolicyNetwork"):
        """Copy weights from another PPO network."""
        self.shared.copy_from(other.shared)
        self.actor.copy_from(other.actor)
        self.critic.copy_from(other.critic)
        self.target_critic.copy_from(other.target_critic)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OPPONENT MODELING (Wave 4)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpponentFingerprint:
    """Complete behavioral fingerprint of an opponent."""
    # Attack patterns
    attack_tendencies: Dict[str, float] = field(default_factory=dict)  # situation -> prob
    preferred_attacks: Counter = field(default_factory=Counter)
    
    # Movement patterns
    approach_style: str = "unknown"  # aggressive/balanced/defensive
    retreat_threshold: float = 0.0  # distance threshold for retreating
    jump_frequency: float = 0.0  # normalized 0-1
    
    # Defensive patterns
    dodge_direction_bias: str = "away"  # toward/away/mixed
    shield_frequency: float = 0.0
    dodge_timing: float = 3.0  # frames after attack start
    
    # Combo behavior
    combo_length_avg: float = 0.0
    combo_end_practice: float = 0.0  # how well they end combos
    
    # Recovery
    ledge_option_preference: str = "mixed"  # which ledge options they prefer
    recovery_speed: float = 0.5  # 0-1 how fast they recover
    
    # Meta
    reaction_time_frames: float = 3.0  # estimated reaction time
    confidence: float = 0.0  # how confident we are in this model (0-1)
    archetype: str = "unknown"  # aggressive/defensive/technical/unpredictable
    archetype_confidence: float = 0.0  # confidence in archetype assignment
    sample_count: int = 0

class OpponentModel:
    """
    Learns and adapts to specific opponent behavior.
    Builds a behavioral fingerprint and predicts opponent actions.
    """
    
    def __init__(self):
        self.fingerprint = OpponentFingerprint()
        self._action_history: deque = deque(maxlen=500)
        self._situation_history: deque = deque(maxlen=500)
        self._attack_frames: deque = deque(maxlen=50)
        self._dodge_frames: deque = deque(maxlen=50)
        self._current_opponent: str = "unknown"
        self._last_update = 0
    
    def update(self, frame: int, opponent_action: str, situation: dict, player_action: str = None):
        """Update opponent model with new observation."""
        self._action_history.append(opponent_action)
        self._situation_history.append(situation.copy())
        
        # Track attack patterns
        if "attack" in opponent_action.lower():
            self._attack_frames.append(frame)
        
        # Track dodge patterns
        if opponent_action in ("jump", "dash_left", "dash_right"):
            self._dodge_frames.append(frame)
        
        # Update fingerprint
        self._update_attack_tendencies(situation, opponent_action)
        self._update_movement_patterns()
        self._update_defensive_patterns(frame)
        self._update_combo_pattern()
        
        self.fingerprint.sample_count += 1
        self.fingerprint.confidence = min(1.0, self.fingerprint.sample_count / 100)
    
    def _update_attack_tendencies(self, situation: dict, action: str):
        """Build situation → action probability mapping."""
        # Create situation signature (quantized state)
        dist = situation.get("dist", 0.5)
        player_above = situation.get("player_above", False)
        
        if dist < HIT_CLOSE:
            key = "close_range"
        elif dist < HIT_MEDIUM:
            key = "medium_range"
        elif dist < HIT_FAR:
            key = "far_range"
        else:
            key = "very_far"
        
        if player_above:
            key += "_airborne"
        
        # Bayesian update of attack probability
        current = self.fingerprint.attack_tendencies.get(key, 0.0)
        is_attack = 1.0 if "attack" in action.lower() else 0.0
        
        # EMA update
        self.fingerprint.attack_tendencies[key] = current * 0.9 + is_attack * 0.1
    
    def _update_movement_patterns(self):
        """Analyze movement behavior."""
        actions = list(self._action_history)
        if len(actions) < 10:
            return
        
        approach_count = sum(1 for a in actions if "move_right" in a)
        retreat_count = sum(1 for a in actions if "move_left" in a)
        total_moves = approach_count + retreat_count
        
        if total_moves > 0:
            ratio = approach_count / total_moves
            if ratio > 0.7:
                self.fingerprint.approach_style = "aggressive"
            elif ratio < 0.3:
                self.fingerprint.approach_style = "defensive"
            else:
                self.fingerprint.approach_style = "balanced"
        
        jump_count = sum(1 for a in actions if "jump" in a)
        self.fingerprint.jump_frequency = jump_count / len(actions)
    
    def _update_defensive_patterns(self, frame: int):
        """Analyze defensive behavior."""
        actions = list(self._action_history)
        if len(actions) < 10:
            return
        
        dodge_count = sum(1 for a in actions if "jump" in a or "dash" in a)
        shield_count = sum(1 for a in actions if "shield" in a)
        
        total_defensive = dodge_count + shield_count
        if total_defensive > 0:
            self.fingerprint.dodge_direction_bias = "away"  # default
            self.fingerprint.shield_frequency = shield_count / len(actions)
        
        # Estimate reaction time from attack spacing
        if len(self._attack_frames) >= 2:
            gaps = [self._attack_frames[i] - self._attack_frames[i-1]
                    for i in range(1, len(self._attack_frames))]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                self.fingerprint.reaction_time_frames = max(1, avg_gap * 0.3)
    
    def _update_combo_pattern(self):
        """Analyze combo behavior."""
        actions = list(self._action_history)
        
        # Count consecutive attack patterns
        in_combo = 0
        combo_lengths = []
        max_combo = 0
        
        for a in actions:
            if "attack" in a.lower():
                in_combo += 1
                max_combo = max(max_combo, in_combo)
            else:
                if in_combo > 0:
                    combo_lengths.append(in_combo)
                in_combo = 0
        
        if combo_lengths:
            self.fingerprint.combo_length_avg = sum(combo_lengths) / len(combo_lengths)
        else:
            self.fingerprint.combo_length_avg = max_combo
    
    def predict_action(self, situation: dict) -> Dict[str, Any]:
        """
        Predict opponent's next action based on learned fingerprint.
        
        Returns:
            {"action": str, "confidence": float, "counter_action": str, "reasoning": str}
        """
        if self.fingerprint.sample_count < 5:
            return {"action": "unknown", "confidence": 0.0, "counter_action": "idle", "reasoning": "insufficient data"}
        
        # Build situation key
        dist = situation.get("dist", 0.5)
        player_above = situation.get("player_above", False)
        
        if dist < HIT_CLOSE:
            key = "close_range"
        elif dist < HIT_MEDIUM:
            key = "medium_range"
        elif dist < HIT_FAR:
            key = "far_range"
        else:
            key = "very_far"
        
        if player_above:
            key += "_airborne"
        
        # Get attack probability for this situation
        attack_prob = self.fingerprint.attack_tendencies.get(key, 0.3)
        confidence = self.fingerprint.confidence
        
        # Predict action
        if attack_prob > 0.5:
            predicted = "heavy_attack"  # Most common high-damage attack
            counter = "dodge_away"
            reasoning = f"high attack prob ({attack_prob:.2f}) in {key}"
        elif self.fingerprint.approach_style == "aggressive":
            predicted = "move_right"
            counter = "shield_back"
            reasoning = "aggressive player approaching"
        elif self.fingerprint.approach_style == "defensive":
            predicted = "idle"
            counter = "light_attack"
            reasoning = "defensive player, waiting"
        else:
            predicted = random.choice(["move_right", "idle", "jump"])
            counter = "jump"
            reasoning = "balanced player, mixed prediction"
        
        return {
            "action": predicted,
            "confidence": confidence * attack_prob,
            "counter_action": counter,
            "reasoning": reasoning,
        }
    def reset(self):
        """Reset model for new opponent."""
        self.fingerprint = OpponentFingerprint()
        self._action_history.clear()
        self._situation_history.clear()
        self._attack_frames.clear()
        self._dodge_frames.clear()
    
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Extract a fixed-size feature vector from the current fingerprint.
        Used for clustering opponents into archetypes.
        Feature order (10 dimensions):
          0: attack_rate       - proportion of actions that are attacks
          1: jump_rate         - jump frequency
          2: shield_rate       - shield frequency
          3: approach_ratio    - approach / total moves (aggressive = high)
          4: combo_length      - average combo length (normalized)
          5: reaction_time     - normalized reaction time (low = technical)
          6: recovery_speed    - recovery speed (0-1)
          7: dodge_rate        - dodge / total actions
          8: retreat_threshold - retreat distance threshold
          9: combo_practice    - combo end practice score
        """
        fp = self.fingerprint
        total = max(1, len(self._action_history))
        # Attack rate
        attack_rate = sum(1 for a in self._action_history if "attack" in a.lower()) / total
        # Jump rate
        jump_rate = sum(1 for a in self._action_history if "jump" in a.lower()) / total
        # Shield rate
        shield_rate = sum(1 for a in self._action_history if "shield" in a.lower()) / total
        # Approach ratio
        approach = sum(1 for a in self._action_history if "move_right" in a)
        retreat = sum(1 for a in self._action_history if "move_left" in a)
        approach_ratio = approach / max(1, approach + retreat)
        # Dodge rate
        dodge_rate = sum(1 for a in self._action_history if "dash" in a or "jump" in a) / total
        return np.array([
            attack_rate,
            jump_rate,
            shield_rate,
            approach_ratio,
            min(1.0, fp.combo_length_avg / 10.0),
            max(0.0, min(1.0, fp.reaction_time_frames / 20.0)),
            fp.recovery_speed,
            dodge_rate,
            min(1.0, fp.retreat_threshold),
            fp.combo_end_practice,
        ], dtype=np.float64)

    def adjust_behavior(self, archetype: str, archetype_confidence: float) -> Dict[str, Any]:
        """
        Adjust AI behavior parameters based on opponent archetype.
        Also updates the fingerprint with archetype assignment.
        Returns tuned parameters for NeuralAI to use:
          aggression_mod: multiplier for attack tendency
          patience_mod: multiplier for waiting/defensive patience
          dodge_bias: bias for dodge timing adjustments
          combo_risk: acceptable risk in combo situations
          approach_strategy: preferred approach style
        """

        self.fingerprint.archetype = archetype
        self.fingerprint.archetype_confidence = archetype_confidence
        if archetype == "aggressive":
            return {
                "aggression_mod": 1.4,
                "patience_mod": 0.7,
                "dodge_bias": 0.2,
                "combo_risk": 0.9,
                "approach_strategy": "retreat_bait",
                "shield_weight": 1.5,
                "edge_guard_aggression": 1.2,
            }
        elif archetype == "defensive":
            return {
                "aggression_mod": 0.6,
                "patience_mod": 1.3,
                "dodge_bias": -0.1,
                "combo_risk": 0.5,
                "approach_strategy": "pressure_close",
                "shield_weight": 0.8,
                "edge_guard_aggression": 0.7,
            }
        elif archetype == "technical":
            return {
                "aggression_mod": 1.0,
                "patience_mod": 1.1,
                "dodge_bias": 0.0,
                "combo_risk": 0.7,
                "approach_strategy": "neutral_focus",
                "shield_weight": 1.0,
                "edge_guard_aggression": 1.0,
            }
        else:  # unpredictable
            return {
                "aggression_mod": 1.0,
                "patience_mod": 1.0,
                "dodge_bias": 0.0,
                "combo_risk": 0.6,
                "approach_strategy": "adaptive",
                "shield_weight": 1.0,
                "edge_guard_aggression": 1.0,
            }

    def get_fingerprint_summary(self) -> dict:
        """Get human-readable fingerprint summary."""
        return {
            "approach_style": self.fingerprint.approach_style,
            "jump_frequency": f"{self.fingerprint.jump_frequency:.1%}",
            "shield_frequency": f"{self.fingerprint.shield_frequency:.1%}",
            "combo_avg": f"{self.fingerprint.combo_length_avg:.1f}",
            "reaction_time": f"{self.fingerprint.reaction_time_frames:.1f}f",
            "confidence": f"{self.fingerprint.confidence:.0%}",
            "archetype": self.fingerprint.archetype,
            "archetype_confidence": f"{self.fingerprint.archetype_confidence:.0%}",
            "samples": self.fingerprint.sample_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
# OPPONENT CLUSTERING (Archetype Discovery)
# ══════════════════════════════════════════════════════════════════════════════
ARCHETYPE_NAMES = ["aggressive", "defensive", "technical", "unpredictable"]
ARCHETYPE_FEATURES = {
    "aggressive":   [0.7, 0.5, 0.2, 0.8, 0.5, 0.5, 0.5, 0.4, 0.3, 0.5],
    "defensive":    [0.3, 0.3, 0.7, 0.2, 0.4, 0.6, 0.6, 0.5, 0.7, 0.6],
    "technical":    [0.5, 0.6, 0.4, 0.5, 0.8, 0.3, 0.7, 0.6, 0.5, 0.8],
    "unpredictable":[0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5],
}
class KMeansClusterer:
    """
    Pure NumPy k-means implementation for opponent archetype clustering.
    Clusters opponent feature vectors into behavioral archetypes.
    Uses k-means++ initialization for better convergence.
    """
    def __init__(self, n_clusters: int = 4, max_iters: int = 100, tol: float = 1e-4,
                 seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray = None
        self.labels: np.ndarray = None
        self.inertia_: float = 0.0
    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            self.centroids = X
            self.labels = np.arange(n_samples)
            self.inertia_ = 0.0
            return self
        self.centroids = self._init_centroids(X, n_features)
        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        self.labels = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels)
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._assign_clusters(X)
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels
    def _init_centroids(self, X: np.ndarray, n_features: int) -> np.ndarray:
        """k-means++ initialization."""
        centroids = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            dists = np.zeros(len(X))
            for c in centroids:
                d = X - c
                dists += (d * d).sum(axis=1)
            probs = dists / dists.sum()
            centroids.append(X[self.rng.choice(len(X), p=probs)])
        return np.array(centroids, dtype=np.float64)
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        dists = np.zeros((len(X), self.n_clusters))
        for i, c in enumerate(self.centroids):
            d = X - c
            dists[:, i] = (d * d).sum(axis=1)
        return dists.argmin(axis=1)
    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute mean of each cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = X[self.rng.integers(len(X))]
        return new_centroids
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Sum of squared distances to nearest centroid."""
        total = 0.0
        for i, k in enumerate(labels):
            d = X[i] - self.centroids[k]
            total += (d * d).sum()
        return total
    def get_archetype_labels(self, label_names: List[str]) -> List[str]:
        """Map cluster indices to archetype names."""
        return [label_names[k] for k in self.labels]
class OpponentClusterManager:
    """
    Manages multiple opponent models and clusters them into archetypes.
    Responsibilities:
      Register opponents and their feature vectors
      Run k-means clustering to discover archetype groups
      Track cluster centroids and assign archetype labels
      Provide archetype-aware behavior adjustments for new opponents
    """
    def __init__(self, n_archetypes: int = 4, min_opponents: int = 3,
                 cluster_seed: int = 42):
        self.n_archetypes = n_archetypes
        self.min_opponents = min_opponents
        self.clusterer = KMeansClusterer(n_clusters=n_archetypes, seed=cluster_seed)
        self.opponents: Dict[str, "OpponentModel"] = {}
        self.feature_vectors: Dict[str, np.ndarray] = {}
        self.centroids: np.ndarray = None
        self.archetype_labels: List[str] = ARCHETYPE_NAMES[:n_archetypes]
        self._last_cluster_round = -1
    def register_opponent(self, opponent_id: str, model: "OpponentModel"):
        """Register an opponent's model for clustering."""
        self.opponents[opponent_id] = model
        self._last_cluster_round = -1
    def update_opponent(self, opponent_id: str, model: "OpponentModel"):
        """Update an existing opponent's model."""
        self.opponents[opponent_id] = model
        self._last_cluster_round = -1
    def unregister_opponent(self, opponent_id: str):
        """Remove an opponent from clustering."""
        self.opponents.pop(opponent_id, None)
        self.feature_vectors.pop(opponent_id, None)
        self._last_cluster_round = -1
    def get_feature_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """Collect current feature vectors for all registered opponents."""
        ids = []
        vectors = []
        for opp_id, model in self.opponents.items():
            if model.fingerprint.sample_count >= 10:
                vec = model.get_feature_vector()
                vectors.append(vec)
                ids.append(opp_id)
        if vectors:
            return np.array(vectors, dtype=np.float64), ids
        return np.zeros((0, 10), dtype=np.float64), []
    def fit_clusters(self) -> Dict[str, str]:
        """Run k-means clustering over all registered opponents."""
        X, opp_ids = self.get_feature_vectors()
        if len(X) < self.min_opponents:
            self.centroids = None
            return {oid: "unknown" for oid in self.opponents}
        n_c = min(self.n_archetypes, len(X))
        clusterer = KMeansClusterer(n_clusters=n_c, seed=self.clusterer.rng.integers(1e9))
        clusterer.fit(X)
        self.centroids = clusterer.centroids
        self._last_cluster_round = len(X)
        centroid_to_archetype = self._match_centroids_to_archetypes(clusterer.centroids)
        assignments = {}
        for k, oid in zip(clusterer.labels, opp_ids):
            archetype = centroid_to_archetype.get(k, "unpredictable")
            assignments[oid] = archetype
            confidence = clusterer.inertia_ / max(1, len(X))
            self.opponents[oid].adjust_behavior(archetype, confidence)
        for oid in self.opponents:
            if oid not in assignments:
                assignments[oid] = "unknown"
        return assignments
    def _match_centroids_to_archetypes(self, centroids: np.ndarray) -> Dict[int, str]:
        """Assign each cluster centroid to nearest archetype by cosine similarity."""
        archetype_vecs = np.array(
            [ARCHETYPE_FEATURES.get(a, [0.5] * 10) for a in self.archetype_labels],
            dtype=np.float64
        )
        assignment = {}
        used = set()
        for k in range(len(centroids)):
            best_arch, best_score = None, -1.0
            for i, arch in enumerate(self.archetype_labels):
                if i in used:
                    continue
                dot = np.dot(centroids[k], archetype_vecs[i])
                norm = np.linalg.norm(centroids[k]) * np.linalg.norm(archetype_vecs[i])
                score = dot / max(1e-8, norm)
                if score > best_score:
                    best_score = score
                    best_arch = arch
                    best_i = i
            if best_arch is not None:
                assignment[k] = best_arch
                used.add(best_i)
            else:
                assignment[k] = self.archetype_labels[k % len(self.archetype_labels)]
        return assignment
    def get_archetype_for_opponent(self, opponent_id: str) -> Tuple[str, float]:
        """Get the archetype and confidence for a specific opponent."""
        if opponent_id in self.opponents:
            model = self.opponents[opponent_id]
            return model.fingerprint.archetype, model.fingerprint.archetype_confidence
        return "unknown", 0.0
    def get_archetype_stats(self) -> Dict[str, Any]:
        """Get summary statistics about archetype distribution."""
        archetype_counts = Counter(
            m.fingerprint.archetype for m in self.opponents.values()
        )
        return {
            "total_opponents": len(self.opponents),
            "archetype_distribution": dict(archetype_counts),
            "centroids_shape": self.centroids.shape if self.centroids is not None else None,
            "cluster_round_samples": self._last_cluster_round,
        }
    def recommend_adjustment(self, opponent_id: str) -> Dict[str, Any]:
        """Get behavior adjustment parameters for a specific opponent."""
        if opponent_id not in self.opponents:
            return {
                "aggression_mod": 1.0,
                "patience_mod": 1.0,
                "dodge_bias": 0.0,
                "combo_risk": 0.6,
                "approach_strategy": "adaptive",
                "shield_weight": 1.0,
                "edge_guard_aggression": 1.0,
            }
        model = self.opponents[opponent_id]
        base = model.adjust_behavior(
            model.fingerprint.archetype,
            model.fingerprint.archetype_confidence
        )
        fp = model.fingerprint
        if fp.shield_frequency < 0.1:
            base["aggression_mod"] *= 1.2
        if fp.combo_length_avg > 4:
            base["patience_mod"] *= 1.15
        if fp.combo_end_practice > 0.7:
            base["combo_risk"] *= 0.85
        return base
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MCTS ENGINE (Wave 3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """Monte Carlo Tree Search node."""
    state_hash: int = 0
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    wins: float = 0.0
    q_value: float = 0.0  # average reward
    prior_prob: float = 0.0
    depth: int = 0


class MCTSEngine:
    """
    Monte Carlo Tree Search for critical decision lookahead.
    
    Uses:
    - UCT (Upper Confidence Bound for Trees) for selection
    - Policy network for prior probabilities and rollouts
    - Progressive widening for large action spaces
    """
    
    def __init__(
        self,
        policy_net: PPOPolicyNetwork,
        opponent_model: OpponentModel = None,
        max_iterations: int = 200,
        max_depth: int = 8,
        exploration_const: float = 1.41,
        discount_factor: float = 0.99,
    ):
        self.policy = policy_net
        self.opponent = opponent_model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.C = exploration_const
        self.gamma = discount_factor
        
        self.root: Optional[MCTSNode] = None
        self._rng = random.Random()
        
        # Transposition table for O(1) node lookup (avoids recomputing seen states)
        self._transposition_table: Dict[int, Dict[str, Any]] = {}
        self._transposition_max_size = 10000
        
        # Parallel search support
        self._use_parallel = False
    
    def search(self, state: np.ndarray, legal_actions: List[int],
               critical_only: bool = True) -> Optional[int]:
        """
        Run MCTS search to find best action.
        Args:
            state: Current state vector
            legal_actions: List of valid action indices
            critical_only: Only use MCTS for critical decisions (reduces computation)
        Returns:
            Best action index, or None if search should be skipped
        """
        # Budget management: skip MCTS if not enough iterations can be done
        if critical_only and len(legal_actions) <= 2:
            return None
        # Initialize root
        state_hash = hash(state.tobytes()[:32])
        self.root = MCTSNode(state_hash=state_hash, depth=0)
        # Check transposition table for existing tree
        if state_hash in self._transposition_table:
            cached = self._transposition_table[state_hash]
            self.root.visits = cached.get("visits", 0)
            self.root.q_value = cached.get("q_value", 0.0)
        # Set prior probabilities from policy network
        probs = self.policy.forward_shared(state.reshape(1, -1))
        action_probs = self.policy.actor.predict(probs).flatten()
        for action in legal_actions:
            prior = float(action_probs[action])
            child_node = MCTSNode(
                state_hash=state_hash + action * 7919,  # Simple state hash
                parent=self.root,
                action=action,
                prior_prob=prior + 1e-8,
                depth=1,
            )
            self.root.children[action] = child_node
        # MCTS iterations
        for _ in range(self.max_iterations):
            self._uct_iteration(state.copy(), legal_actions)
        # Store in transposition table
        self._transposition_table[state_hash] = {
            "visits": self.root.visits,
            "q_value": self.root.q_value,
            "children": {a: c.visits for a, c in self.root.children.items()},
        }
        # Return most visited action (robust to noise)
        if not self.root.children:
            return legal_actions[0] if legal_actions else None
        return max(self.root.children.items(), key=lambda x: x[1].visits)[0]

    def sample_root_action(self, top_k: int = None, temperature: float = 1.0) -> Optional[int]:
        """
        Sample an action from the root using visit-count-weighted probability.
        
        Encourages exploration by sampling from top-k candidates rather than
        always picking the most visited action.
        
        Args:
            top_k: Number of top actions to consider. None means all children.
                   Defaults to min(8, len(children)).
            temperature: Sampling temperature. Higher = more uniform sampling.
                        temperature -> 0 = deterministic (most visited).
        
        Returns:
            Sampled action index, or None if root has no children.
        """
        if not self.root or not self.root.children:
            return None
        
        children = list(self.root.children.items())
        if top_k is not None:
            children = sorted(children, key=lambda x: x[1].visits, reverse=True)[:top_k]
        else:
            children = sorted(children, key=lambda x: x[1].visits, reverse=True)[:min(8, len(children))]
        actions = [action for action, _ in children]
        visits = np.array([c.visits for _, c in children], dtype=np.float64)
        if temperature > 0:
            logits = visits ** (1.0 / temperature)
            probs = logits / logits.sum()
        else:
            # Deterministic: return most visited
            return actions[0]
        return self._rng.choices(actions, weights=probs, k=1)[0]

    def sample_root_with_prior_blend(
        self, top_k: int = None, prior_weight: float = 0.3
    ) -> Optional[int]:
        """
        Sample from root with a blend of visit counts and prior probabilities.
        
        Args:
            top_k: Number of top actions to consider.
            prior_weight: Weight for prior probability in blend (0.0-1.0).
        
        Returns:
            Sampled action index, or None if root has no children.
        """
        if not self.root or not self.root.children:
            return None
        
        children = list(self.root.children.items())
        if top_k is not None:
            children = sorted(children, key=lambda x: x[1].visits, reverse=True)[:top_k]
        else:
            children = sorted(children, key=lambda x: x[1].visits, reverse=True)[:min(8, len(children))]
        
        visits = np.array([c.visits for _, c in children], dtype=np.float64)
        priors = np.array([c.prior_prob for _, c in children], dtype=np.float64)
        actions = [a for a, _ in children]
        
        visit_probs = visits / (visits.sum() + 1e-8)
        prior_probs = priors / (priors.sum() + 1e-8)
        blended_probs = (1 - prior_weight) * visit_probs + prior_weight * prior_probs
        blended_probs /= blended_probs.sum()
        
        return self._rng.choices(actions, weights=blended_probs, k=1)[0]

    def _uct_iteration(self, state: np.ndarray, legal_actions: List[int]):
        """Perform one UCT iteration: selection -> expansion -> simulation -> backprop."""
        path = [self.root]
        node = self.root
        depth = 0
        
        # Selection: traverse tree using UCB1
        while node.children and depth < self.max_depth:
            best_score = float('-inf')
            best_child = None
            for action, child in node.children.items():
                if child.visits == 0:
                    ucb_score = float('inf')  # Unexplored nodes are preferred
                else:
                    exploitation = child.q_value
                    exploration = self.C * math.sqrt(math.log(node.visits) / child.visits)
                    prior = child.prior_prob
                    ucb_score = exploitation + exploration + 0.5 * prior
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_child = child
            if best_child is None:
                break
            node = best_child
            path.append(node)
            depth += 1
        
        # Expansion: add new child if not at max depth
        if depth < self.max_depth and node.children:
            action = max(node.children.keys())
            new_action = legal_actions[self._rng.randint(0, len(legal_actions) - 1)] if legal_actions else 0
            for a in legal_actions:
                if a not in node.children:
                    new_action = a
                    break
            state_hash = node.state_hash + new_action * 7919
            child_node = MCTSNode(
                state_hash=state_hash,
                parent=node,
                action=new_action,
                prior_prob=0.1 + self._rng.random() * 0.1,
                depth=depth + 1,
            )
            node.children[new_action] = child_node
            path.append(child_node)
        elif node.children:
            # Select from existing children for rollout
            node = self._rng.choice(list(node.children.values()))
            path.append(node)
        
        # Simulation: quick rollout
        total_reward = 0.0
        rollout_state = state.copy()
        discount = 1.0
        
        for step in range(self.max_depth - depth):
            action, _, value_estimate = self.policy.get_action(rollout_state, epsilon=0.1)
            if self.opponent and self.opponent.fingerprint.sample_count > 20:
                opp_pred = self.opponent.predict_action({"dist": 0.1})
                reward_mod = 0.5 if opp_pred["action"] == "shield_back" else 1.0
            else:
                reward_mod = 1.0
            reward = value_estimate * reward_mod
            rollout_state = self._apply_state_transition(rollout_state, action)
            total_reward += discount * reward
            discount *= self.gamma
            if step > 15:
                break
        
        # Backpropagation
        for node in reversed(path):
            node.visits += 1
            node.q_value = (node.q_value * (node.visits - 1) + total_reward) / node.visits

    def _apply_state_transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Simple approximate state transition for MCTS simulation."""
        new_state = state.copy()
        
        # Very rough state dynamics
        if action == 1:  # move_left
            new_state[3] = max(0.01, state[3] - 0.01)  # player_cx
        elif action == 2:  # move_right
            new_state[3] = min(0.99, state[3] + 0.01)
        elif action in (3, 4, 5):  # jump variants
            new_state[4] = min(0.95, state[4] + 0.02)
            new_state[9] = 1.0  # airborne flag
        elif action in (6, 7, 8):  # attacks
            new_state[12] = 1.0  # attack flag (temporarily)
        
        # Decay attack flag
        new_state[12] = state[12] * 0.5
        
        # Update distance (simplified)
        new_state[0] = abs(state[1] - state[3]) + abs(state[2] - state[4])
        
        return new_state

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class SelfPlayEngine:
    """
    Self-play training: AI plays against itself, evolving over time.
    
    Features:
    - ELO rating system for skill tracking
    - Copy opponent strategy (best practices)
    - Mutation-based evolution
    - Symmetric game support
    """
    
    def __init__(self, policy_net: PPOPolicyNetwork):
        self.policy = policy_net
        self.elo = 1500.0
        self.opponent_elo = 1500.0
        
        # Evolution history
        self.best_elo = 1500.0
        self.evolution_history: deque = deque(maxlen=50)
        
        # Copy of best agent for evolution
        self._best_agent = copy.deepcopy(policy_net)
        
        # Game history
        self.game_results: deque = deque(maxlen=100)
        self._game_count = 0
    
    def play_self_game(self, max_frames: int = 1800) -> Dict[str, Any]:
        """
        Simulate a self-play game between current agent and evolving opponent.
        
        Returns:
            {"winner": str, "frames": int, "agent_score": float, "opp_score": float}
        """
        # Create opponent (copy of current with potential mutations)
        opponent = copy.deepcopy(self.policy)
        
        # Mutation probability increases with ELO difference
        if self.elo - self.opponent_elo > 200:
            opponent.mutate(rate=0.05, strength=0.2)
        
        # State for both "players"
        state = np.zeros(STATE_DIM, dtype=np.float32)
        agent_score = 0.0
        opp_score = 0.0
        
        for frame in range(max_frames):
            # Agent's turn
            action_a, _, _ = self.policy.get_action(state, epsilon=0.05)
            
            # Opponent's turn (can see mirrored state)
            mirrored_state = state.copy()
            mirrored_state[3] = 1.0 - mirrored_state[3]  # Mirror player x
            mirrored_state[1] = 1.0 - mirrored_state[1]  # Mirror enemy x
            action_o, _, _ = opponent.get_action(mirrored_state, epsilon=0.05)
            
            # Simulate interaction (simplified)
            reward_a, reward_o = self._simulate_frame(state, action_a, action_o)
            agent_score += reward_a
            opp_score += reward_o
            
            # Update state
            state = self._state_transition(state, action_a)
            
            # Check for game end
            if state[13] > 0.8:  # blast_zone_danger for agent
                return {"winner": "opponent", "frames": frame, "agent_score": agent_score, "opp_score": opp_score}
            if self._opponent_blast_zone_danger(state) > 0.8:
                return {"winner": "agent", "frames": frame, "agent_score": agent_score, "opp_score": opp_score}
        
        return {"winner": "draw", "frames": max_frames, "agent_score": agent_score, "opp_score": opp_score}
    
    def _simulate_frame(self, state, action_a, action_o) -> Tuple[float, float]:
        """Simulate one frame of interaction. Returns rewards for each agent."""
        reward_a = 0.0
        reward_o = 0.0
        
        # Close range = more damage potential
        dist = state[0]
        
        if action_a in (6, 7, 8):  # attacks
            if dist < HIT_CLOSE:
                reward_a += 1.0
            elif dist < HIT_MEDIUM:
                reward_a += 0.3
        
        if action_o in (6, 7, 8):
            mirrored_dist = abs(state[1] - (1.0 - state[3]))
            if mirrored_dist < HIT_CLOSE:
                reward_o += 1.0
            elif mirrored_dist < HIT_MEDIUM:
                reward_o += 0.3
        
        # Defense bonus
        if action_a in (9, 10, 11):  # shield/dash
            reward_a += 0.1
        
        # Staying in blast zone penalty
        if state[13] > 0.5:
            reward_a -= 0.5
        
        return reward_a, reward_o
    
    def _state_transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Simple state transition for self-play simulation."""
        new_state = state.copy()
        
        if action == 1:  # move left
            new_state[3] = max(0.02, state[3] - 0.008)
        elif action == 2:  # move right
            new_state[3] = min(0.98, state[3] + 0.008)
        elif action in (3, 4, 5):  # jump
            new_state[4] = min(0.9, state[4] + 0.015)
            new_state[9] = 1.0
        
        # Decay airborne
        new_state[9] *= 0.95
        
        # Update distance (enemy x simplified)
        new_state[0] = abs(state[1] - state[3])
        
        # Update blast zone danger
        new_state[13] = max(0.0, min(1.0,
            max(0, state[3] - STAGE_L) / 0.05 if state[3] < STAGE_L + 0.05 else
            max(0, STAGE_R - state[3]) / 0.05 if state[3] > STAGE_R - 0.05 else
            max(0, state[4] - STAGE_B) / 0.05 if state[4] > STAGE_B - 0.05 else 0.0
        ))
        
        return new_state
    
    def _opponent_blast_zone_danger(self, state: np.ndarray) -> float:
        """Simulated opponent blast zone danger (mirrored)."""
        opp_x = 1.0 - state[3]
        return max(0.0, min(1.0,
            max(0, opp_x - STAGE_L) / 0.05 if opp_x < STAGE_L + 0.05 else
            max(0, STAGE_R - opp_x) / 0.05 if opp_x > STAGE_R - 0.05 else 0.0
        ))
    
    def update_elo(self, result: str, frames: int):
        """Update ELO ratings based on game result."""
        K = 32
        
        expected_agent = 1.0 / (1.0 + 10 ** ((self.opponent_elo - self.elo) / 400))
        expected_opp = 1.0 - expected_agent
        
        if result == "agent":
            actual_agent = 1.0
        elif result == "opponent":
            actual_agent = 0.0
        else:
            actual_agent = 0.5
        
        self.elo += K * (actual_agent - expected_agent)
        self.opponent_elo += K * ((1 - actual_agent) - expected_opp)
        
        self.game_results.append(result)
        self._game_count += 1
        
        # Track best agent
        if self.elo > self.best_elo:
            self.best_elo = self.elo
            self._best_agent = copy.deepcopy(self.policy)
        
        # Store evolution data
        self.evolution_history.append({
            "game": self._game_count,
            "elo": self.elo,
            "opponent_elo": self.opponent_elo,
            "result": result,
        })
    
    def evolve_if_needed(self, games_played: int, threshold_elo: int = 1600):
        """Evolve opponent if agent has reached skill threshold."""
        if self.elo > threshold_elo and games_played % 50 == 0:
            # Copy best agent as new opponent
            self.policy.copy_from(self._best_agent)
            logger.info("[SELFPLAY] Evolved: ELO=%.0f, best=%.0f", self.elo, self.best_elo)
    
    def get_stats(self) -> dict:
        recent_wins = sum(1 for r in self.game_results if r == "agent")
        recent_games = len(self.game_results)
        return {
            "elo": self.elo,
            "opponent_elo": self.opponent_elo,
            "best_elo": self.best_elo,
            "games_played": self._game_count,
            "recent_win_rate": recent_wins / max(1, recent_games),
        }




# ══════════════════════════════════════════════════════════════════════════════
# SELF-PLAY EXTENSIONS: Nash Equilibrium Opponent Selection
# ══════════════════════════════════════════════════════════════════════════════

class RegretMinimizer:
    """
    Counterfactual Regret Minimization (CFR) for opponent archetype selection.
    Tracks regrets for selecting each opponent archetype based on observed
    agent performance against them, then uses regret-matching to select
    opponents that exploit the agent's weaknesses.
    """

    def __init__(self, num_archetypes: int, epsilon: float = 0.0):
        self.num_archetypes = num_archetypes
        self.epsilon = epsilon  # exploration probability
        self.strategy: np.ndarray = np.ones(num_archetypes) / num_archetypes
        self.cumulative_regrets: np.ndarray = np.zeros(num_archetypes)
        self.strategy_sums: np.ndarray = np.zeros(num_archetypes)
        self.iterations: int = 0

    def _regret_matching(self) -> np.ndarray:
        """Compute regret-matching strategy from cumulative regrets."""
        positive_regrets = np.maximum(self.cumulative_regrets, 0.0)
        total_positive = positive_regrets.sum()
        if total_positive > 1e-10:
            return positive_regrets / total_positive
        # Uniform on tie or all-negative
        return np.ones(self.num_archetypes) / self.num_archetypes

    def update(self, archetype_index: int, utility: float):
        """
        Update regrets after playing against a specific archetype.

        Args:
            archetype_index: Which archetype was selected this iteration.
            utility: Agent's normalized reward against this archetype (0=loss, 0.5=draw, 1=win).
        """
        expected_utility = np.dot(self.strategy, np.array([
            utility if i == archetype_index else 0.0
            for i in range(self.num_archetypes)
        ]))
        # Regret = how much better playing the chosen archetype was vs expected
        regret = utility - expected_utility
        self.cumulative_regrets[archetype_index] += regret
        self.strategy = self._regret_matching()
        self.strategy_sums += self.strategy
        self.iterations += 1

    def get_average_strategy(self) -> np.ndarray:
        """Return the average strategy over all iterations (Nash equilibrium approximation)."""
        total = self.strategy_sums.sum()
        if total > 1e-10:
            return self.strategy_sums / total
        return np.ones(self.num_archetypes) / self.num_archetypes

    def sample(self) -> int:
        """Sample an archetype using epsilon-greedy regret-matching."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_archetypes - 1)
        return random.choices(
            range(self.num_archetypes),
            weights=self.strategy,
            k=1
        )[0]


class NashOpponentSelector:
    """
    Nash equilibrium-inspired opponent selection using regret minimization.

    Maintains a pool of opponent archetypes with distinct playstyles (aggressive,
    defensive, evasive, combo-focused). Uses CFR to adaptively select opponents
    that maximize exploitation of the agent's discovered weaknesses.

    Archetypes are evolved with game-specific parameter tweaks to create
    diverse, specialized opponents for targeted training.
    """

    # Predefined opponent archetypes with distinct behavioral profiles
    ARCHETYPE_DEFINITIONS = [
        {
            "name": "aggressive_rushdown",
            "epsilon": 0.02,
            "mutation_strength": 0.25,
            "mutation_rate": 0.08,
            "approach_bias": 0.9,   # prefers closing distance
            "defense_bias": 0.1,    # rarely shields
            "combo_bias": 0.4,      # medium combo tendency
            "stage_ctrl": 0.7,      # moderate stage control
        },
        {
            "name": "defensive_shield",
            "epsilon": 0.01,
            "mutation_strength": 0.15,
            "mutation_rate": 0.05,
            "approach_bias": 0.2,
            "defense_bias": 0.9,
            "combo_bias": 0.2,
            "stage_ctrl": 0.8,
        },
        {
            "name": "evasive_counter",
            "epsilon": 0.05,
            "mutation_strength": 0.3,
            "mutation_rate": 0.1,
            "approach_bias": 0.3,
            "defense_bias": 0.6,
            "combo_bias": 0.3,
            "stage_ctrl": 0.6,
        },
        {
            "name": "combo_optimizer",
            "epsilon": 0.02,
            "mutation_strength": 0.2,
            "mutation_rate": 0.07,
            "approach_bias": 0.5,
            "defense_bias": 0.4,
            "combo_bias": 0.9,
            "stage_ctrl": 0.5,
        },
        {
            "name": "stage_control",
            "epsilon": 0.01,
            "mutation_strength": 0.18,
            "mutation_rate": 0.06,
            "approach_bias": 0.6,
            "defense_bias": 0.3,
            "combo_bias": 0.3,
            "stage_ctrl": 0.95,
        },
        {
            "name": "adaptive_balancer",
            "epsilon": 0.03,
            "mutation_strength": 0.22,
            "mutation_rate": 0.08,
            "approach_bias": 0.5,
            "defense_bias": 0.5,
            "combo_bias": 0.5,
            "stage_ctrl": 0.7,
        },
    ]

    def __init__(self, num_archetypes: int = 6, epsilon: float = 0.1):
        self.num_archetypes = num_archetypes
        self.regret_minimizer = RegretMinimizer(num_archetypes, epsilon=epsilon)
        self._archetype_stats: Dict[int, Dict[str, Any]] = {
            i: {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            for i in range(num_archetypes)
        }
        self._weakness_cache: Dict[int, float] = {}  # archetype -> weakness score
        self._exploit_threshold: float = 0.3  # minimum win-rate gap to consider weakness

    def select_opponent_archetype(self) -> int:
        """Select opponent archetype using Nash equilibrium strategy."""
        return self.regret_minimizer.sample()

    def create_opponent_from_archetype(
        self,
        base_policy: 'PPOPolicyNetwork',
        archetype_index: int
    ) -> 'PPOPolicyNetwork':
        """
        Create a specialized opponent from a base policy and archetype profile.

        Args:
            base_policy: The policy network to clone and specialize.
            archetype_index: Index of the archetype to specialize for.

        Returns:
            A mutated copy of base_policy tuned to the archetype's style.
        """
        opponent = copy.deepcopy(base_policy)
        archetype = self.ARCHETYPE_DEFINITIONS[archetype_index]

        # Apply archetype-specific mutation
        opponent.mutate(
            rate=archetype["mutation_rate"],
            strength=archetype["mutation_strength"]
        )

        # Tag the opponent for tracking
        opponent._archetype_id = archetype_index
        opponent._archetype_name = archetype["name"]
        return opponent

    def record_result(
        self,
        archetype_index: int,
        result: str,
        agent_utility: float
    ):
        """
        Record game result and update regrets for the selected archetype.

        Args:
            archetype_index: Which archetype was used this game.
            result: "agent", "opponent", or "draw".
            agent_utility: Normalized agent reward [0, 1].
        """
        self._archetype_stats[archetype_index]["games"] += 1
        if result == "agent":
            self._archetype_stats[archetype_index]["wins"] += 1
        elif result == "opponent":
            self._archetype_stats[archetype_index]["losses"] += 1
        else:
            self._archetype_stats[archetype_index]["draws"] += 1

        self.regret_minimizer.update(archetype_index, agent_utility)
        self._update_weakness_cache(archetype_index)

    def _update_weakness_cache(self, archetype_index: int):
        """Update cached weakness score for an archetype based on recent results."""
        stats = self._archetype_stats[archetype_index]
        total = stats["games"]
        if total < 3:
            self._weakness_cache[archetype_index] = 0.0
            return
        # Weakness = loss rate relative to the agent's overall performance
        loss_rate = stats["losses"] / total
        avg_win_rate = sum(s["wins"] / max(1, s["games"]) for s in self._archetype_stats.values()) / self.num_archetypes
        self._weakness_cache[archetype_index] = max(0.0, loss_rate - avg_win_rate)

    def get_most_exploitative_archetype(self) -> int:
        """
        Return the archetype with the highest regret (most exploitative for agent).
        """
        regrets = self.regret_minimizer.cumulative_regrets
        return int(np.argmax(regrets))

    def get_archetype_stats(self) -> Dict[str, Any]:
        """Return per-archetype statistics and Nash strategy."""
        avg_strategy = self.regret_minimizer.get_average_strategy()
        return {
            "archetypes": [
                {
                    "index": i,
                    "name": self.ARCHETYPE_DEFINITIONS[i]["name"],
                    "stats": self._archetype_stats[i],
                    "weakness_score": self._weakness_cache.get(i, 0.0),
                    "nash_probability": avg_strategy[i],
                    "cumulative_regret": self.regret_minimizer.cumulative_regrets[i],
                }
                for i in range(self.num_archetypes)
            ],
            "most_exploitative": self.get_most_exploitative_archetype(),
            "total_games": sum(s["games"] for s in self._archetype_stats.values()),
        }

    def get_adversarial_batch(
        self,
        base_policy: 'PPOPolicyNetwork',
        batch_size: int,
        exploit_weight: float = 0.6
    ) -> List[Tuple['PPOPolicyNetwork', int]]:
        """
        Generate a batch of diverse opponents biased toward exploiting agent weaknesses.

        Args:
            base_policy: Base policy to clone.
            batch_size: Number of opponents to generate.
            exploit_weight: Weight of exploitability vs diversity (0=diversity, 1=exploit).

        Returns:
            List of (opponent_policy, archetype_index) tuples.
        """
        opponents = []
        nash_strat = self.regret_minimizer.get_average_strategy()
        most_exploited = self.get_most_exploitative_archetype()

        for _ in range(batch_size):
            if random.random() < exploit_weight and self._weakness_cache:
                # Bias toward exploitatively strong archetypes
                probs = nash_strat.copy()
                probs[most_exploited] *= 2.0
                probs /= probs.sum()
                archetype = random.choices(range(self.num_archetypes), weights=probs, k=1)[0]
            else:
                archetype = random.choices(range(self.num_archetypes), weights=nash_strat, k=1)[0]

            opponent = self.create_opponent_from_archetype(base_policy, archetype)
            opponents.append((opponent, archetype))

        return opponents


class SelfPlayEngineNash(SelfPlayEngine):
    """
    SelfPlayEngine extended with Nash equilibrium opponent selection.

    Uses RegretMinimizer and NashOpponentSelector to maintain a pool of
    specialized archetypes and select opponents that exploit agent weaknesses
    via regret-matching — approximating a Nash equilibrium over the opponent
    space rather than using random selection.
    """

    def __init__(self, policy_net: 'PPOPolicyNetwork', nash_epsilon: float = 0.1):
        super().__init__(policy_net)
        self.nash_selector = NashOpponentSelector(
            num_archetypes=len(NashOpponentSelector.ARCHETYPE_DEFINITIONS),
            epsilon=nash_epsilon
        )
        self._archetype_history: deque = deque(maxlen=100)

    def play_self_game(self, max_frames: int = 1800, use_nash: bool = True) -> Dict[str, Any]:
        """
        Play a self-play game, optionally using Nash equilibrium opponent selection.

        Args:
            max_frames: Maximum frames before draw.
            use_nash: If True, select opponent via Nash selector; else use default.

        Returns:
            Game result dict with winner, frames, scores, and archetype info.
        """
        if use_nash:
            archetype_index = self.nash_selector.select_opponent_archetype()
            opponent = self.nash_selector.create_opponent_from_archetype(
                self.policy, archetype_index
            )
        else:
            archetype_index = -1
            opponent = copy.deepcopy(self.policy)
            if self.elo - self.opponent_elo > 200:
                opponent.mutate(rate=0.05, strength=0.2)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        agent_score = 0.0
        opp_score = 0.0

        for frame in range(max_frames):
            action_a, _, _ = self.policy.get_action(state, epsilon=0.05)

            mirrored_state = state.copy()
            mirrored_state[3] = 1.0 - mirrored_state[3]
            mirrored_state[1] = 1.0 - mirrored_state[1]
            action_o, _, _ = opponent.get_action(mirrored_state, epsilon=0.05)

            reward_a, reward_o = self._simulate_frame(state, action_a, action_o)
            agent_score += reward_a
            opp_score += reward_o

            state = self._state_transition(state, action_a)

            if state[13] > 0.8:
                result = "opponent"
                break
            if self._opponent_blast_zone_danger(state) > 0.8:
                result = "agent"
                break
        else:
            result = "draw"

        result_dict = {
            "winner": result,
            "frames": frame if result != "draw" else max_frames,
            "agent_score": agent_score,
            "opp_score": opp_score,
            "archetype_index": archetype_index,
            "archetype_name": getattr(opponent, "_archetype_name", "default"),
        }

        # Record for Nash update
        if use_nash:
            agent_utility = 1.0 if result == "agent" else (0.5 if result == "draw" else 0.0)
            self.nash_selector.record_result(archetype_index, result, agent_utility)
            self._archetype_history.append(archetype_index)
            # Sync ELO tracking with base engine
            self.update_elo(result, result_dict["frames"])

        return result_dict

    def get_stats(self) -> dict:
        """Return extended stats including Nash equilibrium information."""
        base_stats = super().get_stats()
        nash_stats = self.nash_selector.get_archetype_stats()
        return {
            **base_stats,
            "nash_stats": nash_stats,
            "archetype_distribution": dict(Counter(self._archetype_history)),
        }

    def get_nash_strategy(self) -> Dict[str, Any]:
        """Return the current Nash equilibrium strategy over archetypes."""
        avg_strat = self.nash_selector.regret_minimizer.get_average_strategy()
        return {
            "strategy": avg_strat.tolist(),
            "most_exploitative": self.nash_selector.get_most_exploitative_archetype(),
            "iterations": self.nash_selector.regret_minimizer.iterations,
        }
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CURRICULUM LEARNING (Wave 5)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumStage:
    name: str
    elo_min: int
    elo_max: int
    focus: str
    epsilon_start: float
    epsilon_end: float
    batch_size: int
    epochs: int
    reward_multipliers: Dict[str, float]


CURRICULUM = [
    CurriculumStage("lab_chase", 1000, 1200, "approach",
                    epsilon_start=0.5, epsilon_end=0.2,
                    batch_size=32, epochs=5,
                    reward_multipliers={"approach": 3.0, "hit": 1.0}),
    CurriculumStage("lab_combos", 1200, 1400, "combo",
                    epsilon_start=0.3, epsilon_end=0.1,
                    batch_size=48, epochs=8,
                    reward_multipliers={"hit": 3.0, "combo": 5.0, "approach": 0.5}),
    CurriculumStage("lab_defense", 1400, 1600, "defense",
                    epsilon_start=0.2, epsilon_end=0.05,
                    batch_size=64, epochs=10,
                    reward_multipliers={"dodge": 5.0, "shield": 4.0, "approach": 0.5}),
    CurriculumStage("lab_ledge", 1600, 1800, "ledge",
                    epsilon_start=0.15, epsilon_end=0.03,
                    batch_size=64, epochs=12,
                    reward_multipliers={"edge_guard": 8.0, "recovery": 6.0}),
    CurriculumStage("lab_reads", 1800, 2000, "prediction",
                    epsilon_start=0.1, epsilon_end=0.02,
                    batch_size=64, epochs=15,
                    reward_multipliers={"counter": 10.0, "read": 8.0}),
    CurriculumStage("ranked", 2000, 9999, "adaptation",
                    epsilon_start=0.05, epsilon_end=0.01,
                    batch_size=64, epochs=20,
                    reward_multipliers={"win": 20.0, "hit": 2.0, "combo": 3.0}),
]


class CurriculumManager:
    """Manages curriculum learning from beginner to master level."""
    def __init__(self, policy_net: PPOPolicyNetwork, elo: float = 1000.0):
        self.policy = policy_net
        self.elo = elo
        self.current_stage_idx = 0
        self.games_in_stage = 0
        self.stage_mastery = 0.0
        # Per-stage tracking
        self.stage_stats: Dict[str, Dict] = defaultdict(lambda: {
            "games": 0, "wins": 0, "losses": 0, "avg_reward": 0.0
        })
        # Overall progress
        self.total_games = 0
        self.total_wins = 0
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage based on ELO."""
        for i, stage in enumerate(CURRICULUM):
            if stage.elo_min <= self.elo < stage.elo_max:
                if i != self.current_stage_idx:
                    logger.info("[CURRICULUM] Advancing to stage: %s", stage.name)
                    self.current_stage_idx = i
                    self.games_in_stage = 0
                return stage
        return CURRICULUM[-1]
    def get_epsilon(self) -> float:
        """Get exploration rate for current stage and progress."""
        stage = self.get_current_stage()
        progress = min(1.0, self.games_in_stage / 100)
        return stage.epsilon_start + (stage.epsilon_end - stage.epsilon_start) * progress
    def record_game(self, won: bool, reward: float, stage_focus: str):
        """Record game result and update curriculum progress."""
        self.total_games += 1
        if won:
            self.total_wins += 1
        stage = self.get_current_stage()
        self.stage_stats[stage.name]["games"] += 1
        if won:
            self.stage_stats[stage.name]["wins"] += 1
            # Win = ELO increase
            self.elo = min(9999, self.elo + 15)
        else:
            self.stage_stats[stage.name]["losses"] += 1
            self.elo = max(1000, self.elo - 10)
        # Update reward tracking
        prev_avg = self.stage_stats[stage.name]["avg_reward"]
        n = self.stage_stats[stage.name]["games"]
        self.stage_stats[stage.name]["avg_reward"] = (prev_avg * (n - 1) + reward) / n
        self.games_in_stage += 1
        # Check if stage is mastered (70% win rate + enough games)
        stats = self.stage_stats[stage.name]
        if stats["games"] >= 50:
            win_rate = stats["wins"] / stats["games"]
            self.stage_mastery = win_rate
            if win_rate >= 0.7:
                # Ready for next stage
                logger.info("[CURRICULUM] Stage %s mastered (%.0f%% win rate). Advancing.",
                           stage.name, win_rate * 100)
    def should_use_mcts(self) -> bool:
        """Use MCTS only at higher skill levels (when fundamentals are solid)."""
        return self.elo >= 1400
    def get_batch_config(self) -> Tuple[int, int]:
        """Get training batch size and epochs for current stage."""
        stage = self.get_current_stage()
        return stage.batch_size, stage.epochs
    def get_reward_multipliers(self) -> Dict[str, float]:
        """Get reward multipliers for current stage focus."""
        stage = self.get_current_stage()
        return stage.reward_multipliers
    def get_summary(self) -> dict:
        """Get curriculum progress summary."""
        stage = self.get_current_stage()
        stats = self.stage_stats.get(stage.name, {})
        return {
            "current_stage": stage.name,
            "elo": self.elo,
            "stage_mastery": f"{self.stage_mastery:.0%}",
            "stage_games": stats.get("games", 0),
            "stage_winrate": f"{stats.get('wins', 0) / max(1, stats.get('games', 1)):.0%}",
            "epsilon": f"{self.get_epsilon():.3f}",
            "total_games": self.total_games,
            "overall_winrate": f"{self.total_wins / max(1, self.total_games):.0%}",
        }
    def get_stage_shaping_config(self) -> Dict[str, float]:
        """Get stage-specific reward shaping configuration."""
        stage = self.get_current_stage()
        # Base reward configuration with movement, combat, defense, positioning
        base_config = {
            # Movement rewards
            "move_toward_enemy": 0.5,
            "move_away_enemy": 0.3,
            "jump": 0.2,
            "dash": 0.3,
            # Combat rewards
            "hit": 1.0,
            "combo": 2.0,
            "kill": 5.0,
            "death": -2.0,
            # Defense rewards
            "dodge": 0.5,
            "shield": 0.5,
            "air_recovery": 0.5,
            # Positioning rewards
            "ledge_control": 0.5,
            "center_control": 0.3,
            "edge_guard": 0.5,
        }
        # Stage-specific adjustments: early stages emphasize movement,
        # later stages emphasize combat effectiveness
        if stage.focus == "approach":
            # Early stage: focus on closing distance
            config = base_config.copy()
            config["move_toward_enemy"] = 3.0
            config["move_away_enemy"] = -0.5  # Penalize retreating early
            config["hit"] = 1.0
            config["combo"] = 0.5
            config["dodge"] = 0.2
        elif stage.focus == "combo":
            # Focus on hit confirmation and combo extension
            config = base_config.copy()
            config["move_toward_enemy"] = 1.5
            config["hit"] = 3.0
            config["combo"] = 5.0
            config["move_away_enemy"] = 0.1
        elif stage.focus == "defense":
            # Focus on defensive options
            config = base_config.copy()
            config["move_toward_enemy"] = 0.5
            config["dodge"] = 5.0
            config["shield"] = 4.0
            config["hit"] = 1.5
            config["combo"] = 1.0
        elif stage.focus == "ledge":
            # Focus on ledge plays and recovery
            config = base_config.copy()
            config["ledge_control"] = 8.0
            config["edge_guard"] = 8.0
            config["air_recovery"] = 6.0
            config["hit"] = 2.0
            config["death"] = -3.0  # Higher death penalty
        elif stage.focus == "prediction":
            # Focus on reads and counter-play
            config = base_config.copy()
            config["dodge"] = 3.0
            config["hit"] = 4.0
            config["combo"] = 4.0
            config["counter"] = 10.0  # Rewarded for reading opponent
        else:  # adaptation
            # Balanced but weighted toward winning
            config = base_config.copy()
            config["hit"] = 2.0
            config["combo"] = 3.0
            config["kill"] = 20.0
            config["death"] = -3.0
        return config
    def shape_reward(
        self,
        raw_reward: float,
        state: dict = None,
        action: int = None,
        hit_result: str = "none"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply stage-specific reward shaping to raw reward.
        Args:
            raw_reward: Base reward from game engine
            state: Current game state dict (optional)
            action: Action taken (optional)
            hit_result: Result of action ("hit", "combo", "counter", "none")
        Returns:
            Tuple of (shaped_reward, reward_breakdown)
        """
        stage = self.get_current_stage()
        config = self.get_stage_shaping_config()
        shaped_reward = raw_reward
        breakdown = {"base": raw_reward}
        # Apply movement shaping if state and action available
        if state is not None and action is not None:
            player = state.get("player", {})
            enemies = state.get("enemies", [])
            movement_bonus = 0.0
            if enemies:
                enemy = enemies[0]
                player_cx = player.get("cx", 0.5)
                enemy_cx = enemy.get("cx", 0.5)
                player_cy = player.get("cy", 0.5)
                enemy_cy = enemy.get("cy", 0.5)
                dx = enemy_cx - player_cx
                # Movement actions: 1=move_left, 2=move_right, 3=jump, 4=crouch, 5=dash
                if action == 1 and dx < 0:  # Moving left toward enemy
                    movement_bonus += config["move_toward_enemy"]
                elif action == 2 and dx > 0:  # Moving right toward enemy
                    movement_bonus += config["move_toward_enemy"]
                elif action == 1 and dx > 0:  # Moving left away from enemy
                    movement_bonus += config["move_away_enemy"]
                elif action == 2 and dx < 0:  # Moving right away from enemy
                    movement_bonus += config["move_away_enemy"]
                if action == 3:  # Jump
                    movement_bonus += config["jump"]
                if action == 5:  # Dash
                    movement_bonus += config["dash"]
            # Shield/dodge actions (9=shield, 10=dodge, 11=air_dodge)
            if action == 9:
                movement_bonus += config["shield"]
            elif action == 10 or action == 11:
                movement_bonus += config["dodge"]
            # Air recovery bonus
            if action == 11 and player.get("cy", 0.5) > 0.6:
                movement_bonus += config["air_recovery"]
            if movement_bonus != 0:
                shaped_reward += movement_bonus
                breakdown["movement"] = movement_bonus
        # Apply hit result shaping
        if hit_result == "hit":
            hit_bonus = config["hit"]
            shaped_reward += hit_bonus
            breakdown["hit"] = hit_bonus
        elif hit_result == "combo":
            combo_bonus = config["combo"]
            shaped_reward += combo_bonus
            breakdown["combo"] = combo_bonus
        elif hit_result == "counter":
            counter_bonus = config.get("counter", 5.0)
            shaped_reward += counter_bonus
            breakdown["counter"] = counter_bonus
        elif hit_result == "kill":
            kill_bonus = config["kill"]
            shaped_reward += kill_bonus
            breakdown["kill"] = kill_bonus
        elif hit_result == "death":
            death_penalty = config["death"]
            shaped_reward += death_penalty
            breakdown["death"] = death_penalty
        # Apply positioning rewards if state available
        if state is not None:
            player = state.get("player", {})
            # Ledge control (player near edge)
            cx = player.get("cx", 0.5)
            if cx < STAGE_L + 0.05 or cx > STAGE_R - 0.05:
                ledge_bonus = config["ledge_control"] * 0.5
                shaped_reward += ledge_bonus
                breakdown["ledge_control"] = ledge_bonus
            # Center control
            center_dist = abs(cx - 0.5)
            if center_dist < 0.2:
                center_bonus = config["center_control"] * 0.5
                shaped_reward += center_bonus
                breakdown["center_control"] = center_bonus
        # Apply stage-specific curriculum multipliers
        stage_multipliers = stage.reward_multipliers
        if hit_result == "hit" and "hit" in stage_multipliers:
            multiplier = stage_multipliers["hit"]
            shaped_reward *= multiplier
            breakdown["curriculum_mult"] = multiplier
        # Normalize reward based on stage progression
        # Early stages get larger rewards to accelerate learning
        stage_progress = min(1.0, self.games_in_stage / 100)
        if self.current_stage_idx < 2:  # First two stages
            shaped_reward *= (1.0 + (1.0 - stage_progress) * 0.5)
            breakdown["stage_normalization"] = "early"
        elif self.current_stage_idx >= 4:  # Later stages
            shaped_reward *= 0.8  # Smaller rewards, focus on winning
            breakdown["stage_normalization"] = "late"
        return shaped_reward, breakdown
    def get_curriculum_stage_index(self) -> int:
        """Get current stage index for external access."""
        return self.current_stage_idx
    def get_stage_name(self) -> str:
        """Get current stage name."""
        return self.get_current_stage().name
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: NEURAL AI — FULL INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class NeuralAI:
    """
    Full neural AI that combines all 5 systems:
    1. PPO Policy Network — core decision making
    2. MCTS Engine — lookahead for critical moments
    3. Opponent Model — adapts to specific opponents
    4. Self-Play Engine — self-improvement
    5. Curriculum Manager — progressive learning
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        load_path: str = None,
    ):
        # Core policy network (PPO)
        self.policy = PPOPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=256,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
        )
        
        # Load pretrained if available
        if load_path and os.path.exists(load_path):
            self.policy.load(load_path)
            logger.info("[NeuralAI] Loaded pretrained policy from: %s", load_path)
        
        # MCTS for critical decisions
        self.mcts = MCTSEngine(
            policy_net=self.policy,
            max_iterations=150,
            max_depth=6,
            exploration_const=1.41,
        )
        
        # Opponent modeling
        self.opponent_model = OpponentModel()
        
        # Self-play engine
        self.selfplay = SelfPlayEngine(policy_net=self.policy)
        
        # Curriculum learning
        self.curriculum = CurriculumManager(policy_net=self.policy)
        
        # State tracking
        self._frame = 0
        self._episode_rewards: deque = deque(maxlen=100)
        self._total_reward = 0.0
        
        # Mode flags
        self._training_mode = True
        self._use_mcts = True
        self._use_opponent_model = True
    
    def choose_action(self, state: dict) -> Tuple[int, str, float]:
        """
        Choose action using the full neural stack.
        
        Returns:
            (action_index, action_name, confidence)
        """
        # Convert game state dict to state vector
        state_vec = self._game_state_to_vector(state)
        
        # Get current exploration rate from curriculum
        epsilon = self.curriculum.get_epsilon()
        
        # Check if this is a critical decision (use MCTS)
        is_critical = self._is_critical_decision(state)
        
        if self._use_mcts and is_critical and self.curriculum.should_use_mcts():
            # Use MCTS for critical decisions
            legal_actions = self._get_legal_actions(state)
            mcts_action = self.mcts.search(state_vec, legal_actions, critical_only=True)
            
            if mcts_action is not None:
                action_idx = mcts_action
                action_name = ACTIONS[action_idx]
                confidence = 0.9
            else:
                action_idx, _, value = self.policy.get_action(state_vec, epsilon)
                action_name = ACTIONS[action_idx]
                confidence = abs(value)
        else:
            # Use policy network directly
            action_idx, log_prob, value = self.policy.get_action(state_vec, epsilon)
            action_name = ACTIONS[action_idx]
            confidence = abs(value)
        
        return action_idx, action_name, confidence
    
    def _game_state_to_vector(self, state: dict) -> np.ndarray:
        """Convert game state dict to 26-feature state vector."""
        player = state.get("player", {})
        enemies = state.get("enemies", [])
        gadgets = state.get("gadgets", [])
        
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        
        # Position features (5)
        if enemies:
            e = enemies[0]
            vec[0] = player.get("cx", 0.5) - e.get("cx", 0.5) if player else 0.5  # enemy_dist
            vec[1] = e.get("cx", 0.5)  # enemy_cx
            vec[2] = e.get("cy", 0.5)  # enemy_cy
        
        vec[3] = player.get("cx", 0.5) if player else 0.5  # player_cx
        vec[4] = player.get("cy", 0.5) if player else 0.5  # player_cy
        
        # Velocity features (4)
        vec[5] = player.get("vx", 0.0) if player else 0.0  # player_vx
        vec[6] = player.get("vy", 0.0) if player else 0.0  # player_vy
        vec[7] = enemies[0].get("vx", 0.0) if enemies else 0.0  # enemy_vx
        vec[8] = enemies[0].get("vy", 0.0) if enemies else 0.0  # enemy_vy
        
        # State flags (6)
        vec[9] = 1.0 if abs(vec[6]) > 0.005 else 0.0  # player_airborne
        cx = player.get("cx", 0.5) if player else 0.5
        vec[10] = 1.0 if (cx < 0.07 or cx > 0.93) else 0.0  # near_edge
        vec[11] = 0.0  # in_combo (would need history)
        vec[12] = 0.0  # enemy_attacking (would need pattern detection)
        
        # Blast zone and weapon (2)
        bz = player.get("blast_zone", {}) if player else {}
        vec[13] = bz.get("danger_level", 0.0) if bz else 0.0  # blast_zone_danger
        
        has_weapon = any(g.get("label") == "weapon" for g in gadgets)
        vec[14] = 1.0 if has_weapon else 0.0  # weapon_nearby
        
        # History features (6) — zeros for now (would need rolling history)
        # prev_dist, prev_enemy_vx, prev_player_vx, dist_change, combo_progress, hit_streak
        vec[15] = vec[0]  # prev_dist (simplified)
        vec[16] = vec[7]  # prev_enemy_vx
        vec[17] = vec[5]  # prev_player_vx
        vec[18] = 0.0  # dist_change (simplified)
        vec[19] = 0.0  # combo_progress
        vec[20] = 0.0  # hit_streak
        
        # Padding (6 more features to reach 26)
        vec[21] = enemies[0].get("conf", 0.5) if enemies else 0.5  # enemy_confidence
        vec[22] = player.get("speed", 0.0) if player else 0.0  # player_speed
        vec[23] = 0.0  # reserved
        vec[24] = 0.0  # reserved
        vec[25] = self.curriculum.get_epsilon()  # exploration (info for network)
        
        return vec
    
    def _is_critical_decision(self, state: dict) -> bool:
        """Determine if current state is critical enough for MCTS."""
        player = state.get("player")
        if not player:
            return False
        
        bz = player.get("blast_zone", {})
        if bz.get("danger_level", 0) > 0.5:
            return True
        
        enemies = state.get("enemies", [])
        if enemies:
            dist = player.get("dist_to_player", 999)
            if dist < HIT_CLOSE:
                return True
        
        return False
    
    def _get_legal_actions(self, state: dict) -> List[int]:
        """Get list of legal action indices."""
        actions = list(range(ACTION_DIM))
        
        player = state.get("player")
        if player:
            cx = player.get("cx", 0.5)
            # Can't move further left at left edge
            if cx < STAGE_L + 0.02:
                actions.remove(1)  # move_left
            # Can't move further right at right edge
            if cx > STAGE_R - 0.02:
                actions.remove(2)  # move_right
        
        return actions
    
    def record_transition(
        self, state: dict, action: int, reward: float,
        next_state: dict, done: bool, hit_result: str = "none"
    ):
        """Record transition for PPO training."""
        state_vec = self._game_state_to_vector(state)
        next_state_vec = self._game_state_to_vector(next_state)
        
        # Apply reward multipliers from curriculum
        multipliers = self.curriculum.get_reward_multipliers()
        if hit_result == "hit":
            reward *= multipliers.get("hit", 1.0)
        elif "combo" in hit_result:
            reward *= multipliers.get("combo", 1.0)
        
        _, log_prob, value = self.policy.get_action(state_vec, epsilon=0.0)
        
        self.policy.store(state_vec, action, reward, next_state_vec, done, log_prob, value)
        
        self._total_reward += reward
        self._episode_rewards.append(reward)
        
        # Update opponent model
        if self._use_opponent_model:
            action_name = ACTIONS[action]
            situation = {"dist": state_vec[0], "player_above": bool(state_vec[9])}
            self.opponent_model.update(self._frame, action_name, situation)
    
    def update(self) -> dict:
        """Run PPO update using collected experience."""
        batch_size, epochs = self.curriculum.get_batch_config()
        stats = self.policy.update(batch_size=batch_size, epochs=epochs)
        
        if self._frame > 0 and self._frame % 1000 == 0:
            self.policy.save("./model/neural_policy.npz")
        
        return stats
    
    def end_episode(self, won: bool):
        """Called at end of episode."""
        reward = 20.0 if won else -10.0
        self.curriculum.record_game(won, self._total_reward, self.curriculum.get_current_stage().focus)
        self._total_reward = 0.0
    
    def enable_training(self, enabled: bool):
        """Enable/disable training mode."""
        self._training_mode = enabled
    
    def get_stats(self) -> dict:
        """Get comprehensive AI statistics."""
        return {
            "avg_loss": self.policy._avg_loss,
            "update_count": self.policy._update_count,
            "frame": self._frame,
            "memory_size": len(self.policy.memory),
            "curriculum": self.curriculum.get_summary(),
            "selfplay": self.selfplay.get_stats(),
            "opponent_model": self.opponent_model.get_fingerprint_summary(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_neural_ai(
    neural_ai: NeuralAI,
    total_episodes: int = 10000,
    selfplay_games: int = 100,
    save_path: str = "./model/neural_policy.npz",
):
    """Full training loop with all 5 systems."""
    
    logger.info("[TRAIN] Starting neural AI training...")
    logger.info("[TRAIN] Episodes: %d | Self-play games: %d", total_episodes, selfplay_games)
    
    for episode in range(total_episodes):
        # Run self-play games periodically
        if episode % 100 == 0 and selfplay_games > 0:
            for _ in range(min(selfplay_games, 20)):
                result = neural_ai.selfplay.play_self_game()
                neural_ai.selfplay.update_elo(result["winner"], result["frames"])
            
            neural_ai.selfplay.evolve_if_needed(episode)
        
        # Collect experience (simulated environment)
        # In real usage, this would be connected to the actual game
        total_reward = 0.0
        
        for step in range(200):  # ~3 second episodes
            # Simulate state
            state_vec = np.random.randn(STATE_DIM).astype(np.float32) * 0.1
            state_vec[0] = random.random() * 0.3  # dist
            
            # Get action
            action_idx, action_name, conf = neural_ai.choose_action({"player": {}, "enemies": [], "gadgets": []})
            
            # Simulate reward
            reward = random.choice([0.0, 1.0, -1.0, 2.0, -2.0])
            total_reward += reward
            
            # Store transition
            next_state_vec = state_vec + np.random.randn(STATE_DIM) * 0.05
            done = step == 199
            
            neural_ai.record_transition(
                {"player": {}, "enemies": [], "gadgets": []},
                action_idx, reward,
                {"player": {}, "enemies": [], "gadgets": []},
                done, "none"
            )
        
        # PPO update
        if episode % 10 == 0:
            stats = neural_ai.update()
            logger.info(
                "[TRAIN] Episode %d | Reward: %.1f | Memory: %d | PolicyLoss: %.4f | ELO: %.0f | Stage: %s",
                episode, total_reward, len(neural_ai.policy.memory),
                stats.get("policy_loss", 0), neural_ai.curriculum.elo,
                neural_ai.curriculum.get_current_stage().name
            )
        
        # End episode tracking
        won = total_reward > 50
        neural_ai.end_episode(won)
        
        # Save checkpoint
        if episode % 500 == 0:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            neural_ai.policy.save(save_path)
            logger.info("[TRAIN] Saved checkpoint: %s", save_path)
    
    logger.info("[TRAIN] Training complete!")
    neural_ai.policy.save(save_path)
    return neural_ai


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Brawlhalla Neural AI Training")
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--load", type=str, default=None, help="Load pretrained policy")
    parser.add_argument("--save", type=str, default="./model/neural_policy.npz")
    parser.add_argument("--selfplay-games", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if args.mode == "train":
        neural_ai = NeuralAI(load_path=args.load)
        neural_ai.policy.shared.weights[0]  # Touch to ensure initialized
        train_neural_ai(
            neural_ai,
            total_episodes=args.episodes,
            selfplay_games=args.selfplay_games,
            save_path=args.save,
        )
    
    elif args.mode == "test":
        neural_ai = NeuralAI(load_path=args.load or args.save)
        
        # Test: run a few simulated decisions
        for i in range(10):
            state = {
                "player": {"cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
                           "blast_zone": {"danger_level": 0.0}},
                "enemies": [{"cx": 0.45, "cy": 0.5, "vx": 0.0, "vy": 0.0, "conf": 0.9}],
                "gadgets": [],
            }
            action_idx, action_name, conf = neural_ai.choose_action(state)
            value = neural_ai.policy.get_value(neural_ai._game_state_to_vector(state))
            print(f"[TEST] State {i}: action={action_name} (conf={conf:.2f}, value={value:.3f})")
    
    elif args.mode == "eval":
        neural_ai = NeuralAI(load_path=args.load or args.save)
        print("[EVAL] Neural AI Statistics:")
        print(json.dumps(neural_ai.get_stats(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
# ══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Constants
    "ACTIONS",
    "ACTION_DIM",
    "STATE_DIM",
    "STAGE_L",
    "STAGE_R",
    "STAGE_B",
    "HIT_CLOSE",
    "HIT_MEDIUM",
    "HIT_FAR",
    "HIT_SPECIAL",
    "CURRICULUM",
    # Neural Network Foundation
    "xavier_init",
    "he_init",
    "orthogonal_init",
    "NeuralNetwork",
    # PPO Policy Network
    "PPOPolicyNetwork",
    # Opponent Modeling
    "OpponentFingerprint",
    "OpponentModel",
    # MCTS Engine
    "MCTSNode",
    "MCTSEngine",
    # Self-Play Engine
    "SelfPlayEngine",
    # Curriculum Learning
    "CurriculumStage",
    "CurriculumManager",
    # Neural AI
    "NeuralAI",
    # Training
    "train_neural_ai",
    # CLI
    "main",
]