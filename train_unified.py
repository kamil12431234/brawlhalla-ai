#!/usr/bin/env python3
"""
Brawlhalla AI — Unified Training System

Combines all training components:
- Neural PPO with GAE-Lambda
- MCTS with Transposition Tables
- Self-Play with Population Training
- Curriculum Learning
- PyTorch DQN with Prioritized Replay

Usage:
    python train_unified.py --mode train --episodes 10000
    python train_unified.py --mode eval --model model/neural_policy.npz
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class TrainingConfig:
    """Unified training configuration."""
    
    def __init__(self, **kwargs):
        # Neural PPO settings
        self.ppo_hidden = kwargs.get('ppo_hidden', 256)
        self.ppo_lr = kwargs.get('ppo_lr', 3e-4)
        self.ppo_gamma = kwargs.get('ppo_gamma', 0.99)
        self.ppo_lambda = kwargs.get('ppo_lambda', 0.95)
        self.ppo_clip_epsilon = kwargs.get('ppo_clip_epsilon', 0.2)
        self.ppo_epochs = kwargs.get('ppo_epochs', 10)
        self.ppo_batch_size = kwargs.get('ppo_batch_size', 64)
        
        # MCTS settings
        self.mcts_iterations = kwargs.get('mcts_iterations', 200)
        self.mcts_depth = kwargs.get('mcts_depth', 8)
        self.mcts_exploration = kwargs.get('mcts_exploration', 1.41)
        
        # Self-play settings
        self.selfplay_games = kwargs.get('selfplay_games', 100)
        self.population_size = kwargs.get('population_size', 5)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        
        # Curriculum settings
        self.curriculum_stages = kwargs.get('curriculum_stages', 5)
        self.stage_episodes = kwargs.get('stage_episodes', 1000)
        
        # DQN settings (PyTorch)
        self.dqn_lr = kwargs.get('dqn_lr', 0.001)
        self.dqn_batch_size = kwargs.get('dqn_batch_size', 64)
        self.dqn_memory = kwargs.get('dqn_memory', 50000)
        self.dqn_target_update = kwargs.get('dqn_target_update', 100)
        
        # General
        self.save_freq = kwargs.get('save_freq', 500)
        self.eval_freq = kwargs.get('eval_freq', 100)
        self.device = kwargs.get('device', 'cuda' if _check_cuda() else 'cpu')
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION-BASED TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class PopulationMember:
    """A single agent in the population with its own weights and ELO."""
    
    def __init__(self, policy, elo: float = 1500.0):
        self.policy = policy
        self.elo = elo
        self.games_played = 0
        self.wins = 0
    
    def mutate(self, rate: float = 0.1, strength: float = 0.2):
        """Apply random mutations to policy weights."""
        for network in [self.policy.shared, self.policy.actor, self.policy.critic]:
            for i in range(len(network.weights)):
                mask = np.random.random(network.weights[i].shape) < rate
                noise = np.random.randn(*network.weights[i].shape) * strength
                network.weights[i] += mask * noise
    
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)


class PopulationManager:
    """Manages population of agents for self-play training."""
    
    def __init__(self, base_policy, size: int = 5):
        self.base_policy = base_policy
        self.population: List[PopulationMember] = []
        self._initialize_population(size)
    
    def _initialize_population(self, size: int):
        """Create initial population from base policy."""
        import copy
        for _ in range(size):
            policy = copy.deepcopy(self.base_policy)
            self.population.append(PopulationMember(policy))
    
    def select_opponent(self, exclude_idx: int = None) -> PopulationMember:
        """Select opponent using ELO-based probability."""
        elos = np.array([p.elo for p in self.population])
        if exclude_idx is not None:
            elos = np.delete(elos, exclude_idx)
        
        # Temperature-based selection (higher temp = more exploration)
        probs = np.exp((elos - elos.mean()) / 100)
        probs /= probs.sum()
        
        idx = np.random.choice(len(self.population))
        if exclude_idx is not None and idx >= exclude_idx:
            idx += 1
        
        return self.population[idx]
    
    def update_elo(self, idx: int, new_elo: float):
        """Update member's ELO after match."""
        self.population[idx].elo = new_elo
    
    def get_elite(self, n: int = 3) -> List[PopulationMember]:
        """Get top N agents by ELO."""
        return sorted(self.population, key=lambda x: x.elo, reverse=True)[:n]
    
    def evolve(self, elite: List[PopulationMember], mutations: int = 3):
        """Create new agents from elite members with mutations."""
        import copy
        new_population = []
        
        for member in elite:
            new_population.append(member)  # Keep elite unchanged
        
        for _ in range(mutations):
            parent = elite[np.random.randint(len(elite))]
            new_agent = PopulationMember(copy.deepcopy(parent.policy))
            new_agent.mutate(rate=0.1, strength=0.2)
            new_agent.elo = parent.elo - 100  # Start slightly lower
            new_population.append(new_agent)
        
        self.population = new_population[:len(self.population)]


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedTrainer:
    """Main training orchestrator combining all systems."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._init_components()
    
    def _init_components(self):
        """Initialize all training components."""
        # Import from neural_ai
        from neural_ai import (
            NeuralNetwork, PPOPolicyNetwork, NeuralAI,
            MCTSEngine, OpponentModel, SelfPlayEngine, CurriculumManager
        )
        
        STATE_DIM = 26
        ACTION_DIM = 12
        
        # PPO Policy Network
        self.policy = PPOPolicyNetwork(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden=self.config.ppo_hidden,
            lr=self.config.ppo_lr,
            gamma=self.config.ppo_gamma,
            lam=self.config.ppo_lambda,
            clip_epsilon=self.config.ppo_clip_epsilon,
        )
        
        # Opponent Model
        self.opponent_model = OpponentModel()
        
        # MCTS Engine
        self.mcts = MCTSEngine(
            policy_net=self.policy,
            opponent_model=self.opponent_model,
            max_iterations=self.config.mcts_iterations,
            max_depth=self.config.mcts_depth,
            exploration_const=self.config.mcts_exploration,
        )
        
        # Neural AI (full integration)
        # Neural AI (full integration) - using the NeuralAI class directly
        self.neural_ai = NeuralAI(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
        # Override neural_ai's policy with our custom one
        self.neural_ai.policy = self.policy
        self.neural_ai.mcts = self.mcts
        
        # Self-Play Engine
        self.selfplay = SelfPlayEngine(self.policy)
        
        # Curriculum Manager
        self.curriculum = CurriculumManager(policy_net=self.policy)
        
        logger.info("[UNIFIED] Components initialized")
    
    def train(self, total_episodes: int, save_path: str) -> dict:
        """Run full training loop."""
        logger.info("[UNIFIED] Starting training for %d episodes", total_episodes)
        
        metrics = {
            'episodes': [],
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'elo': [],
            'stage': [],
        }
        
        for episode in range(total_episodes):
            # Curriculum stage progression
            stage = self.curriculum.get_current_stage()
            
            # Run self-play game periodically
            if episode % 50 == 0:
                opponent = self.population.select_opponent()
                result = self.selfplay.play_self_game()
                self.selfplay.update_elo(result['winner'], result['frames'])
                
                # Update population
                for i, member in enumerate(self.population.population):
                    if member.policy is self.selfplay.policy:
                        self.population.update_elo(i, self.selfplay.elo)
                        break
            
            # Collect experience (simulated)
            total_reward = 0.0
            state = np.random.randn(26).astype(np.float32) * 0.1
            
            for step in range(200):
                action_idx, action_name, conf = self.neural_ai.choose_action(
                    {'player': {}, 'enemies': [], 'gadgets': []}
                )
                
                # Simulate transition
                reward = self._simulate_reward(action_idx, state)
                total_reward += reward
                
                next_state = state + np.random.randn(26) * 0.05
                done = step == 199
                
                self.neural_ai.record_transition(
                    {'player': {}, 'enemies': [], 'gadgets': []},
                    action_idx, reward,
                    {'player': {}, 'enemies': [], 'gadgets': []},
                    done, 'none'
                )
                
                state = next_state
            
            # PPO update
            if episode % 10 == 0 and len(self.policy.memory) >= self.config.ppo_batch_size:
                stats = self.neural_ai.update()
                self.neural_ai.selfplay.evolve_if_needed(episode)
            
            # Track metrics
            won = total_reward > 50
            self.neural_ai.end_episode(won)
            
            if episode % 10 == 0:
                stats = self.neural_ai.get_stats()
                metrics['episodes'].append(episode)
                metrics['rewards'].append(total_reward)
                metrics['elo'].append(stats.get('elo', 1500))
                metrics['stage'].append(stage.name)
                
                if 'policy_loss' in stats:
                    metrics['policy_loss'].append(stats['policy_loss'])
                if 'value_loss' in stats:
                    metrics['value_loss'].append(stats['value_loss'])
                
                logger.info(
                    "[TRAIN] Ep %d | Reward: %6.1f | ELO: %6.0f | "
                    "Stage: %s | Memory: %d",
                    episode, total_reward, stats.get('elo', 1500),
                    stage.name, len(self.policy.memory)
                )
            
            # Curriculum progression
            if episode > 0 and episode % self.config.stage_episodes == 0:
                self.curriculum.advance_stage()
                logger.info("[CURRICULUM] Advanced to: %s", 
                           self.curriculum.get_current_stage().name)
            
            # Save checkpoint
            if episode % self.config.save_freq == 0 and episode > 0:
                self.save_checkpoint(save_path.replace('.npz', f'_ep{episode}.npz'))
                logger.info("[SAVE] Checkpoint saved: %s", save_path)
            
            # Periodic evaluation
            if episode % self.config.eval_freq == 0 and episode > 0:
                eval_result = self.evaluate_agent(num_episodes=10)
                logger.info("[EVAL] Score: %.2f | Win rate: %.1f%%",
                           eval_result['mean_score'], eval_result['win_rate'] * 100)
        
        # Final save
        self.save_checkpoint(save_path)
        
        return metrics
    
    def _simulate_reward(self, action_idx: int, state: np.ndarray) -> float:
        """Simulate reward for action given state."""
        dist = state[0] if len(state) > 0 else 0.1
        
        # Attack rewards
        if action_idx in (6, 7, 8):  # attack actions
            if dist < 0.07:
                return 2.0
            elif dist < 0.12:
                return 0.5
        
        # Movement rewards
        if action_idx in (1, 2):  # move left/right
            return 0.1
        
        # Defensive rewards
        if action_idx in (9, 10, 11):  # shield/dodge
            return 0.2
        
        return 0.0
    
    def evaluate_agent(self, num_episodes: int = 20) -> dict:
        """Evaluate agent performance."""
        scores = []
        wins = 0
        
        for _ in range(num_episodes):
            total_reward = 0.0
            
            for _ in range(200):
                state = np.random.randn(26).astype(np.float32) * 0.1
                action_idx, _, _ = self.neural_ai.choose_action(
                    {'player': {}, 'enemies': [], 'gadgets': []}
                )
                reward = self._simulate_reward(action_idx, state)
                total_reward += reward
            
            scores.append(total_reward)
            if total_reward > 50:
                wins += 1
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'win_rate': wins / num_episodes,
        }
    
    def save_checkpoint(self, path: str):
        """Save full training state."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        checkpoint = {
            'policy': self.policy,
            'neural_ai': self.neural_ai,
            'selfplay_elo': self.selfplay.elo,
            'curriculum_stage': self.curriculum.get_current_stage().name,
            'population_elos': [p.elo for p in self.population.population],
        }
        
        # Save policy weights
        self.policy.save(path)
        
        logger.info("[CHECKPOINT] Saved to: %s", path)
    
    def load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        self.policy.load(path)
        logger.info("[CHECKPOINT] Loaded from: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Brawlhalla AI Unified Training System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'eval', 'benchmark'],
                       default='train', help='Training mode')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=10000, help='Total episodes')
    parser.add_argument('--save-path', default='model/neural_policy.npz',
                       help='Model save path')
    parser.add_argument('--load-path', default=None, help='Model load path')
    
    # PPO settings
    parser.add_argument('--ppo-hidden', type=int, default=256, help='PPO hidden size')
    parser.add_argument('--ppo-lr', type=float, default=3e-4, help='PPO learning rate')
    parser.add_argument('--ppo-gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--ppo-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--ppo-clip', type=float, default=0.2, help='PPO clip epsilon')
    
    # MCTS settings
    parser.add_argument('--mcts-iterations', type=int, default=200, help='MCTS iterations')
    parser.add_argument('--mcts-depth', type=int, default=8, help='MCTS max depth')
    
    # Self-play settings
    parser.add_argument('--population-size', type=int, default=5, help='Population size')
    
    # Device
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Training device')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create config
    config = TrainingConfig(
        ppo_hidden=args.ppo_hidden,
        ppo_lr=args.ppo_lr,
        ppo_gamma=args.ppo_gamma,
        ppo_lambda=args.ppo_lambda,
        ppo_clip_epsilon=args.ppo_clip,
        mcts_iterations=args.mcts_iterations,
        mcts_depth=args.mcts_depth,
        population_size=args.population_size,
        device=args.device,
    )
    
    if args.mode == 'train':
        logger.info("[UNIFIED] Initializing trainer...")
        trainer = UnifiedTrainer(config)
        
        if args.load_path:
            logger.info("[UNIFIED] Loading checkpoint: %s", args.load_path)
            trainer.load_checkpoint(args.load_path)
        
        logger.info("[UNIFIED] Starting training...")
        metrics = trainer.train(args.episodes, args.save_path)
        
        logger.info("[UNIFIED] Training complete!")
        logger.info("[UNIFIED] Final metrics: %s", json.dumps(metrics, indent=2))
        
        # Save training history
        history_path = args.save_path.replace('.npz', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("[UNIFIED] Training history saved: %s", history_path)
    
    elif args.mode == 'eval':
        from neural_ai import NeuralAI, PPOPolicyNetwork, MCTSEngine, OpponentModel
        
        policy = PPOPolicyNetwork()
        if args.load_path:
            policy.load(args.load_path)
        
        neural_ai = NeuralAI(policy=policy)
        
        logger.info("[EVAL] Evaluating agent...")
        trainer = UnifiedTrainer(config)
        trainer.neural_ai = neural_ai
        
        results = trainer.evaluate_agent(num_episodes=50)
        logger.info("[EVAL] Results: %s", json.dumps(results, indent=2))
    
    elif args.mode == 'benchmark':
        logger.info("[BENCH] Running performance benchmark...")
        
        import time
        
        # Benchmark PPO
        from neural_ai import PPOPolicyNetwork
        
        policy = PPOPolicyNetwork(hidden=256)
        
        # Generate random data
        states = np.random.randn(64, 26).astype(np.float32)
        actions = np.random.randint(0, 12, size=64)
        rewards = np.random.randn(64)
        
        # Add to memory
        for i in range(64):
            policy.store(states[i], actions[i], rewards[i], states[i], False, 0.0, 0.0)
        
        # Benchmark update
        start = time.time()
        for _ in range(10):
            policy.compute_gae()
            policy.update(batch_size=32, epochs=2)
        elapsed = time.time() - start
        
        logger.info("[BENCH] 10 PPO updates took %.2f seconds (%.2f sec/update)",
                   elapsed, elapsed / 10)
        logger.info("[BENCH] Memory size: %d", len(policy.memory))


if __name__ == '__main__':
    sys.exit(main())