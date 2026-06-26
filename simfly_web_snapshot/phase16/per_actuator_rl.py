#!/usr/bin/env python3
"""
Phase 16: Per-Actuator RL Framework
====================================
Each actuator gets its OWN tiny RL controller.
All controllers read the SAME connectome state.
Each produces DIFFERENT torque for its specific joint.

Architecture:
- PPO with tiny network (64 hidden, 2 layers)
- State: DN firing vector + MN activations + joint angle + joint velocity
- Action: 1 scalar torque for this actuator
- Reward: angular_displacement * sign_toward_center (reward movement, penalize limits)
- Training: ISOLATION mode (only target joint gets torque)
"""
from __future__ import annotations
import json, time, os, sys, argparse, traceback

import sys
sys.stdout.reconfigure(line_buffering=True)
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import requests

# ── Config ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PerActuatorRL] Device: {DEVICE}")
torch.set_default_dtype(torch.float32)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PPO Policy Network (TINY — 64 hidden, 2 layers)                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
class TinyActorCritic(nn.Module):
    """Tiny PPO network: shared backbone → actor head (μ,σ) + critic head (V)."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, 1)   # scalar torque mean ∈ [-1, 1]
        self.actor_logstd = nn.Parameter(torch.zeros(1))  # learnable std
        self.critic = nn.Linear(hidden, 1)       # state value

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(state)
        mean = torch.tanh(self.actor_mean(shared))  # torque ∈ [-1, 1]
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(shared)
        return mean, std, value

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, value)."""
        mean, std, value = self.forward(state)
        dist = Normal(mean, std + 1e-6)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob, entropy, value for stored actions."""
        mean, std, value = self.forward(state)
        dist = Normal(mean, std + 1e-6)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Per-Actuator RL Agent                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
class PerActuatorRL:
    """
    Tiny RL agent controlling ONE actuator via HTTP from a Brain2 server.

    Reads: DN firing state, MN activation, joint angle, joint velocity
    Outputs: scalar torque for its actuator
    """

    def __init__(
        self,
        joint_name: str,
        server_url: str = "http://localhost:8080",
        pathway_json: Optional[str] = None,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 5,
        batch_size: int = 64,
        joint_angle_limit: float = 0.8,  # radians — penalize exceeding this
        angular_displacement_weight: float = 10.0,  # reward scale
    ):
        self.joint_name = joint_name
        self.server_url = server_url.rstrip("/")

        # ── Load pathway to determine state dimensions ──
        if pathway_json and os.path.exists(pathway_json):
            with open(pathway_json) as f:
                self.pathway = json.load(f)
            self.dn_count = len(self.pathway.get("descending_neurons", []))
            self.mn_count = len(self.pathway.get("motor_neurons", []))
        else:
            print(f"  ⚠ No pathway JSON for {joint_name}, using defaults (839 DN, 61 MN)")
            self.pathway = None
            self.dn_count = 839
            self.mn_count = 61

        # state = [DN_firing..., MN_activation..., joint_angle, joint_velocity]
        # Note: DN firing is a binary vector; MN activations are float counts
        # For simplicity we use a fixed-size encoding:
        # - DN firing: summary stats (count, fraction) + top-k
        # - MN activation: summary stats + decayed accumulation
        # - joint_angle, joint_velocity: 2 scalars

        # Compact state encoding (not full sparse vector):
        # [dn_fired_count, dn_fraction, mn_active_count, mn_mean_activation,
        #  joint_angle, joint_velocity] = 6 dims
        # This keeps the network tiny while capturing essential state
        self.state_dim = 6
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.joint_angle_limit = joint_angle_limit
        self.angular_displacement_weight = angular_displacement_weight

        # ── Build network ──
        self.policy = TinyActorCritic(self.state_dim, hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # ── Tracking ──
        self.train_metrics: Dict[str, List[float]] = defaultdict(list)
        self.best_reward = -float("inf")

        print(f"[PerActuatorRL] {joint_name}: "
              f"state_dim={self.state_dim}, hidden={hidden_dim}, "
              f"DN={self.dn_count}, MN={self.mn_count}, device={DEVICE}")

    def get_state(self) -> np.ndarray:
        """
        Read actuator state from Brain2 API.
        Returns compact state vector [6,].
        """
        try:
            r = requests.get(
                f"{self.server_url}/api/actuator/state/{self.joint_name}",
                timeout=2.0
            )
            if r.status_code == 200:
                data = r.json()
                dn_fired = data.get("dn_fired", 0)
                dn_total = data.get("dn_total", max(self.dn_count, 1))
                mn_active = data.get("mn_active", 0)
                mn_mean = data.get("mn_mean_activation", 0.0)
                joint_angle = data.get("joint_angle", 0.0)
                joint_velocity = data.get("joint_velocity", 0.0)
            else:
                # Fallback: read /api/status
                r2 = requests.get(f"{self.server_url}/api/status", timeout=2.0)
                d = r2.json()
                metrics = d.get("metrics", {})
                dn_fired = metrics.get("dn_matches", 0)
                dn_total = max(self.dn_count, 1)
                mn_active = metrics.get("mns_activated", 0)
                mn_mean = mn_active / max(self.mn_count, 1)
                joint_angle = 0.0
                joint_velocity = 0.0
        except Exception:
            return np.zeros(self.state_dim, dtype=np.float32)

        state = np.array([
            float(dn_fired) / max(dn_total, 1),         # DN fraction
            float(dn_fired),                             # DN count (clipped)
            float(mn_active) / max(self.mn_count, 1),    # MN fraction
            min(float(mn_mean), 10.0),                   # MN mean activation (clipped)
            np.clip(float(joint_angle), -np.pi, np.pi),  # joint angle
            np.clip(float(joint_velocity), -20.0, 20.0), # joint velocity
        ], dtype=np.float32)
        return state

    def apply_torque(self, torque: float) -> bool:
        """Send torque command to Brain2 server for this actuator."""
        try:
            r = requests.post(
                f"{self.server_url}/api/actuator/torque",
                json={"joint": self.joint_name, "torque": float(np.clip(torque, -1.0, 1.0))},
                timeout=2.0
            )
            return r.status_code == 200
        except Exception:
            return False

    def compute_reward(self, joint_angle: float, prev_joint_angle: float,
                       joint_velocity: float) -> float:
        """
        Reward: angular displacement * sign toward center.
        Penalize hitting joint limits.
        """
        displacement = abs(joint_angle - prev_joint_angle)

        # Sign toward center: positive if moving toward 0
        toward_center = -np.sign(joint_angle * joint_velocity) if abs(joint_velocity) > 0.001 else 0.0

        # Movement reward
        move_reward = displacement * self.angular_displacement_weight

        # Penalty for exceeding angle limits
        limit_penalty = 0.0
        if abs(joint_angle) > self.joint_angle_limit:
            limit_penalty = -1.0 * (abs(joint_angle) - self.joint_angle_limit) * 5.0

        # Bonus for moving toward center
        center_bonus = displacement * max(0.0, toward_center) * 2.0

        # Small penalty for zero movement (encourage exploration)
        zero_penalty = -0.01 if displacement < 0.0001 else 0.0

        reward = move_reward + center_bonus + limit_penalty + zero_penalty
        return float(reward)

    # ── PPO Training ────────────────────────────────────────────────
    def _compute_returns_and_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t]
                next_nonterminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0

            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _update_policy(self, states: torch.Tensor, actions: torch.Tensor,
                       old_log_probs: torch.Tensor, returns: torch.Tensor,
                       advantages: torch.Tensor) -> Dict[str, float]:
        """PPO update step on collected trajectory."""
        metrics = defaultdict(list)

        for _ in range(self.ppo_epochs):
            # Shuffle
            indices = torch.randperm(len(states), device=DEVICE)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                s = states[idx]
                a = actions[idx]
                old_lp = old_log_probs[idx]
                ret = returns[idx]
                adv = advantages[idx]

                new_lp, entropy, values = self.policy.evaluate(s, a)

                # PPO clipped objective
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values - ret).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ── Training Loop ───────────────────────────────────────────────
    def train(self, n_episodes: int = 100, steps_per_episode: int = 200,
              collect_timeout: float = 0.05, verbose: bool = True):
        """
        Train the RL agent in ISOLATION mode.
        - Brain2 must be running with --target-joint {joint_name}
        - Only this actuator receives torque
        """
        start_time = time.perf_counter()

        for episode in range(n_episodes):
            # ── Collect trajectory ──
            states_list = []
            actions_list = []
            log_probs_list = []
            rewards_list = []
            values_list = []
            dones_list = []

            prev_joint_angle = 0.0
            episode_reward = 0.0
            episode_displacement = 0.0

            state = self.get_state()

            for step in range(steps_per_episode):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(state_t)

                torque = action.item()
                self.apply_torque(torque)

                # Wait for physics step
                time.sleep(collect_timeout)

                next_state = self.get_state()
                joint_angle = float(next_state[4])
                joint_velocity = float(next_state[5])

                reward = self.compute_reward(joint_angle, prev_joint_angle, joint_velocity)

                states_list.append(state)
                actions_list.append(action.cpu().numpy().flatten())
                log_probs_list.append(log_prob.cpu().numpy().flatten())
                rewards_list.append(reward)
                values_list.append(value.cpu().numpy().flatten())
                dones_list.append(step == steps_per_episode - 1)

                episode_reward += reward
                episode_displacement += abs(joint_angle - prev_joint_angle)
                prev_joint_angle = joint_angle
                state = next_state

            # ── Compute returns & advantages ──
            rewards_arr = np.array(rewards_list, dtype=np.float32)
            values_arr = np.array(values_list, dtype=np.float32).flatten()
            dones_arr = np.array(dones_list, dtype=np.float32)
            returns, advantages = self._compute_returns_and_advantages(
                rewards_arr, values_arr, dones_arr)

            # ── PPO update ──
            states_t = torch.FloatTensor(np.array(states_list)).to(DEVICE)
            actions_t = torch.FloatTensor(np.array(actions_list)).to(DEVICE)
            old_log_probs_t = torch.FloatTensor(np.array(log_probs_list)).to(DEVICE)
            returns_t = torch.FloatTensor(returns).unsqueeze(-1).to(DEVICE)
            advantages_t = torch.FloatTensor(advantages).unsqueeze(-1).to(DEVICE)

            update_metrics = self._update_policy(
                states_t, actions_t, old_log_probs_t, returns_t, advantages_t)

            # ── Logging ──
            self.train_metrics['episode_reward'].append(episode_reward)
            self.train_metrics['episode_displacement'].append(episode_displacement)
            self.train_metrics['mean_action'].append(float(np.mean(actions_list)))
            self.train_metrics['mean_advantage'].append(float(np.mean(advantages)))
            for k, v in update_metrics.items():
                self.train_metrics[k].append(v)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if verbose and (episode % 10 == 0 or episode == n_episodes - 1):
                elapsed = time.perf_counter() - start_time
                r_mean = float(np.mean(self.train_metrics['episode_reward'][-10:]))
                print(f"  Ep {episode:4d}/{n_episodes} │ "
                      f"R={episode_reward:7.2f} (avg10: {r_mean:7.2f}) │ "
                      f"disp={episode_displacement:6.4f} │ "
                      f"time={elapsed:5.1f}s")

        elapsed = time.perf_counter() - start_time
        print(f"\n[Training Complete] {n_episodes} episodes in {elapsed:.1f}s")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"  Final avg10 reward: {np.mean(self.train_metrics['episode_reward'][-10:]):.2f}")
        return self.train_metrics

    def save(self, path: str):
        """Save model, config, and training metrics."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'joint_name': self.joint_name,
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim,
                'dn_count': self.dn_count,
                'mn_count': self.mn_count,
                'joint_angle_limit': self.joint_angle_limit,
                'angular_displacement_weight': self.angular_displacement_weight,
                'lr': self.lr, 'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'ppo_epochs': self.ppo_epochs,
            },
            'train_metrics': dict(self.train_metrics),
            'best_reward': self.best_reward,
        }
        torch.save(save_data, path)
        print(f"[Saved] {path}")

        # Also save training metrics as JSON
        metrics_path = path.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.train_metrics), f, indent=2)
        print(f"[Saved Metrics] {metrics_path}")

    @classmethod
    def load(cls, path: str, server_url: str = "http://localhost:8080",
             pathway_json: Optional[str] = None) -> "PerActuatorRL":
        """Load a saved model."""
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = data['config']

        agent = cls(
            joint_name=cfg['joint_name'],
            server_url=server_url,
            pathway_json=pathway_json,
            hidden_dim=cfg.get('hidden_dim', 64),
            lr=cfg.get('lr', 3e-4),
            gamma=cfg.get('gamma', 0.99),
            gae_lambda=cfg.get('gae_lambda', 0.95),
            clip_epsilon=cfg.get('clip_epsilon', 0.2),
            ppo_epochs=cfg.get('ppo_epochs', 5),
            joint_angle_limit=cfg.get('joint_angle_limit', 0.8),
            angular_displacement_weight=cfg.get('angular_displacement_weight', 10.0),
        )
        agent.policy.load_state_dict(data['policy_state_dict'])
        agent.optimizer.load_state_dict(data['optimizer_state_dict'])
        agent.best_reward = data.get('best_reward', -float('inf'))
        if 'train_metrics' in data:
            for k, v in data['train_metrics'].items():
                agent.train_metrics[k] = v
        print(f"[Loaded] {path} (best_reward={agent.best_reward:.2f})")
        return agent

    def get_torque(self, state: Optional[np.ndarray] = None) -> float:
        """Inference: get deterministic torque for current state."""
        if state is None:
            state = self.get_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, _, _ = self.policy(state_t)
        return float(mean.item())


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Parallel Multi-Actuator Controller                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MultiActuatorController:
    """
    Runs multiple PerActuatorRL agents in parallel.
    All agents read the SAME connectome state.
    Each produces DIFFERENT torque for its own actuator.
    """

    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        self.agents: Dict[str, PerActuatorRL] = {}
        self.joint_names: List[str] = []

    def add_agent(self, agent: PerActuatorRL):
        self.agents[agent.joint_name] = agent
        self.joint_names.append(agent.joint_name)

    def step(self) -> Dict[str, float]:
        """
        One parallel control step:
        1. Read connectome state ONCE
        2. Each agent computes its torque
        3. Post all torques to server
        """
        # Read global connectome state
        try:
            r = requests.get(f"{self.server_url}/api/status", timeout=2.0)
            global_state = r.json()
        except Exception:
            global_state = {}

        torques = {}
        for jname, agent in self.agents.items():
            # Each agent reads its OWN actuator state
            state = agent.get_state()
            torque = agent.get_torque(state)
            torques[jname] = torque

        # Post all torques
        for jname, torque in torques.items():
            try:
                requests.post(
                    f"{self.server_url}/api/actuator/torque",
                    json={"joint": jname, "torque": float(torque)},
                    timeout=1.0
                )
            except Exception:
                pass

        return torques


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CLI Interface                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Actuator RL Training")
    parser.add_argument("--joint", type=str, required=True,
                        help="Joint name (e.g. coxa_T1_left)")
    parser.add_argument("--server", type=str, default="http://localhost:8080",
                        help="Brain2 server URL")
    parser.add_argument("--pathway", type=str, default=None,
                        help="Path to pathway JSON")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Training episodes")
    parser.add_argument("--steps", type=int, default=200,
                        help="Steps per episode")
    parser.add_argument("--save", type=str, default=None,
                        help="Model save path (.pt)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--collect-timeout", type=float, default=0.05,
                        help="Seconds to wait between steps for physics update")
    args = parser.parse_args()

    agent = PerActuatorRL(
        joint_name=args.joint,
        server_url=args.server,
        pathway_json=args.pathway,
        hidden_dim=args.hidden,
        lr=args.lr,
    )

    print(f"\n{'='*60}")
    print(f"Training {args.joint} — {args.episodes} episodes × {args.steps} steps")
    print(f"State dim: {agent.state_dim} | Hidden: {args.hidden} | Device: {DEVICE}")
    print(f"Server: {args.server}")
    print(f"{'='*60}\n")

    metrics = agent.train(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        collect_timeout=args.collect_timeout,
        verbose=True,
    )

    save_path = args.save or f"/tmp/simfly_web/phase16/models/{args.joint}.pt"
    agent.save(save_path)
    print(f"\nDone. Model: {save_path}")
