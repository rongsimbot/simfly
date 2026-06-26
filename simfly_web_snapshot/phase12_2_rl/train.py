#!/usr/bin/env python3
"""
PPO Training Loop — Per-Joint Torque Scale Optimization.

Trains a policy to discover per-joint torque scaling multipliers
that maximize connectome-driven locomotion.

The RL does NOT control the fly — it only tunes the motor decoder interface.
The connectome provides ALL neural signals.
"""
import sys
import os
import json
import time
import math
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Add local directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_network import create_policy
from training_env import Brain2Env


# ── PPO Implementation ────────────────────────────────────────────────

class PPOTrainer:
    """Minimal but correct PPO implementation for per-joint optimization."""
    
    def __init__(self, policy, config, device='cpu'):
        self.policy = policy.to(device)
        self.device = device
        self.config = config
        
        ppo_cfg = config['ppo']
        self.clip_epsilon = ppo_cfg['clip_epsilon']
        self.gamma = ppo_cfg['gamma']
        self.gae_lambda = ppo_cfg['gae_lambda']
        self.entropy_coef = ppo_cfg['entropy_coef']
        self.value_coef = ppo_cfg['value_coef']
        self.max_grad_norm = ppo_cfg['max_grad_norm']
        self.epochs = ppo_cfg['epochs']
        self.batch_size = ppo_cfg.get('batch_size', 4)
        
        lr = ppo_cfg['learning_rate']
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['total_episodes']
        )
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # terminal state (each episode is terminal)
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            adv_std = float(advantages.std())
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """Single PPO update over collected data."""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        n = len(states)
        indices = np.arange(n)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch_i in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]
                
                _, new_log_probs, entropy, values = self.policy.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Entropy bonus
                entropy_bonus = entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                n_updates += 1
        
        self.scheduler.step()
        
        if n_updates == 0:
            n_updates = 1
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'lr': float(self.optimizer.param_groups[0]['lr']),
        }


# ── Training Utilities ────────────────────────────────────────────────

class TrainingLogger:
    """Log training progress and save results."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.episodes = []
        self.best_cv = 0.0
        self.best_scales = None
        self.best_episode = 0
    
    def log_episode(self, episode, reward, info, update_info=None):
        entry = {
            'episode': episode,
            'reward': float(reward),
            'torque_cv': info.get('torque_cv', 0),
            'food_distance': info.get('food_distance', 0),
            'food_delta': info.get('food_delta', 0),
            'cv_reward': info.get('cv_reward', 0),
            'food_reward': info.get('food_reward', 0),
            'saturation_penalty': info.get('saturation_penalty', 0),
            'sat_ratio': info.get('sat_ratio', 0),
            'multiplier_mean': info.get('multiplier_mean', 0),
            'multiplier_std': info.get('multiplier_std', 0),
            'multipliers': info.get('multipliers', []),
        }
        if update_info:
            entry.update(update_info)
        
        self.episodes.append(entry)
        
        # Track best
        cv = info.get('torque_cv', 0)
        if cv > self.best_cv:
            self.best_cv = cv
            self.best_scales = info.get('multipliers', [])
            self.best_episode = episode
    
    def save(self):
        results = {
            'episodes': self.episodes,
            'best': {
                'episode': self.best_episode,
                'torque_cv': self.best_cv,
                'scales': self.best_scales,
            },
        }
        
        with open(os.path.join(self.save_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best scales separately with joint names
        best_scales_data = {
            'episode': self.best_episode,
            'torque_cv': self.best_cv,
            'joint_scales': {},
        }
        if self.best_scales and len(self.best_scales) == len(Brain2Env.JOINT_NAMES):
            for jname, scale in zip(Brain2Env.JOINT_NAMES, self.best_scales):
                best_scales_data['joint_scales'][jname] = float(scale)
        
        with open(os.path.join(self.save_dir, 'best_scales.json'), 'w') as f:
            json.dump(best_scales_data, f, indent=2)
    
    def print_summary(self):
        if not self.episodes:
            return
        cv_values = [e['torque_cv'] for e in self.episodes]
        
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print("  Episodes: %d" % len(self.episodes))
        print("  Best CV:  %.4f (episode %d)" % (self.best_cv, self.best_episode))
        print("  Mean CV:  %.4f +/- %.4f" % (np.mean(cv_values), np.std(cv_values)))
        print("  Baseline (Phase 12.1 sweet spot): 0.4350")
        print("  Engine ceiling (uniform scaling):  0.4746")
        
        if self.best_cv > 0.435:
            improvement = (self.best_cv - 0.435) / 0.435 * 100
            print("  Improvement over sweet spot:       +%.1f%%" % improvement)
        if self.best_cv > 0.4746:
            print("  *** BEAT THE 47.46%% CEILING! ***")
        else:
            print("  Gap to ceiling: %.4f" % (0.4746 - self.best_cv))
        
        # Show best per-joint scales
        if self.best_scales and len(self.best_scales) == 36:
            print("\n  Best Per-Joint Scales (top/bottom 5):")
            joint_scales = list(zip(Brain2Env.JOINT_NAMES, self.best_scales))
            joint_scales.sort(key=lambda x: x[1], reverse=True)
            
            print("  >> Highest gain joints:")
            for name, scale in joint_scales[:5]:
                print("      %-35s = %.4f" % (name, float(scale)))
            print("  << Lowest gain joints:")
            for name, scale in joint_scales[-5:]:
                print("      %-35s = %.4f" % (name, float(scale)))
        
        print("=" * 60)


# ── Main Training Loop ────────────────────────────────────────────────

def train(config_path='rl_config.json'):
    """Main training entry point."""
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    train_cfg = config['training']
    env_cfg = config['environment']
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s" % device)
    
    # Create environment
    env = Brain2Env(
        base_url=env_cfg['base_url'],
        global_gain=env_cfg['global_gain'],
        tau_decay=env_cfg['tau_decay'],
        stabilize_seconds=env_cfg['stabilize_seconds'],
        sample_steps=env_cfg['sample_steps'],
        sample_interval=env_cfg['sample_interval'],
        cv_weight=env_cfg['cv_weight'],
        food_weight=env_cfg['food_weight'],
        sat_penalty=env_cfg['sat_penalty'],
    )
    
    print("Environment: %s" % env_cfg['base_url'])
    print("  Sweet spot: gain=%.3f, tau=%.0fms" % (env_cfg['global_gain'], env_cfg['tau_decay']))
    print("  State dim: %d, Action dim: %d" % (env.STATE_DIM, env.N_JOINTS))
    
    # Create policy
    policy_cfg = config['policy']
    policy = create_policy(
        network_type=policy_cfg['network_type'],
        state_dim=env.STATE_DIM,
        action_dim=env.N_JOINTS,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print("Policy: %s, params: %d" % (type(policy).__name__, n_params))
    
    # Create trainer
    trainer = PPOTrainer(policy, config, device=device)
    
    # Logger
    save_dir = train_cfg.get('save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger(save_dir)
    
    # Training
    total_episodes = train_cfg['total_episodes']
    buffer_size = train_cfg.get('buffer_size', 8)
    print_freq = train_cfg.get('print_freq', 5)
    
    print("\n" + "=" * 60)
    print("Starting Training: %d episodes" % total_episodes)
    print("  Buffer size: %d, Print freq: %d" % (buffer_size, print_freq))
    est_time_per_ep = env_cfg['stabilize_seconds'] + env_cfg['sample_steps'] * env_cfg['sample_interval']
    print("  Each episode: %.0fs stabilize + %.0fs sampling" % (
        env_cfg['stabilize_seconds'], env_cfg['sample_steps'] * env_cfg['sample_interval']))
    print("=" * 60 + "\n")
    
    # Initialize at sweet spot
    obs, _ = env.reset()
    
    # Rolling stats
    cv_window = deque(maxlen=10)
    reward_window = deque(maxlen=10)
    
    start_time = time.time()
    
    # Storage for PPO buffer
    buffer_states = []
    buffer_actions = []
    buffer_log_probs = []
    buffer_rewards = []
    buffer_values = []
    buffer_dones = []
    
    for episode in range(total_episodes):
        # Get action from policy
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_raw, log_prob, _, value = policy.get_action_and_value(state_tensor)
        
        action_np = action_raw.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.item()
        value_np = value.item()
        
        # Execute action in environment
        next_obs, reward, done, trunc, info = env.step(action_np)
        
        # Store in buffer
        buffer_states.append(obs)
        buffer_actions.append(action_np)
        buffer_log_probs.append(log_prob_np)
        buffer_rewards.append(reward)
        buffer_values.append(value_np)
        buffer_dones.append(float(done))
        
        # Update stats
        cv_window.append(info.get('torque_cv', 0))
        reward_window.append(reward)
        logger.log_episode(episode + 1, reward, info)
        
        # PPO update when buffer is full
        update_info = None
        if len(buffer_states) >= buffer_size:
            # Compute GAE
            advantages, returns = trainer.compute_gae(
                buffer_rewards, buffer_values, buffer_dones
            )
            
            update_info = trainer.update(
                buffer_states, buffer_actions, buffer_log_probs,
                advantages, returns
            )
            
            # Clear buffer
            buffer_states.clear()
            buffer_actions.clear()
            buffer_log_probs.clear()
            buffer_rewards.clear()
            buffer_values.clear()
            buffer_dones.clear()
        
        obs = next_obs
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (episode + 1)) * (total_episodes - episode - 1)
            
            cv_val = info.get('torque_cv', cv_window[-1] if cv_window else 0)
            fd_val = info.get('food_distance', 0)
            mu_val = info.get('multiplier_mean', 0)
            line = "Ep %3d/%d | R=%+.3f | CV=%.4f (avg10=%.4f) | FD=%.3f | mu_scale=%.3f" % (
                episode + 1, total_episodes, reward, cv_val,
                np.mean(cv_window), fd_val, mu_val)
            if update_info:
                line += " | loss=%.4f" % update_info['loss']
            line += " | %ds elapsed, ETA %ds" % (int(elapsed), int(eta))
            print(line)
            sys.stdout.flush()
        
        # Periodic save (every 10 episodes to be safe)
        if (episode + 1) % 10 == 0:
            logger.save()
            torch.save(policy.state_dict(), 
                      os.path.join(save_dir, 'policy_ep%d.pt' % (episode + 1)))
    
    # Final save
    elapsed = time.time() - start_time
    print("\nTraining complete in %ds (%.1f min)" % (int(elapsed), elapsed / 60))
    logger.save()
    torch.save(policy.state_dict(), os.path.join(save_dir, 'policy_final.pt'))
    logger.print_summary()
    
    return logger


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PPO Per-Joint Torque Optimization')
    parser.add_argument('--config', default='rl_config.json', help='Config file path')
    parser.add_argument('--episodes', type=int, help='Override total episodes')
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    
    if args.episodes:
        with open(config_path) as f:
            config = json.load(f)
        config['training']['total_episodes'] = args.episodes
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_runtime_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    try:
        logger = train(config_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print("\nTraining error: %s" % e)
        import traceback
        traceback.print_exc()
        print("\nPartial results saved")
