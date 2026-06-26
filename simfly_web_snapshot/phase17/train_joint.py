#!/usr/bin/env python3
"""
Phase 18: Per-Actuator RL Training with Joint-Specific Rewards
==============================================================
Trains ONE actuator at a time using biomechanically-tuned reward functions.

Usage:
  python3 train_joint.py --joint coxa_twist_T1_left
  python3 train_joint.py --joint coxa_abduct_T1_left --episodes 100 --steps 200
"""
from __future__ import annotations
import sys, os, argparse, json, time, math
import numpy as np

# Add phase16 for PerActuatorRL base class
sys.path.insert(0, '/tmp/simfly_web/phase16')
from per_actuator_rl import PerActuatorRL, DEVICE

# Add phase17 for reward function
sys.path.insert(0, '/tmp/simfly_web/phase17')
from reward import (
    compute_reward_simple, get_joint_type, get_joint_profile,
    JOINT_PROFILES
)


class JointSpecificAgent(PerActuatorRL):
    """
    Extends PerActuatorRL with joint-type-specific reward function.
    Also tracks more detailed metrics per episode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_type = get_joint_type(self.joint_name)
        self.joint_profile = get_joint_profile(self.joint_name)
        self.prev_velocity = 0.0

        print(f"[JointSpecificAgent] {self.joint_name} → type={self.joint_type}")
        print(f"  Profile: movement_w={self.joint_profile['movement_weight']:.1f}, "
              f"stability_w={self.joint_profile['stability_weight']:.1f}, "
              f"limit_penalty_scale={self.joint_profile['limit_penalty_scale']:.1f}, "
              f"range={self.joint_profile['joint_range']}")

    # Torque limits per joint type (prevents immediate limit-hitting)
    TORQUE_LIMITS = {
        'coxa_T': 1.0,
        'coxa_abduct': 0.5,
        'coxa_twist': 0.3,
        'femur_T': 1.0,
        'femur_twist': 0.4,
        'tibia_T': 0.8,
    }

    def apply_torque(self, torque: float) -> bool:
        """Override: clamp torque per joint type."""
        max_torque = self.TORQUE_LIMITS.get(self.joint_type, 1.0)
        return super().apply_torque(np.clip(torque, -max_torque, max_torque))

    def compute_reward(self, joint_angle: float, prev_joint_angle: float,
                       joint_velocity: float) -> float:
        """Override: use joint-type-specific reward."""
        reward = compute_reward_simple(
            self.joint_name,
            joint_angle,
            joint_velocity,
            prev_joint_angle
        )
        self.prev_velocity = joint_velocity
        return reward

    def train(self, n_episodes: int = 100, steps_per_episode: int = 200,
              collect_timeout: float = 0.05, verbose: bool = True):
        """
        Extended training loop with per-episode detailed metrics.
        """
        start_time = time.perf_counter()
        detailed_history = []  # per-episode detailed metrics

        for episode in range(n_episodes):
            states_list = []
            actions_list = []
            log_probs_list = []
            rewards_list = []
            values_list = []
            dones_list = []

            prev_joint_angle = 0.0
            episode_reward = 0.0
            episode_displacement = 0.0
            max_angle = -float('inf')
            min_angle = float('inf')
            torque_sum = 0.0
            osc_count = 0
            limit_violations = 0

            state = self.get_state()
            self.prev_velocity = 0.0

            for step in range(steps_per_episode):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(state_t)

                torque = action.item()
                self.apply_torque(torque)
                torque_sum += abs(torque)

                time.sleep(collect_timeout)

                next_state = self.get_state()
                joint_angle = float(next_state[4])
                joint_velocity = float(next_state[5])

                reward = self.compute_reward(joint_angle, prev_joint_angle, joint_velocity)

                # Track additional metrics
                if abs(joint_velocity) > 0.001 and abs(self.prev_velocity) > 0.001:
                    if np.sign(joint_velocity) != np.sign(self.prev_velocity):
                        osc_count += 1

                lo, hi = self.joint_profile['joint_range']
                if joint_angle > hi or joint_angle < lo:
                    limit_violations += 1

                max_angle = max(max_angle, joint_angle)
                min_angle = min(min_angle, joint_angle)

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
                self.prev_velocity = joint_velocity

            # PPO update
            rewards_arr = np.array(rewards_list, dtype=np.float32)
            values_arr = np.array(values_list, dtype=np.float32).flatten()
            dones_arr = np.array(dones_list, dtype=np.float32)
            returns, advantages = self._compute_returns_and_advantages(
                rewards_arr, values_arr, dones_arr)

            states_t = torch.FloatTensor(np.array(states_list)).to(DEVICE)
            actions_t = torch.FloatTensor(np.array(actions_list)).to(DEVICE)
            old_log_probs_t = torch.FloatTensor(np.array(log_probs_list)).to(DEVICE)
            returns_t = torch.FloatTensor(returns).unsqueeze(-1).to(DEVICE)
            advantages_t = torch.FloatTensor(advantages).unsqueeze(-1).to(DEVICE)

            update_metrics = self._update_policy(
                states_t, actions_t, old_log_probs_t, returns_t, advantages_t)

            # Logging
            self.train_metrics['episode_reward'].append(episode_reward)
            self.train_metrics['episode_displacement'].append(episode_displacement)
            self.train_metrics['mean_action'].append(float(np.mean(actions_list)))
            self.train_metrics['mean_advantage'].append(float(np.mean(advantages)))
            for k, v in update_metrics.items():
                self.train_metrics[k].append(v)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            # Detailed per-episode record
            detailed_history.append({
                'episode': episode,
                'reward': float(episode_reward),
                'displacement': float(episode_displacement),
                'max_angle': float(max_angle),
                'min_angle': float(min_angle),
                'mean_torque': float(torque_sum / steps_per_episode),
                'oscillations': osc_count,
                'limit_violations': limit_violations,
                'policy_loss': float(update_metrics.get('policy_loss', 0)),
                'value_loss': float(update_metrics.get('value_loss', 0)),
                'entropy': float(update_metrics.get('entropy', 0)),
                'mean_advantage': float(np.mean(advantages)),
            })

            if verbose and (episode % 10 == 0 or episode == n_episodes - 1):
                elapsed = time.perf_counter() - start_time
                r_mean = float(np.mean(self.train_metrics['episode_reward'][-10:]))
                print(f"  Ep {episode:4d}/{n_episodes} │ "
                      f"R={episode_reward:7.2f} (avg10: {r_mean:7.2f}) │ "
                      f"disp={episode_displacement:6.4f} │ "
                      f"θ∈[{min_angle:.3f},{max_angle:.3f}] │ "
                      f"osc={osc_count} │ "
                      f"time={elapsed:5.1f}s")

        elapsed = time.perf_counter() - start_time
        print(f"\n[Training Complete] {n_episodes} episodes in {elapsed:.1f}s")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"  Final avg10 reward: {np.mean(self.train_metrics['episode_reward'][-10:]):.2f}")

        # Store detailed history
        self.detailed_history = detailed_history
        return self.train_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 18: Per-Actuator RL with Joint-Specific Rewards")
    parser.add_argument("--joint", type=str, required=True,
                        help="Joint name (e.g., coxa_twist_T1_left)")
    parser.add_argument("--server", type=str, default="http://localhost:8080")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--save-dir", type=str,
                        default="/tmp/simfly_web/phase17/models")
    parser.add_argument("--reports-dir", type=str,
                        default="/tmp/simfly_web/phase17/reports")
    parser.add_argument("--baseline-steps", type=int, default=30,
                        help="Seconds to observe baseline torque")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline observation")
    return parser.parse_args()


def observe_baseline(joint_name: str, server_url: str, duration_s: int = 30):
    """Observe raw connectome torque for a joint without RL intervention."""
    import requests
    url = server_url.rstrip("/")
    samples = []

    print(f"  Observing baseline torque for {duration_s}s...")
    for i in range(duration_s * 2):  # sample every 0.5s
        try:
            r = requests.get(f"{url}/api/torque", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                joints = data.get("joints", {})
                if joint_name in joints:
                    samples.append(float(joints[joint_name]))
        except Exception:
            pass
        time.sleep(0.5)

    if samples:
        baseline = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'n_samples': len(samples),
            'raw_samples': [round(s, 6) for s in samples[-20:]],  # last 20
        }
    else:
        baseline = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n_samples': 0}

    print(f"  Baseline: mean={baseline['mean']:.6f}, std={baseline['std']:.6f}, "
          f"range=[{baseline['min']:.6f}, {baseline['max']:.6f}]")
    return baseline


def load_pathway_stats(joint_name: str):
    """Load pathway stats for a joint."""
    pathway_path = f"/tmp/simfly_web/phase17/pathways/{joint_name}.json"
    if not os.path.exists(pathway_path):
        return None

    with open(pathway_path) as f:
        data = json.load(f)

    # Handle both dict and list format
    if isinstance(data, dict):
        # Phase 17 format: dict with descending_neurons list
        dn_entries = data.get('descending_neurons', [])
        total_dns = len(dn_entries)
        all_mns = set(data.get('motor_neurons', []))
        total_mns = set(all_mns)
        total_interneurons = set()
        dn_types = []
        mn_counts = []
        for entry in dn_entries:
            if isinstance(entry, dict):
                dn_types.append(entry.get('dn_type', 'unknown'))
                mn_counts.append(entry.get('mn_count', 0))
                for mn in entry.get('mns', []):
                    total_mns.add(mn)
                for ineuron in entry.get('interneurons', []):
                    total_interneurons.add(ineuron)
    elif isinstance(data, list):
        # Legacy format: list of DN entries
        dn_entries = data
        total_dns = len(data)
        total_mns = set()
        total_interneurons = set()
        dn_types = []
        mn_counts = []
        for entry in data:
            dn_types.append(entry.get('dn_type', 'unknown'))
            mn_counts.append(entry.get('mn_count', 0))
            for mn in entry.get('mns', []):
                total_mns.add(mn)
            for ineuron in entry.get('interneurons', []):
                total_interneurons.add(ineuron)
    else:
        return None

    return {
        'dn_count': total_dns,
        'unique_mn_count': len(total_mns),
        'unique_interneuron_count': len(total_interneurons),
        'dn_types': list(set(dn_types)),
        'mn_counts': mn_counts,
        'mean_mn_per_dn': float(np.mean(mn_counts)) if mn_counts else 0,
        'total_pathway_neurons': total_dns + len(total_mns) + len(total_interneurons),
    }


def generate_report(joint_name, agent, baseline, pathway_stats, args):
    """Generate a comprehensive training report JSON."""
    metrics = dict(agent.train_metrics)
    detailed = getattr(agent, 'detailed_history', [])

    rewards = metrics.get('episode_reward', [])
    final_avg10 = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else 0.0

    report = {
        'joint_name': joint_name,
        'joint_type': agent.joint_type,
        'joint_profile': agent.joint_profile,
        'training_config': {
            'episodes': args.episodes,
            'steps_per_episode': args.steps,
            'hidden_dim': args.hidden,
            'lr': args.lr,
            'state_dim': agent.state_dim,
        },
        'pathway_stats': pathway_stats,
        'baseline_torque': baseline,
        'results': {
            'best_reward': float(agent.best_reward),
            'final_avg10_reward': final_avg10,
            'mean_reward_all': float(np.mean(rewards)) if rewards else 0.0,
            'total_episodes': len(rewards),
            'convergence': {
                'converged': final_avg10 > 0.0,
                'episodes_to_best': int(np.argmax(rewards)) + 1 if rewards else 0,
            },
        },
        'training_curve': {
            'episodes': list(range(len(rewards))),
            'rewards': rewards,
            'displacements': metrics.get('episode_displacement', []),
        },
        'detailed_history': detailed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    return report


def main():
    args = parse_args()
    joint_name = args.joint
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # ── Print header ──
    jtype = get_joint_type(joint_name)
    profile = get_joint_profile(joint_name)
    print(f"\n{'='*70}")
    print(f"Phase 18: Training {joint_name}")
    print(f"  Joint Type: {jtype}")
    print(f"  Profile: movement_w={profile['movement_weight']}, "
          f"stability_w={profile['stability_weight']}, "
          f"limit_scale={profile['limit_penalty_scale']}")
    print(f"  Episodes: {args.episodes} × {args.steps} steps")
    print(f"  Device: {DEVICE} | Server: {args.server}")
    print(f"{'='*70}\n")

    # ── 1. Load pathway stats ──
    pathway_stats = load_pathway_stats(joint_name)
    if pathway_stats:
        print(f"  Pathway: {pathway_stats['dn_count']} DNs → "
              f"{pathway_stats['unique_mn_count']} unique MNs → "
              f"{pathway_stats['unique_interneuron_count']} interneurons")
        print(f"  Mean MN/DN: {pathway_stats['mean_mn_per_dn']:.1f}")
        print(f"  DN types: {', '.join(pathway_stats['dn_types'][:5])}"
              + (f" +{len(pathway_stats['dn_types']) - 5} more" if len(pathway_stats['dn_types']) > 5 else ""))
    else:
        print(f"  ⚠ No pathway file for {joint_name}")

    # ── 2. Observe baseline ──
    baseline = None
    if not args.no_baseline:
        baseline = observe_baseline(joint_name, args.server, args.baseline_steps)
    else:
        baseline = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n_samples': 0}

    # ── 3. Find pathway JSON ──
    pathway_json = f"/tmp/simfly_web/phase17/pathways/{joint_name}.json"
    if not os.path.exists(pathway_json):
        pathway_json = f"/tmp/simfly_web/actuator_{joint_name.lower()}_pathway.json"
        if not os.path.exists(pathway_json):
            pathway_json = None

    # ── 4. Create agent ──
    agent = JointSpecificAgent(
        joint_name=joint_name,
        server_url=args.server,
        pathway_json=pathway_json,
        hidden_dim=args.hidden,
        lr=args.lr,
    )

    # ── 5. Train ──
    print(f"\n{'─'*70}")
    print(f"Training {joint_name}...")
    print(f"{'─'*70}")

    metrics = agent.train(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        collect_timeout=args.timeout,
        verbose=True,
    )

    # ── 6. Save model ──
    model_path = os.path.join(args.save_dir, f"{joint_name}.pt")
    agent.save(model_path)

    # ── 7. Generate report ──
    report = generate_report(joint_name, agent, baseline, pathway_stats, args)
    report_path = os.path.join(args.reports_dir, f"{joint_name}_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── 8. Print summary ──
    rewards = metrics.get('episode_reward', [])
    print(f"\n{'='*70}")
    print(f"Training Complete: {joint_name}")
    print(f"  Joint Type: {jtype}")
    print(f"  Best Reward: {agent.best_reward:.2f}")
    print(f"  Mean Reward: {np.mean(rewards):.2f}" if rewards else "  Mean Reward: N/A")
    print(f"  Final avg10: {np.mean(rewards[-10:]):.2f}" if len(rewards) >= 10 else "  Final avg10: N/A")
    print(f"  Model: {model_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*70}\n")

    return agent, report


if __name__ == "__main__":
    import torch  # needed inside PerActuatorRL
    main()
