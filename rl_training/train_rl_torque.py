#!/usr/bin/env python3
"""
train_rl_torque.py — RL-Enhanced Torque Decoder Training Script.

Trains a PPO policy to calibrate connectome-driven torque outputs for
food-seeking chemotaxis behavior. The connectome drives ALL movement;
RL only learns per-joint gain/bias calibration.

USAGE (on GB10):
    DISPLAY=:10 MUJOCO_GL=egl \
    /home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 \
    train_rl_torque.py --iterations 200 --neurons 2000 --joints 36

Architecture:
  SimFlyRLPipeline (real MuJoCo + connectome) → SimFlyRLEnv (rl_bridge.py)
  → PPO trainer → learned gain/bias → torque calibration

Outputs:
  rl_training_output/
    ├── train_log.jsonl       — training metrics per iteration
    ├── snapshots/            — policy weights per iteration
    ├── best_policy.npz       — best-performing policy
    ├── comparison_report.json — fixed-gain vs RL performance
    └── training_metrics.png  — learning curve plot
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from typing import Dict, List, Any
import numpy as np

# Force unbuffered output for real-time monitoring
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(line_buffering=True)


# Force unbuffered output for real-time progress monitoring
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)

# ── Path Setup ──────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
RL_TRAINING_DIR = os.path.join(CODE_ROOT, "rl_training")

for d in [CODE_ROOT, RL_TRAINING_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# ── Imports ─────────────────────────────────────────────────────────────
from rl_bridge import (
    RLConfig, ConnectomeModulationPolicy, SimFlyRLEnv,
    train_ppo, apply_modulation,
)
from rl_simfly_pipeline import SimFlyRLPipeline


def run_fixed_gain_baseline(pipeline: SimFlyRLPipeline, n_eval_steps: int = 400):
    """Run fixed-gain baseline (gains=1.0, biases=0.0 — connectome passthrough).

    Returns metrics dict.
    """
    print(f"\n[BASELINE] Running fixed-gain baseline ({n_eval_steps} steps)...", flush=True)
    pipeline.reset()
    ones = np.ones(pipeline.cfg.n_joints)
    zeros = np.zeros(pipeline.cfg.n_joints)

    food_distances = []
    x_positions = []
    velocities = []
    active_joints_list = []
    upright_list = []
    reached_food = False
    total_reward = 0.0

    for step in range(n_eval_steps):
        # Get connectome torques (passthrough: gain=1, bias=0)
        raw_torques = pipeline.get_connectome_torques()
        modulated = apply_modulation(raw_torques, pipeline.joint_names, ones, zeros)
        pipeline.apply_torques(modulated)
        pipeline.step_physics()

        state = pipeline.get_state()
        velocities.append(state['x_velocity'])
        food_distances.append(state['food_distance'])
        x_positions.append(float(pipeline.data.qpos[0]) if pipeline.data is not None else 0)
        active_joints_list.append(
            sum(1 for v in modulated.values() if abs(float(v)) > 0.001))
        upright_list.append(state['upright'])

        reward = (float(np.clip(state.get('x_velocity', 0), -1, 2))
                  + 0.5 * state.get('upright', 0)
                  + 0.1)  # alive bonus
        total_reward += reward

        if state.get('food_distance', 999) < 0.5:
            reached_food = True
            break
        if state.get('fell', False):
            break

    mean_vel = float(np.mean(velocities)) if velocities else 0.0
    return {
        'method': 'fixed_gain',
        'steps_run': step + 1,
        'total_reward': float(total_reward),
        'mean_x_velocity': mean_vel,
        'max_x_velocity': float(np.max(np.abs(velocities))) if velocities else 0.0,
        'final_food_distance': float(food_distances[-1]) if food_distances else 0.0,
        'min_food_distance': float(min(food_distances)) if food_distances else 0.0,
        'initial_food_distance': float(food_distances[0]) if food_distances else 0.0,
        'food_distance_reduction': (float(food_distances[0]) - float(food_distances[-1]))
            if food_distances else 0.0,
        'reached_food': reached_food,
        'final_x': float(x_positions[-1]) if x_positions else 0.0,
        'mean_active_joints': float(np.mean(active_joints_list)) if active_joints_list else 0.0,
        'mean_upright': float(np.mean(upright_list)) if upright_list else 0.0,
        'torque_smoothness': _compute_torque_smoothness(modulated),
    }


def run_rl_policy_baseline(pipeline: SimFlyRLPipeline, policy: ConnectomeModulationPolicy,
                           n_eval_steps: int = 400):
    """Run RL-trained policy evaluation.

    Returns metrics dict.
    """
    print(f"\n[RL-EVAL] Running RL policy evaluation ({n_eval_steps} steps)...", flush=True)
    pipeline.reset()

    food_distances = []
    x_positions = []
    velocities = []
    active_joints_list = []
    upright_list = []
    gains_history = []
    biases_history = []
    reached_food = False
    total_reward = 0.0
    all_modulated = []

    for step in range(n_eval_steps):
        obs = pipeline.get_observation()
        squashed, _, _, _ = policy.act(obs)
        n = pipeline.cfg.n_joints
        gains, biases = squashed[:n], squashed[n:]
        gains_history.append(gains.copy())
        biases_history.append(biases.copy())

        raw_torques = pipeline.get_connectome_torques()
        modulated = apply_modulation(raw_torques, pipeline.joint_names, gains, biases)
        all_modulated.append(dict(modulated))
        pipeline.apply_torques(modulated)
        pipeline.step_physics()

        state = pipeline.get_state()
        velocities.append(state['x_velocity'])
        food_distances.append(state['food_distance'])
        x_positions.append(float(pipeline.data.qpos[0]) if pipeline.data is not None else 0)
        active_joints_list.append(
            sum(1 for v in modulated.values() if abs(float(v)) > 0.001))
        upright_list.append(state['upright'])

        reward = (float(np.clip(state.get('x_velocity', 0), -1, 2))
                  + 0.5 * state.get('upright', 0)
                  + 0.1
                  - (state['food_distance'] / 10.0) * 0.2)  # chemotaxis bonus
        total_reward += reward

        if state.get('food_distance', 999) < 0.5:
            reached_food = True
            break
        if state.get('fell', False):
            break

    mean_vel = float(np.mean(velocities)) if velocities else 0.0
    mean_gain = float(np.mean([np.mean(g) for g in gains_history])) if gains_history else 1.0
    mean_bias = float(np.mean([np.mean(b) for b in biases_history])) if biases_history else 0.0

    return {
        'method': 'rl_policy',
        'steps_run': step + 1,
        'total_reward': float(total_reward),
        'mean_x_velocity': mean_vel,
        'max_x_velocity': float(np.max(np.abs(velocities))) if velocities else 0.0,
        'final_food_distance': float(food_distances[-1]) if food_distances else 0.0,
        'min_food_distance': float(min(food_distances)) if food_distances else 0.0,
        'initial_food_distance': float(food_distances[0]) if food_distances else 0.0,
        'food_distance_reduction': (float(food_distances[0]) - float(food_distances[-1]))
            if food_distances else 0.0,
        'reached_food': reached_food,
        'final_x': float(x_positions[-1]) if x_positions else 0.0,
        'mean_active_joints': float(np.mean(active_joints_list)) if active_joints_list else 0.0,
        'mean_upright': float(np.mean(upright_list)) if upright_list else 0.0,
        'mean_gain': float(mean_gain),
        'mean_bias': float(mean_bias),
        'gain_std': float(np.std([np.mean(g) for g in gains_history])) if gains_history else 0.0,
        'bias_std': float(np.std([np.mean(b) for b in biases_history])) if biases_history else 0.0,
        'torque_smoothness': (_compute_torque_smoothness(all_modulated[-1])
                              if all_modulated else 0.0),
    }


def _compute_torque_smoothness(torques: Dict[str, float]) -> float:
    """Compute torque smoothness: lower variance = smoother."""
    vals = [abs(float(v)) for v in torques.values()]
    if not vals:
        return 1.0
    return 1.0 / (1.0 + float(np.std(vals)))


def plot_learning_curve(history: List[Dict], output_path: str):
    """Plot training learning curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        iters = [h['iteration'] for h in history]
        rewards = [h['mean_episode_reward'] for h in history]
        gains = [h['mean_gain'] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(iters, rewards, 'b-', linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('PPO Training: SimFly Torque Calibration')
        ax1.grid(True, alpha=0.3)

        ax2.plot(iters, gains, 'g-', linewidth=1.5, label='Mean Gain')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Fixed-Gain (1.0)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Gain')
        ax2.set_title('Policy Gain Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  [PLOT] Learning curve saved: {output_path}", flush=True)
    except Exception as e:
        print(f"  [PLOT] Could not generate plot: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="RL-Enhanced Torque Decoder Training")
    parser.add_argument('--iterations', type=int, default=100, help='PPO iterations')
    parser.add_argument('--neurons', type=int, default=0, help='Connectome neurons (0=all)')
    parser.add_argument('--joints', type=int, default=36, help='Active leg joints')
    parser.add_argument('--rollout', type=int, default=1024, help='Rollout steps per iteration')
    parser.add_argument('--max-ep-steps', type=int, default=200, help='Max episode steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--skip-init', action='store_true', help='Skip pipeline init (load from cache)')
    parser.add_argument('--eval-only', type=str, help='Evaluate saved policy (path to .npz)')
    parser.add_argument('--compare-only', action='store_true', help='Only run comparison, no training')
    parser.add_argument('--food-x', type=float, default=8.0, help='Food source x position')
    parser.add_argument('--food-y', type=float, default=3.0, help='Food source y position')
    args = parser.parse_args()

    # ── Output Setup ──────────────────────────────────────────────────
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(RL_TRAINING_DIR, "rl_training_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────
    config = RLConfig(
        n_joints=args.joints,
        hidden=args.hidden,
        lr=args.lr,
        rollout_steps=args.rollout,
        max_ep_steps=args.max_ep_steps,
        seed=args.seed,
    )

    food_pos = (args.food_x, args.food_y, 0.0)

    print(f"\n{'='*60}", flush=True)
    print(f"RL-ENHANCED TORQUE DECODER — TRAINING", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Config: {args.joints} joints, {args.neurons} neurons", flush=True)
    print(f"  Network: MLP {config.obs_dim}→{args.hidden}→{args.hidden}→{2*args.joints}", flush=True)
    print(f"  Algorithm: PPO (numpy, analytic gradients)", flush=True)
    print(f"  Iterations: {args.iterations} × {args.rollout} rollout steps", flush=True)
    print(f"  Food at: ({args.food_x}, {args.food_y})", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Initialize Pipeline ───────────────────────────────────────────
    if not args.skip_init:
        print("[1/4] Initializing connectome pipeline...", flush=True)
        t0 = time.perf_counter()
        pipeline = SimFlyRLPipeline(config, max_neurons=args.neurons, food_pos=food_pos)
        pipeline.initialize(verbose=True)
        print(f"  Pipeline init: {time.perf_counter() - t0:.1f}s\n", flush=True)
    else:
        raise NotImplementedError("--skip-init not yet supported")

    # ── Eval-Only Mode ────────────────────────────────────────────────
    if args.eval_only:
        print(f"[EVAL] Loading policy from: {args.eval_only}", flush=True)
        policy = ConnectomeModulationPolicy(config)
        data = np.load(args.eval_only, allow_pickle=True)
        policy.set_params({k: data[k] for k in data.files})
        metrics = run_rl_policy_baseline(pipeline, policy, n_eval_steps=400)
        print(f"\n[EVAL] Results: reward={metrics['total_reward']:.1f}, "
              f"food_dist={metrics['final_food_distance']:.1f}, "
              f"vel={metrics['mean_x_velocity']:.3f}, "
              f"reached_food={metrics['reached_food']}", flush=True)
        return 0

    # ── Run Fixed-Gain Baseline ──────────────────────────────────────
    print("[2/4] Running fixed-gain baseline...", flush=True)
    baseline_metrics = run_fixed_gain_baseline(pipeline, n_eval_steps=400)
    print(f"  Fixed-gain: reward={baseline_metrics['total_reward']:.1f}, "
      f"food_dist={baseline_metrics['final_food_distance']:.1f}m, "
      f"vel={baseline_metrics['mean_x_velocity']:.3f}m/s, "
      f"reached_food={baseline_metrics['reached_food']}", flush=True)

    if args.compare_only:
        # Just save baseline and exit
        report_path = os.path.join(output_dir, "comparison_report.json")
        comparison = {
            'config': {
                'n_joints': args.joints, 'n_neurons': args.neurons,
                'algorithm': 'PPO', 'hidden': args.hidden,
                'lr': args.lr, 'iterations': 0,
            },
            'baseline_fixed_gain': baseline_metrics,
            'rl_policy': None,
            'improvement': {},
            'timestamp': __import__('datetime').datetime.now().isoformat(),
        }
        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n  Report saved: {report_path}", flush=True)
        return 0

    # ── Train RL Policy ──────────────────────────────────────────────
    print("\n[3/4] Training RL torque calibration policy...", flush=True)
    env = SimFlyRLEnv(pipeline)
    policy = ConnectomeModulationPolicy(config)

    log_path = os.path.join(output_dir, "train_log.jsonl")
    t_start = time.perf_counter()

    # Training with progress output
    print("  Starting PPO iterations...", flush=True)
    history = train_ppo(env, policy, config, n_iterations=args.iterations, log_path=log_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Training complete: {elapsed:.1f}s ({elapsed/args.iterations:.1f}s/iter)", flush=True)

    # Save final policy
    best_path = os.path.join(output_dir, "best_policy.npz")
    policy.save(best_path)
    print(f"  Best policy saved: {best_path}", flush=True)

    # ── Evaluate RL Policy ───────────────────────────────────────────
    print("\n[4/4] Running RL policy evaluation...", flush=True)
    rl_metrics = run_rl_policy_baseline(pipeline, policy, n_eval_steps=400)

    # ── Comparison Report ────────────────────────────────────────────
    improvement = {}
    for key in ['total_reward', 'mean_x_velocity', 'food_distance_reduction',
                'final_food_distance', 'torque_smoothness']:
        if key in baseline_metrics and key in rl_metrics:
            bl = baseline_metrics[key]
            rl = rl_metrics[key]
            if key == 'final_food_distance':
                # Lower is better
                improvement[key] = {
                    'fixed_gain': bl, 'rl_policy': rl,
                    'delta': rl - bl,
                    'pct_change': ((rl - bl) / (abs(bl) + 1e-8)) * 100,
                }
            else:
                # Higher is better
                improvement[key] = {
                    'fixed_gain': bl, 'rl_policy': rl,
                    'delta': rl - bl,
                    'pct_change': ((rl - bl) / (abs(bl) + 1e-8)) * 100,
                }

    comparison = {
        'config': {
            'n_joints': args.joints,
            'n_neurons': args.neurons,
            'algorithm': 'PPO (numpy, analytic gradients)',
            'hidden': args.hidden,
            'lr': args.lr,
            'iterations': args.iterations,
            'rollout_steps': args.rollout,
            'max_ep_steps': args.max_ep_steps,
        },
        'baseline_fixed_gain': baseline_metrics,
        'rl_policy': rl_metrics,
        'improvement': improvement,
        'training_duration_s': elapsed,
        'training_iterations': args.iterations,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }

    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    # ── Print Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING COMPLETE — COMPARISON REPORT", flush=True)
    print(f"{'='*60}", flush=True)
    for metric, vals in improvement.items():
        arrow = "↑" if vals['delta'] > 0 else "↓"
        print(f"  {metric:30s}: {vals['fixed_gain']:8.3f} → {vals['rl_policy']:8.3f} "
              f"({vals['pct_change']:+6.1f}% {arrow})", flush=True)

    print(f"\n  Fixed-gain reached food: {baseline_metrics['reached_food']}", flush=True)
    print(f"  RL policy reached food: {rl_metrics['reached_food']}", flush=True)
    print(f"  Report: {report_path}", flush=True)

    # ── Plot ─────────────────────────────────────────────────────────
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plot_learning_curve(history, plot_path)

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DELIVERABLES SAVED TO: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
