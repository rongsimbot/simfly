#!/usr/bin/env python3
"""
Phase B: LIF Parameter Optimization via PPO
===========================================
Optimizes NIRON neuron parameters using the RL framework from Phase A.
RL tunes per-NT-type leak rates, refractory delays, and weight scaling
while the connectome drives ALL movement.

SCIENTIFIC RIGOR: Connectome ALWAYS drives movement — RL only tunes LIF params.

Parameters Optimized (18 total):
  - leak_rate (6): ACH, GABA, GLUT, DA, OCT, SER
  - refractory_delay (6): per NT type
  - weight_scale (6): per NT type synaptic weight multiplier

Approach:
  1. Policy outputs 18 continuous LIF params each RL step
  2. Params applied to C++ engine via set_neuron() (no engine rebuild)
  3. Reward: upright stability + joint efficiency (same as refocused Phase A)
  4. Same PPO framework (numpy, analytic gradients, GAE)

USAGE (on GB10):
    DISPLAY=:10 MUJOCO_GL=egl \
    /home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 \
    phase_b_train.py --iterations 50 --bfs-hops 1 --rollout 1024

Outputs:
  phase_b_output/
    ├── train_log.jsonl        — training metrics per iteration
    ├── snapshots/             — policy weights per iteration
    ├── best_policy.npz        — best-performing policy
    ├── baseline_metrics.json  — hand-tuned baseline metrics
    ├── rl_optimized_metrics.json — RL-optimized metrics
    └── comparison_report.json — before/after comparison
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Force unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
RL_TRAINING_DIR = os.path.join(CODE_ROOT, "rl_training")

for d in [CODE_ROOT, RL_TRAINING_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from rl_bridge import RLConfig, MLP, Adam
from rl_simfly_pipeline import SimFlyRLPipeline, ACTIVE_LEG_JOINTS

# ── NT Types and Hand-Tuned Parameters ─────────────────────────────────
NT_TYPES = ['ACH', 'GABA', 'GLUT', 'DA', 'OCT', 'SER']

# Current hand-tuned values (baseline)
HAND_TUNED = {
    'leak_rate':    {'ACH': 0.12, 'GABA': 0.20, 'GLUT': 0.15, 'DA': 0.10, 'OCT': 0.10, 'SER': 0.18},
    'refractory':   {'ACH': 0,    'GABA': 0,    'GLUT': 0,    'DA': 0,    'OCT': 0,    'SER': 0},
    'weight_scale': {'ACH': 1.5,  'GABA': -0.5, 'GLUT': -0.25, 'DA': 0.75, 'OCT': 0.75, 'SER': -0.15},
}

# Parameter bounds for RL optimization
PARAM_BOUNDS = {
    'leak_rate':    (0.01, 0.50),    # Leak rate range
    'refractory':   (0, 10),         # Refractory delay (cycles)
    'weight_scale': (-2.0, 3.0),     # Weight scaling multiplier
}

# ── LIF Parameter Config ───────────────────────────────────────────────
@dataclass
class LIFConfig:
    """Configuration for LIF parameter optimization."""
    n_joints: int = 36
    n_nt_types: int = 6
    n_params_per_nt: int = 3       # leak_rate, refractory_delay, weight_scale
    action_dim: int = field(init=False)
    obs_dim: int = field(init=False)

    # PPO hyperparams
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    rollout_steps: int = 1024
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_ep_steps: int = 400
    seed: int = 42

    def __post_init__(self) -> None:
        self.action_dim = self.n_nt_types * self.n_params_per_nt  # 18
        self.obs_dim = 2 * self.n_joints + 3  # 75 (same as Phase A)


# ── LIF Parameter Policy ───────────────────────────────────────────────
class LIFParameterPolicy:
    """PPO policy that outputs 18 continuous LIF parameters.

    Action space (18 dims): [leak_ACH, leak_GABA, ..., ref_ACH, ..., wscale_ACH, ...]
    Output is raw (pre-squash); squashing maps to valid parameter ranges.
    """

    def __init__(self, config: LIFConfig):
        self.cfg = config
        self.act_dim = config.action_dim
        self.actor = MLP(config.obs_dim, config.hidden, self.act_dim, seed=config.seed)
        self.critic = MLP(config.obs_dim, config.hidden, 1, seed=config.seed + 1)
        self.log_std = np.full(self.act_dim, -0.5, dtype=float)
        self.actor_opt = Adam(self.actor.p, lr=config.lr)
        self.critic_opt = Adam(self.critic.p, lr=config.lr)
        self.logstd_m = np.zeros(self.act_dim)
        self.logstd_v = np.zeros(self.act_dim)
        self._ls_t = 0
        self.rng = np.random.default_rng(config.seed)

        # Precompute bounds for squashing
        self._bounds = []
        for nt in NT_TYPES:
            for param in ['leak_rate', 'refractory', 'weight_scale']:
                lo, hi = PARAM_BOUNDS[param]
                self._bounds.append((lo, hi))

    def squash(self, raw: np.ndarray) -> np.ndarray:
        """Map raw NN output into valid parameter ranges using sigmoid scaling."""
        raw = np.atleast_1d(np.asarray(raw, dtype=float))
        squashed = np.zeros(self.act_dim)
        for i, (lo, hi) in enumerate(self._bounds):
            sig = 1.0 / (1.0 + np.exp(-raw[i]))
            squashed[i] = lo + (hi - lo) * sig
        return squashed

    def unsquash(self, squashed: np.ndarray) -> np.ndarray:
        """Inverse of squash (for logging)."""
        squashed = np.atleast_1d(np.asarray(squashed, dtype=float))
        raw = np.zeros(self.act_dim)
        for i, (lo, hi) in enumerate(self._bounds):
            s = (squashed[i] - lo) / (hi - lo + 1e-8)
            s = np.clip(s, 1e-8, 1 - 1e-8)
            raw[i] = np.log(s / (1 - s))
        return raw

    def params_to_dict(self, squashed: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert squashed action vector to structured parameter dict."""
        result = {'leak_rate': {}, 'refractory': {}, 'weight_scale': {}}
        k = 0
        for param in ['leak_rate', 'refractory', 'weight_scale']:
            for nt in NT_TYPES:
                result[param][nt] = float(squashed[k])
                k += 1
        return result

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Return (squashed_action, logprob, value, raw_action)."""
        mean = self.actor.forward(obs)[0]
        std = np.exp(self.log_std)
        raw = mean + std * self.rng.standard_normal(self.act_dim)
        logp = self._gauss_logp(raw, mean, std)
        value = float(self.critic.forward(obs)[0, 0])
        squashed = self.squash(raw)
        return squashed, float(logp), value, raw

    @staticmethod
    def _gauss_logp(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        var = std ** 2
        return np.sum(-0.5 * ((x - mean) ** 2) / var - np.log(std) - 0.5 * np.log(2 * np.pi), axis=-1)

    def evaluate(self, obs: np.ndarray, raw_actions: np.ndarray):
        mean = self.actor.forward(obs)
        std = np.exp(self.log_std)
        logp = self._gauss_logp(raw_actions, mean, std)
        values = self.critic.forward(obs)[:, 0]
        entropy = np.sum(self.log_std + 0.5 * np.log(2 * np.pi * np.e))
        return logp, values, entropy, mean, std

    def get_params(self) -> Dict[str, np.ndarray]:
        d = {f"actor_{k}": v.copy() for k, v in self.actor.p.items()}
        d.update({f"critic_{k}": v.copy() for k, v in self.critic.p.items()})
        d["log_std"] = self.log_std.copy()
        return d

    def set_params(self, d: Dict[str, np.ndarray]) -> None:
        for k in self.actor.p:
            self.actor.p[k] = d[f"actor_{k}"].copy()
        for k in self.critic.p:
            self.critic.p[k] = d[f"critic_{k}"].copy()
        self.log_std = d["log_std"].copy()

    def save(self, path: str) -> None:
        np.savez(path, **self.get_params())

    def _logstd_adam_step(self, grad: np.ndarray, lr: float) -> None:
        self._ls_t += 1
        self.logstd_m = 0.9 * self.logstd_m + 0.1 * grad
        self.logstd_v = 0.999 * self.logstd_v + 0.001 * grad ** 2
        mh = self.logstd_m / (1 - 0.9 ** self._ls_t)
        vh = self.logstd_v / (1 - 0.999 ** self._ls_t)
        self.log_std -= lr * mh / (np.sqrt(vh) + 1e-8)
        self.log_std = np.clip(self.log_std, -3.0, 1.0)


# ── LIF Environment Wrapper ────────────────────────────────────────────
class LIFSimFlyEnv:
    """Wraps SimFlyRLPipeline for LIF parameter optimization.

    PER-EPISODE PARAMETER APPLICATION (performance-critical):
    - Policy outputs 18 LIF params each step (PPO requirement)
    - Params are only APPLIED to the engine at episode start (reset)
    - This avoids O(N) set_neuron() calls per step for N neurons
    - Weight scaling is applied to torques each step (cheap)

    Scientific justification: LIF parameters are neuron properties that
    shouldn't change every 5ms — they're optimized per-episode.
    """

    def __init__(self, pipeline: SimFlyRLPipeline, lif_config: LIFConfig):
        self.pipeline = pipeline
        self.lif_cfg = lif_config
        self.cfg = lif_config
        self._steps = 0
        self._prev_z = None
        self._prev_food = None
        self._current_params: Optional[Dict] = None
        self._policy: Optional[LIFParameterPolicy] = None

        # Cache engine neuron indices by NT type for fast batch updates
        self._nt_neuron_indices: Dict[str, List[int]] = {}
        if pipeline._initialized and hasattr(pipeline, 'neuron_nt_types'):
            for eng_idx, fw_id in enumerate(pipeline._idx_to_flywire):
                nt = pipeline.neuron_nt_types.get(fw_id, 'ACH')
                if nt not in self._nt_neuron_indices:
                    self._nt_neuron_indices[nt] = []
                self._nt_neuron_indices[nt].append(eng_idx)
            print(f"  [LIF-ENV] Cached {sum(len(v) for v in self._nt_neuron_indices.values()):,} neuron indices by NT type")

    def reset(self, lif_action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Reset simulation and apply new LIF parameters.

        Args:
            lif_action: 18-dim squashed parameter vector to apply at episode start.
                       If None, keeps current params.
        """
        self.pipeline.reset()
        self._steps = 0

        # Apply new LIF params at episode start (ONCE per episode)
        if lif_action is not None:
            params = self._parse_action(lif_action)
            self.apply_lif_params(params)

        st = self.pipeline.get_state()
        self._prev_z = st.get('z_height', 0.06)
        self._prev_food = st.get('food_distance')
        obs = self.pipeline.get_observation().astype(np.float32)
        return obs, {}

    def apply_lif_params(self, params: Dict[str, Dict[str, float]]) -> None:
        """Apply LIF parameters to the C++ engine via set_neuron().

        Updates leak_rate and refractory_delay for each neuron based on
        its neurotransmitter type. This is O(N) and should be called
        sparingly (once per episode).
        """
        eng = self.pipeline.cpp_eng
        if eng is None:
            return

        leak_rates = params['leak_rate']
        refractories = params['refractory']

        for nt, indices in self._nt_neuron_indices.items():
            leak = float(leak_rates.get(nt, 0.03))
            ref = int(refractories.get(nt, 0))
            for idx in indices:
                eng.set_neuron(idx, model=3, leak=leak, refractory=ref)

        self._current_params = params

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one simulation step.

        LIF params are NOT applied here (applied at episode start only).
        The action is used to update the policy's action distribution but
        does not change engine parameters mid-episode.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Get connectome torques (connectome ALWAYS drives movement)
        torques = self.pipeline.get_connectome_torques()

        # Apply weight scaling modulation (cheap: just 36 multiplications)
        if self._current_params:
            wscales = self._current_params['weight_scale']
            mean_wscale = float(np.mean(list(wscales.values())))
        else:
            mean_wscale = 1.0

        modulated = {j: float(np.clip(float(t) * mean_wscale, -1.0, 1.0))
                     for j, t in torques.items()}

        self.pipeline.apply_torques(modulated)
        self.pipeline.step_physics()

        st = self.pipeline.get_state()
        obs = self.pipeline.get_observation().astype(np.float32)

        # ── Stability-based Reward (same as refocused Phase A) ──
        reward, terminated = self._compute_reward(st)

        self._steps += 1
        truncated = self._steps >= self.lif_cfg.max_ep_steps

        info = {
            "z_height": st.get("z_height", 0.0),
            "upright": st.get("upright", 0.0),
            "x_velocity": st.get("x_velocity", 0.0),
        }
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: np.ndarray) -> Dict:
        """Parse 18-dim action into structured params."""
        a = np.asarray(action, dtype=float)
        k = 0
        result = {'leak_rate': {}, 'refractory': {}, 'weight_scale': {}}
        for param in ['leak_rate', 'refractory', 'weight_scale']:
            for nt in NT_TYPES:
                result[param][nt] = float(a[k])
                k += 1
        return result

    def _compute_reward(self, st: Dict[str, float]) -> Tuple[float, bool]:
        """Stability-based reward function.

        Components:
        - upright_bonus: reward for staying upright (z-height maintenance)
        - joint_efficiency: penalty for excessive torque
        - alive_bonus: small survival reward
        - fall_penalty: heavy penalty for falling
        - energy_penalty: small penalty for very high neuron activity
        """
        z_height = float(st.get('z_height', 0.06))
        upright = float(st.get('upright', 0.0))
        fell = bool(st.get('fell', False))

        # Upright bonus: maximize z-height stability
        upright_bonus = 2.0 * upright

        # Joint efficiency: reward moderate joint activation, penalize extremes
        active_joints = 0
        if hasattr(self.pipeline, 'metrics') and self.pipeline.metrics.get('active_joints'):
            aj = self.pipeline.metrics['active_joints']
            if aj:
                active_joints = aj[-1] if aj else 0
        efficiency = -0.01 * max(0, active_joints - 10)  # Penalize >10 active joints

        # Alive bonus
        alive = 0.05

        # Fall penalty
        fall_penalty = -10.0 if fell else 0.0

        # Energy penalty (penalize excessive neuron firing for efficiency)
        energy_penalty = 0.0
        if hasattr(self.pipeline, 'metrics') and self.pipeline.metrics.get('fired_neurons'):
            fn = self.pipeline.metrics['fired_neurons']
            if fn:
                n_fired = fn[-1] if fn else 0
                # Penalty only above threshold
                energy_penalty = -0.0001 * max(0, n_fired - 500)

        total = upright_bonus + efficiency + alive + fall_penalty + energy_penalty
        return float(total), fell

    def set_policy(self, policy: LIFParameterPolicy) -> None:
        """Store policy reference for action parsing."""
        self._policy = policy


# ── PPO Trainer for LIF Parameters ─────────────────────────────────────
def train_lif_ppo(env: LIFSimFlyEnv, policy: LIFParameterPolicy, config: LIFConfig,
                  n_iterations: int, log_path: str) -> List[Dict[str, Any]]:
    """Train PPO for LIF parameter optimization.

    Same GAE+clipped-surrogate PPO as Phase A, adapted for 18-dim action space.

    KEY DESIGN: LIF params are applied ONCE per episode (at reset).
    The policy still produces actions every step for PPO, but only the
    first action after reset is used for engine parameter updates.
    This makes training practical (no 30K×set_neuron per step).
    """
    snap_dir = log_path + ".snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)

    T = config.rollout_steps
    obs_dim = config.obs_dim
    act_dim = config.action_dim
    history: List[Dict[str, Any]] = []

    for it in range(n_iterations):
        t0 = time.perf_counter()

        obs_buf = np.zeros((T, obs_dim), dtype=np.float32)
        raw_buf = np.zeros((T, act_dim), dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)
        val_buf = np.zeros(T, dtype=np.float32)
        logp_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)

        # Get initial LIF params and reset with them
        obs_zero = np.zeros(obs_dim, dtype=np.float32)  # Start with zero obs for initial param selection
        squashed_init, _, _, raw_init = policy.act(obs_zero)
        obs, _ = env.reset(lif_action=squashed_init)

        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        cur_r, cur_l = 0.0, 0
        param_history: List[Dict] = []
        param_history.append(policy.params_to_dict(squashed_init))

        for t in range(T):
            squashed, logp, value, raw = policy.act(obs)
            obs_buf[t] = obs
            raw_buf[t] = raw
            val_buf[t] = value
            logp_buf[t] = logp

            # Step: LIF params NOT applied here (applied at episode start only)
            # The 'action' is passed for PPO logging but env ignores it mid-episode
            obs, rew, terminated, truncated, _ = env.step(squashed)
            rew_buf[t] = rew
            cur_r += rew
            cur_l += 1
            done = terminated or truncated
            done_buf[t] = 1.0 if done else 0.0
            if done:
                ep_rewards.append(cur_r)
                ep_lengths.append(cur_l)
                cur_r, cur_l = 0.0, 0
                # New episode: sample new LIF params and apply at reset
                squashed_new, _, _, _ = policy.act(obs)
                obs, _ = env.reset(lif_action=squashed_new)
                param_history.append(policy.params_to_dict(squashed_new))

        # Value of final observation (may be post-reset from episode end)
        if done:  # obs is from reset of new episode
            last_val = float(policy.critic.forward(obs)[0, 0])
        else:
            last_val = float(policy.critic.forward(obs)[0, 0])

        # ── GAE ──
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            next_nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + config.gamma * next_val * next_nonterm - val_buf[t]
            last_gae = delta + config.gamma * config.lam * next_nonterm * last_gae
            adv[t] = last_gae
        returns = adv + val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── PPO epochs over minibatches ──
        idx = np.arange(T)
        for _ in range(config.epochs):
            np.random.shuffle(idx)
            for s in range(0, T, config.minibatch):
                mb = idx[s:s + config.minibatch]
                o, r_act = obs_buf[mb], raw_buf[mb]
                A, R, old_lp = adv[mb], returns[mb], logp_buf[mb]

                mean = policy.actor.forward(o)
                std = np.exp(policy.log_std)
                var = std ** 2
                new_lp = policy._gauss_logp(r_act, mean, std)
                ratio = np.exp(new_lp - old_lp)

                # Clipped surrogate
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - config.clip, 1 + config.clip) * A
                use_unclipped = (unclipped <= clipped).astype(float)

                B = o.shape[0]
                coeff = -(use_unclipped * ratio * A) / B
                d_logp_d_mean = (r_act - mean) / var
                d_mean = coeff[:, None] * d_logp_d_mean
                actor_grads = policy.actor.backward(d_mean)

                # Log-std gradient
                d_logp_d_logstd = ((r_act - mean) ** 2 / var - 1.0)
                g_logstd = np.sum(coeff[:, None] * d_logp_d_logstd, axis=0)
                g_logstd -= config.ent_coef

                # Critic
                v_pred = policy.critic.forward(o)[:, 0]
                d_v = (config.vf_coef * 2.0 * (v_pred - R) / B)[:, None]
                critic_grads = policy.critic.backward(d_v)

                policy.actor_opt.step(actor_grads)
                policy.critic_opt.step(critic_grads)
                policy._logstd_adam_step(g_logstd, config.lr)

        elapsed = time.perf_counter() - t0
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else float(rew_buf.sum())

        # Compute mean parameters across rollout (only episode-start params)
        if param_history:
            mean_leak = {nt: float(np.mean([p['leak_rate'][nt] for p in param_history]))
                         for nt in NT_TYPES}
            mean_ref = {nt: float(np.mean([p['refractory'][nt] for p in param_history]))
                        for nt in NT_TYPES}
            mean_wscale = {nt: float(np.mean([p['weight_scale'][nt] for p in param_history]))
                           for nt in NT_TYPES}
        else:
            mean_leak = {nt: 0.03 for nt in NT_TYPES}
            mean_ref = {nt: 0 for nt in NT_TYPES}
            mean_wscale = {nt: 1.0 for nt in NT_TYPES}

        rec = {
            "iteration": it,
            "mean_episode_reward": mean_reward,
            "mean_step_reward": float(rew_buf.mean()),
            "n_episodes": len(ep_rewards),
            "mean_episode_len": float(np.mean(ep_lengths)) if ep_lengths else float(T),
            "mean_leak_rate": mean_leak,
            "mean_refractory": mean_ref,
            "mean_weight_scale": mean_wscale,
            "mean_log_std": float(policy.log_std.mean()),
            "elapsed_sec": round(elapsed, 1),
        }
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        policy.save(os.path.join(snap_dir, f"iter_{it:04d}.npz"))

        # Progress output
        leak_str = ", ".join(f"{nt}={mean_leak[nt]:.3f}" for nt in NT_TYPES[:3])
        print(f"[iter {it:3d}] ep_reward={mean_reward:8.2f} "
              f"eps={len(ep_rewards)} "
              f"leak=[{leak_str}] "
              f"wscale_mean={np.mean(list(mean_wscale.values())):.3f} "
              f"logstd={rec['mean_log_std']:+.3f} "
              f"{elapsed:.0f}s",
              flush=True)
    return history


# ── Baseline Evaluator ─────────────────────────────────────────────────
def run_hand_tuned_baseline(pipeline: SimFlyRLPipeline, n_steps: int = 400) -> Dict:
    """Run simulation with hand-tuned LIF parameters and measure stability."""
    print(f"\n[BASELINE] Running hand-tuned LIF parameters ({n_steps} steps)...", flush=True)
    pipeline.reset()

    # Apply hand-tuned params
    eng = pipeline.cpp_eng
    if eng is not None and hasattr(pipeline, 'neuron_nt_types'):
        for eng_idx, fw_id in enumerate(pipeline._idx_to_flywire):
            nt = pipeline.neuron_nt_types.get(fw_id, 'ACH')
            leak = HAND_TUNED['leak_rate'].get(nt, 0.03)
            ref = int(HAND_TUNED['refractory'].get(nt, 0))
            eng.set_neuron(eng_idx, model=3, leak=leak, refractory=ref)

    z_heights = []
    uprights = []
    torques_mag = []
    n_fired = []
    n_active_joints = []
    total_reward = 0.0
    fell_at = -1

    for step in range(n_steps):
        torques = pipeline.get_connectome_torques()
        # Apply hand-tuned weight scaling (mean)
        mean_wscale = float(np.mean(list(HAND_TUNED['weight_scale'].values())))
        modulated = {j: float(np.clip(float(t) * mean_wscale, -1.0, 1.0))
                     for j, t in torques.items()}
        pipeline.apply_torques(modulated)
        pipeline.step_physics()

        st = pipeline.get_state()
        z_heights.append(st['z_height'])
        uprights.append(st['upright'])
        torques_mag.append(float(np.mean([abs(v) for v in modulated.values()])))

        if pipeline.metrics.get('fired_neurons'):
            n_fired.append(pipeline.metrics['fired_neurons'][-1])
        if pipeline.metrics.get('active_joints'):
            n_active_joints.append(pipeline.metrics['active_joints'][-1])

        # Same reward as RL
        r, fell = _baseline_reward(st, pipeline)
        total_reward += r
        if fell and fell_at < 0:
            fell_at = step
            break

    return {
        'method': 'hand_tuned',
        'steps_run': step + 1,
        'total_reward': float(total_reward),
        'mean_z_height': float(np.mean(z_heights)),
        'z_height_std': float(np.std(z_heights)),
        'mean_upright': float(np.mean(uprights)),
        'mean_torque_magnitude': float(np.mean(torques_mag)),
        'torque_std': float(np.std(torques_mag)),
        'mean_fired_neurons': float(np.mean(n_fired)) if n_fired else 0,
        'mean_active_joints': float(np.mean(n_active_joints)) if n_active_joints else 0,
        'fell_at_step': fell_at,
        'fell': fell_at >= 0,
        'params': HAND_TUNED,
    }


def _baseline_reward(st: Dict, pipeline) -> Tuple[float, bool]:
    """Same reward computation as RL env for fair comparison."""
    upright = float(st.get('upright', 0.0))
    fell = bool(st.get('fell', False))
    upright_bonus = 2.0 * upright
    alive = 0.05
    fall_penalty = -10.0 if fell else 0.0
    total = upright_bonus + alive + fall_penalty
    return float(total), fell


def run_rl_optimized(pipeline: SimFlyRLPipeline, policy: LIFParameterPolicy,
                     lif_config: LIFConfig, n_steps: int = 400) -> Dict:
    """Evaluate RL-optimized LIF parameters.

    Runs one episode with RL-optimized params applied at episode start.
    """
    print(f"\n[RL-OPT] Running RL-optimized LIF parameters ({n_steps} steps)...", flush=True)

    # Get RL-optimized params from policy (deterministic: mean only)
    obs_zero = np.zeros(lif_config.obs_dim, dtype=np.float32)
    mean_raw = policy.actor.forward(obs_zero)[0]
    squashed = policy.squash(mean_raw)
    params = policy.params_to_dict(squashed)

    # Apply params once at episode start
    pipeline.reset()
    _apply_params_to_engine(pipeline, params)
    wscales = params['weight_scale']
    mean_wscale = float(np.mean(list(wscales.values())))

    z_heights = []
    uprights = []
    torques_mag = []
    n_fired = []
    n_active_joints = []
    total_reward = 0.0
    fell_at = -1

    for step in range(n_steps):
        torques = pipeline.get_connectome_torques()
        modulated = {j: float(np.clip(float(t) * mean_wscale, -1.0, 1.0))
                     for j, t in torques.items()}
        pipeline.apply_torques(modulated)
        pipeline.step_physics()

        st = pipeline.get_state()
        z_heights.append(st['z_height'])
        uprights.append(st['upright'])
        torques_mag.append(float(np.mean([abs(v) for v in modulated.values()])))

        if pipeline.metrics.get('fired_neurons'):
            n_fired.append(pipeline.metrics['fired_neurons'][-1])
        if pipeline.metrics.get('active_joints'):
            n_active_joints.append(pipeline.metrics['active_joints'][-1])

        r, fell = _baseline_reward(st, pipeline)
        total_reward += r
        if fell and fell_at < 0:
            fell_at = step
            break

    # Use the params directly (applied at episode start, not averaged)
    opt_params = params

    return {
        'method': 'rl_optimized',
        'steps_run': step + 1,
        'total_reward': float(total_reward),
        'mean_z_height': float(np.mean(z_heights)),
        'z_height_std': float(np.std(z_heights)),
        'mean_upright': float(np.mean(uprights)),
        'mean_torque_magnitude': float(np.mean(torques_mag)),
        'torque_std': float(np.std(torques_mag)),
        'mean_fired_neurons': float(np.mean(n_fired)) if n_fired else 0,
        'mean_active_joints': float(np.mean(n_active_joints)) if n_active_joints else 0,
        'fell_at_step': fell_at,
        'fell': fell_at >= 0,
        'params': opt_params,
    }


def _apply_params_to_engine(pipeline, params):
    """Apply LIF params to engine neurons."""
    eng = pipeline.cpp_eng
    if eng is None:
        return
    for eng_idx, fw_id in enumerate(pipeline._idx_to_flywire):
        nt = pipeline.neuron_nt_types.get(fw_id, 'ACH')
        leak = float(params['leak_rate'].get(nt, 0.03))
        ref = int(params['refractory'].get(nt, 0))
        eng.set_neuron(eng_idx, model=3, leak=leak, refractory=ref)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase B: LIF Parameter Optimization via PPO")
    parser.add_argument('--iterations', type=int, default=50, help='PPO iterations')
    parser.add_argument('--bfs-hops', type=int, default=1, help='BFS hops for neuron selection')
    parser.add_argument('--neurons', type=int, default=0, help='Max neurons (0=all, only used when bfs-hops=0)')
    parser.add_argument('--joints', type=int, default=36, help='Active leg joints')
    parser.add_argument('--rollout', type=int, default=1024, help='Rollout steps per iteration')
    parser.add_argument('--max-ep-steps', type=int, default=400, help='Max episode steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--syn-threshold', type=int, default=5, help='Min synapse count')
    parser.add_argument('--brain-steps', type=int, default=2, help='Brain substeps')
    parser.add_argument('--eval-only', type=str, help='Evaluate saved policy (path to .npz)')
    parser.add_argument('--compare-only', action='store_true', help='Only baseline, no training')
    args = parser.parse_args()

    # ── Output Setup ──────────────────────────────────────────────────
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(CODE_ROOT, "phase_b_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────
    lif_config = LIFConfig(
        n_joints=args.joints,
        hidden=args.hidden,
        lr=args.lr,
        rollout_steps=args.rollout,
        max_ep_steps=args.max_ep_steps,
        seed=args.seed,
    )

    rl_config = RLConfig(
        n_joints=args.joints,
        hidden=args.hidden,
        lr=args.lr,
        rollout_steps=args.rollout,
        max_ep_steps=args.max_ep_steps,
        seed=args.seed,
    )

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE B: LIF PARAMETER OPTIMIZATION VIA PPO", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Action dim: {lif_config.action_dim} (6 NT × 3 params)", flush=True)
    print(f"  Observation dim: {lif_config.obs_dim}", flush=True)
    print(f"  Network: MLP {lif_config.obs_dim}→{args.hidden}→{args.hidden}→{lif_config.action_dim}", flush=True)
    print(f"  BFS hops: {args.bfs_hops}, Syn threshold: {args.syn_threshold}", flush=True)
    print(f"  Iterations: {args.iterations} × {args.rollout} rollout steps", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Initialize Pipeline ───────────────────────────────────────────
    print("[1/4] Initializing connectome pipeline (Phase B)...", flush=True)
    t0 = time.perf_counter()
    pipeline = SimFlyRLPipeline(
        rl_config,
        max_neurons=args.neurons,
        bfs_hops=args.bfs_hops,
        syn_threshold=args.syn_threshold,
        brain_steps=args.brain_steps,
    )
    pipeline.initialize(verbose=True)
    print(f"  Pipeline init: {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  Neurons: {len(pipeline._idx_to_flywire):,}", flush=True)

    # ── Eval-Only Mode ────────────────────────────────────────────────
    if args.eval_only:
        print(f"\n[EVAL] Loading policy from: {args.eval_only}", flush=True)
        policy = LIFParameterPolicy(lif_config)
        data = np.load(args.eval_only, allow_pickle=True)
        policy.set_params({k: data[k] for k in data.files})
        metrics = run_rl_optimized(pipeline, policy, lif_config, n_steps=400)
        print(f"\n[EVAL] RL-optimized: reward={metrics['total_reward']:.1f}, "
              f"z_height={metrics['mean_z_height']:.3f}±{metrics['z_height_std']:.3f}, "
              f"upright={metrics['mean_upright']:.3f}, "
              f"fell={metrics['fell']}", flush=True)
        return 0

    # ── Run Hand-Tuned Baseline ───────────────────────────────────────
    print("\n[2/4] Running hand-tuned LIF baseline...", flush=True)
    baseline_metrics = run_hand_tuned_baseline(pipeline, n_steps=400)
    print(f"  Hand-tuned: reward={baseline_metrics['total_reward']:.1f}, "
          f"z_height={baseline_metrics['mean_z_height']:.3f}±{baseline_metrics['z_height_std']:.3f}, "
          f"upright={baseline_metrics['mean_upright']:.3f}, "
          f"fell={baseline_metrics['fell']}", flush=True)

    # Save baseline
    with open(os.path.join(output_dir, "baseline_metrics.json"), 'w') as f:
        json.dump(baseline_metrics, f, indent=2, default=str)

    if args.compare_only:
        print("\n  Baseline saved. Exiting (--compare-only).", flush=True)
        return 0

    # ── Train RL Policy ──────────────────────────────────────────────
    print("\n[3/4] Training LIF parameter optimization policy...", flush=True)
    env = LIFSimFlyEnv(pipeline, lif_config)
    policy = LIFParameterPolicy(lif_config)
    env.set_policy(policy)

    log_path = os.path.join(output_dir, "train_log.jsonl")
    t_start = time.perf_counter()

    print("  Starting PPO iterations...", flush=True)
    history = train_lif_ppo(env, policy, lif_config, n_iterations=args.iterations, log_path=log_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Training complete: {elapsed:.1f}s ({elapsed/max(1,args.iterations):.1f}s/iter)", flush=True)

    # Save best policy
    best_path = os.path.join(output_dir, "best_policy.npz")
    policy.save(best_path)
    print(f"  Best policy saved: {best_path}", flush=True)

    # ── Evaluate RL Policy ───────────────────────────────────────────
    print("\n[4/4] Running RL-optimized evaluation...", flush=True)
    rl_metrics = run_rl_optimized(pipeline, policy, lif_config, n_steps=400)
    print(f"  RL-optimized: reward={rl_metrics['total_reward']:.1f}, "
          f"z_height={rl_metrics['mean_z_height']:.3f}±{rl_metrics['z_height_std']:.3f}, "
          f"upright={rl_metrics['mean_upright']:.3f}, "
          f"fell={rl_metrics['fell']}", flush=True)

    # Save RL metrics
    with open(os.path.join(output_dir, "rl_optimized_metrics.json"), 'w') as f:
        json.dump(rl_metrics, f, indent=2, default=str)

    # ── Comparison Report ────────────────────────────────────────────
    comparison = build_comparison(baseline_metrics, rl_metrics, args, elapsed)
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    # ── Print Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"PHASE B COMPLETE — LIF PARAMETER OPTIMIZATION", flush=True)
    print(f"{'='*60}", flush=True)

    bl = baseline_metrics
    rl = rl_metrics
    print(f"  {'Metric':<30s} {'Hand-Tuned':>12s} {'RL-Optimized':>12s} {'Change':>10s}", flush=True)
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}", flush=True)
    print(f"  {'Total Reward':<30s} {bl['total_reward']:>12.1f} {rl['total_reward']:>12.1f} "
          f"{rl['total_reward']-bl['total_reward']:>+10.1f}", flush=True)
    print(f"  {'Mean Z-Height':<30s} {bl['mean_z_height']:>12.4f} {rl['mean_z_height']:>12.4f} "
          f"{rl['mean_z_height']-bl['mean_z_height']:>+10.4f}", flush=True)
    print(f"  {'Z-Height Std':<30s} {bl['z_height_std']:>12.4f} {rl['z_height_std']:>12.4f} "
          f"{rl['z_height_std']-bl['z_height_std']:>+10.4f}", flush=True)
    print(f"  {'Mean Upright':<30s} {bl['mean_upright']:>12.4f} {rl['mean_upright']:>12.4f} "
          f"{rl['mean_upright']-bl['mean_upright']:>+10.4f}", flush=True)
    print(f"  {'Torque Magnitude':<30s} {bl['mean_torque_magnitude']:>12.4f} {rl['mean_torque_magnitude']:>12.4f} "
          f"{rl['mean_torque_magnitude']-bl['mean_torque_magnitude']:>+10.4f}", flush=True)
    print(f"  {'Fired Neurons':<30s} {bl['mean_fired_neurons']:>12.1f} {rl['mean_fired_neurons']:>12.1f} "
          f"{rl['mean_fired_neurons']-bl['mean_fired_neurons']:>+10.1f}", flush=True)
    print(f"  {'Active Joints':<30s} {bl['mean_active_joints']:>12.1f} {rl['mean_active_joints']:>12.1f} "
          f"{rl['mean_active_joints']-bl['mean_active_joints']:>+10.1f}", flush=True)
    print(f"  {'Fell':<30s} {str(bl['fell']):>12s} {str(rl['fell']):>12s}", flush=True)
    print(f"\n  Report: {report_path}", flush=True)
    print(f"  Training: {elapsed:.1f}s total", flush=True)

    # ── Print optimized parameters ──
    print(f"\n  Optimized LIF Parameters:", flush=True)
    print(f"  {'NT Type':<8s} {'Leak (base→opt)':<25s} {'Ref (base→opt)':<20s} {'WScale (base→opt)':<25s}", flush=True)
    print(f"  {'─'*8} {'─'*25} {'─'*20} {'─'*25}", flush=True)
    for nt in NT_TYPES:
        bl_l = HAND_TUNED['leak_rate'][nt]
        rl_l = rl_metrics['params']['leak_rate'][nt]
        bl_r = HAND_TUNED['refractory'][nt]
        rl_r = rl_metrics['params']['refractory'][nt]
        bl_w = HAND_TUNED['weight_scale'][nt]
        rl_w = rl_metrics['params']['weight_scale'][nt]
        print(f"  {nt:<8s} {bl_l:.3f}→{rl_l:.3f} {'':>9s} {bl_r}→{rl_r:.1f} {'':>9s} {bl_w:.2f}→{rl_w:.2f}",
              flush=True)


def build_comparison(baseline: Dict, rl_metrics: Dict, args, elapsed: float) -> Dict:
    """Build structured comparison report."""
    improvement = {}
    for key in ['total_reward', 'mean_z_height', 'mean_upright', 'mean_torque_magnitude',
                'mean_fired_neurons', 'mean_active_joints']:
        if key in baseline and key in rl_metrics:
            bl = baseline[key]
            rl = rl_metrics[key]
            improvement[key] = {
                'hand_tuned': bl,
                'rl_optimized': rl,
                'delta': rl - bl,
                'pct_change': ((rl - bl) / (abs(bl) + 1e-8)) * 100,
            }

    return {
        'phase': 'B - LIF Parameter Optimization',
        'config': {
            'n_joints': args.joints,
            'bfs_hops': args.bfs_hops,
            'syn_threshold': args.syn_threshold,
            'brain_steps': args.brain_steps,
            'algorithm': 'PPO (numpy, analytic gradients, GAE)',
            'hidden': args.hidden,
            'lr': args.lr,
            'iterations': args.iterations,
            'rollout_steps': args.rollout,
            'max_ep_steps': args.max_ep_steps,
        },
        'baseline_hand_tuned': baseline,
        'rl_optimized': rl_metrics,
        'improvement': improvement,
        'training_duration_s': elapsed,
        'training_iterations': args.iterations,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }


if __name__ == "__main__":
    main()
