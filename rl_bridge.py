"""
rl_bridge.py — Reinforcement-Learning CALIBRATION layer for SimFly.

==============================================================================
SCIENTIFIC INTEGRITY CONTRACT (read before editing)
==============================================================================
The FlyWire connectome provides the ARCHITECTURE: which neurons connect to
which, and therefore which descending neurons (DNs) drive which motor neurons
(MNs) and ultimately which joint torques. That structural map is produced by
the EXISTING pipeline (engine.fire -> DN/MN bridge -> VNCMotorDecoder.decode).

This module does NOT generate motor commands from scratch. It learns a thin
per-joint affine CALIBRATION on top of the connectome-driven torques:

        modulated[j] = connectome_torque[j] * gain[j] + bias[j]

When gain == 1 and bias == 0 the connectome signal passes through UNCHANGED.
That identity is asserted by the smoke test. RL is therefore a *calibration*,
not a *substitute*, for the connectome.

HONESTY CONTROL (the part that makes this science, not theatre):
We provide `ShuffledConnectomePipeline`, a degree/magnitude-preserving but
structure-destroying permutation of the connectome->joint mapping. Train PPO
identically on (a) the real connectome and (b) the shuffled control. If the
connectome carries genuine motor information, the real pipeline must reach
higher reward / better sample efficiency than the shuffled control. If it does
NOT, we report that honestly — it means our connectome->torque path is not yet
contributing signal, and RL is just learning the body, not the brain.

This mirrors the FlyGM paper (arXiv:2602.17997), which validates a connectome
controller specifically by beating degree-preserving rewired and random graphs.

Dependencies: numpy, gymnasium. (mujoco optional, guarded.) NO torch, NO
stable-baselines3 — a compact analytic-gradient PPO is implemented in numpy.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _HAVE_GYM = False

try:
    import mujoco  # noqa: F401  (used by the real pipeline, not here)
except Exception:
    mujoco = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RLConfig:
    """Hyper-parameters + observation/action geometry for the RL calibration layer."""
    n_joints: int = 36
    obs_dim: int = field(init=False)
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    rollout_steps: int = 2048
    minibatch: int = 256
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_ep_steps: int = 400
    seed: int = 0
    action_scale_min: float = 0.1   # min multiplicative gain
    action_scale_max: float = 3.0   # max multiplicative gain
    modulation_range: float = 0.5   # +/- additive bias

    def __post_init__(self) -> None:
        # observation = per-joint angle + per-joint contact + 3 exteroceptive
        # scalars (vision contrast, wall distance, odor gradient).
        self.obs_dim = 2 * self.n_joints + 3


# ---------------------------------------------------------------------------
# Tiny numpy MLP with analytic backprop (tanh hidden, linear output)
# ---------------------------------------------------------------------------
class MLP:
    """2-hidden-layer tanh MLP. Caches activations for analytic backprop."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0, scale: float = 0.1):
        rng = np.random.default_rng(seed)
        self.p: Dict[str, np.ndarray] = {
            "W1": rng.standard_normal((in_dim, hidden)) * (scale / np.sqrt(in_dim)),
            "b1": np.zeros(hidden),
            "W2": rng.standard_normal((hidden, hidden)) * (scale / np.sqrt(hidden)),
            "b2": np.zeros(hidden),
            "W3": rng.standard_normal((hidden, out_dim)) * (scale / np.sqrt(hidden)),
            "b3": np.zeros(out_dim),
        }
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        z1 = x @ self.p["W1"] + self.p["b1"]; a1 = np.tanh(z1)
        z2 = a1 @ self.p["W2"] + self.p["b2"]; a2 = np.tanh(z2)
        out = a2 @ self.p["W3"] + self.p["b3"]
        self._cache = {"x": x, "a1": a1, "a2": a2}
        return out

    def backward(self, d_out: np.ndarray) -> Dict[str, np.ndarray]:
        """d_out: gradient of scalar loss wrt the linear output (batch, out_dim)."""
        x, a1, a2 = self._cache["x"], self._cache["a1"], self._cache["a2"]
        g: Dict[str, np.ndarray] = {}
        g["W3"] = a2.T @ d_out
        g["b3"] = d_out.sum(0)
        da2 = d_out @ self.p["W3"].T
        dz2 = da2 * (1 - a2 ** 2)
        g["W2"] = a1.T @ dz2
        g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.p["W2"].T
        dz1 = da1 * (1 - a1 ** 2)
        g["W1"] = x.T @ dz1
        g["b1"] = dz1.sum(0)
        return g


class Adam:
    """Adam over a dict of numpy parameter arrays (updates in place)."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 3e-4,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Policy: diagonal-Gaussian actor + separate critic, both numpy MLPs
# ---------------------------------------------------------------------------
class ConnectomeModulationPolicy:
    """
    Actor outputs a mean over the RAW pre-squash action (2*n_joints dims) plus a
    learned log_std. The raw action is squashed into per-joint (gain, bias).
    A separate critic MLP estimates state value. Real analytic gradients.
    """

    def __init__(self, config: RLConfig):
        self.cfg = config
        self.act_dim = 2 * config.n_joints
        self.actor = MLP(config.obs_dim, config.hidden, self.act_dim, seed=config.seed)
        self.critic = MLP(config.obs_dim, config.hidden, 1, seed=config.seed + 1)
        self.log_std = np.full(self.act_dim, -0.5, dtype=float)
        # one combined Adam per network + a slot for log_std
        self.actor_opt = Adam(self.actor.p, lr=config.lr)
        self.critic_opt = Adam(self.critic.p, lr=config.lr)
        self.logstd_m = np.zeros(self.act_dim)
        self.logstd_v = np.zeros(self.act_dim)
        self._ls_t = 0
        self.rng = np.random.default_rng(config.seed)

    # ----- squash raw -> (gain, bias) -------------------------------------
    def squash(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map raw actor output into gain in [min,max] and bias in [-range,range]."""
        raw = np.asarray(raw)
        n = self.cfg.n_joints
        sig = 1.0 / (1.0 + np.exp(-raw[..., :n]))
        gains = self.cfg.action_scale_min + (self.cfg.action_scale_max - self.cfg.action_scale_min) * sig
        biases = self.cfg.modulation_range * np.tanh(raw[..., n:])
        return gains, biases

    # ----- act (single obs) ----------------------------------------------
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Return (squashed_action[gains|biases], logprob, value, raw_action)."""
        mean = self.actor.forward(obs)[0]
        std = np.exp(self.log_std)
        raw = mean + std * self.rng.standard_normal(self.act_dim)
        logp = self._gauss_logp(raw, mean, std)
        value = float(self.critic.forward(obs)[0, 0])
        gains, biases = self.squash(raw)
        return np.concatenate([gains, biases]), float(logp), value, raw

    @staticmethod
    def _gauss_logp(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        var = std ** 2
        return np.sum(-0.5 * ((x - mean) ** 2) / var - np.log(std) - 0.5 * np.log(2 * np.pi), axis=-1)

    # ----- evaluate a batch of (obs, raw_action) --------------------------
    def evaluate(self, obs: np.ndarray, raw_actions: np.ndarray):
        """Return logprobs, values, entropy, and cached means for gradient calc."""
        mean = self.actor.forward(obs)              # (B, act_dim)
        std = np.exp(self.log_std)
        logp = self._gauss_logp(raw_actions, mean, std)
        values = self.critic.forward(obs)[:, 0]
        entropy = np.sum(self.log_std + 0.5 * np.log(2 * np.pi * np.e))
        return logp, values, entropy, mean, std

    # ----- snapshot I/O ----------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core integrity function
# ---------------------------------------------------------------------------
def apply_modulation(connectome_joint_torques: Dict[str, float], joint_names: List[str],
                     gains: np.ndarray, biases: np.ndarray) -> Dict[str, float]:
    """
    Affine calibration of connectome-driven torques.

        out[j] = connectome_torque[j] * gain[j] + bias[j]   (clipped to [-1, 1])

    INTEGRITY: if gains==1 and biases==0 for all joints, the returned dict is
    identical (within clipping) to the connectome torques — RL adds nothing.
    """
    out: Dict[str, float] = {}
    for i, name in enumerate(joint_names):
        base = float(connectome_joint_torques.get(name, 0.0))
        out[name] = float(np.clip(base * float(gains[i]) + float(biases[i]), -1.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Gymnasium environment wrapping ANY pipeline implementing the duck interface
# ---------------------------------------------------------------------------
_Base = gym.Env if _HAVE_GYM else object


class SimFlyRLEnv(_Base):
    """
    Wraps a pipeline exposing:
        reset(), get_observation()->np.ndarray, get_connectome_torques()->dict,
        joint_names:list[str], apply_torques(dict), step_physics(),
        get_state()->dict(x_velocity, upright, z_height, food_distance,
                          wall_min_distance, fell)
    and the .cfg (RLConfig) used to build it.
    """

    metadata = {"render_modes": []}

    def __init__(self, pipeline: "PipelineProtocol"):
        self.pipeline = pipeline
        self.cfg: RLConfig = pipeline.cfg
        if _HAVE_GYM:
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.cfg.obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2 * self.cfg.n_joints,), np.float32)
        self._policy_squash = None  # optionally injected for raw->squash in step
        self._steps = 0
        self._prev_food: Optional[float] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.pipeline.reset()
        self._steps = 0
        st = self.pipeline.get_state()
        self._prev_food = st.get("food_distance")
        return self.pipeline.get_observation().astype(np.float32), {}

    def set_squash(self, fn) -> None:
        """Inject the policy's squash so step() can accept RAW actions directly."""
        self._policy_squash = fn

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        n = self.cfg.n_joints
        if self._policy_squash is not None:
            gains, biases = self._policy_squash(action)
        else:  # action already squashed: [gains|biases]
            gains, biases = action[:n], action[n:]

        torques = self.pipeline.get_connectome_torques()
        modulated = apply_modulation(torques, self.pipeline.joint_names, gains, biases)
        self.pipeline.apply_torques(modulated)
        self.pipeline.step_physics()

        st = self.pipeline.get_state()
        obs = self.pipeline.get_observation().astype(np.float32)
        reward, terminated = self._reward(st)
        self._steps += 1
        truncated = self._steps >= self.cfg.max_ep_steps
        info = {"x_velocity": st.get("x_velocity", 0.0), "upright": st.get("upright", 0.0)}
        return obs, reward, terminated, truncated, info

    def _reward(self, st: Dict[str, float]) -> Tuple[float, bool]:
        fwd = float(np.clip(st.get("x_velocity", 0.0), -1.0, 2.0))
        upright = float(st.get("upright", 0.0))
        alive = 0.1
        food = st.get("food_distance")
        food_approach = 0.0
        if food is not None and self._prev_food is not None:
            food_approach = -(food - self._prev_food) * 2.0
        self._prev_food = food
        wall = st.get("wall_min_distance", 1.0)
        wall_penalty = -max(0.0, 0.05 - wall) * 5.0
        fell = bool(st.get("fell", False))
        fall_penalty = -5.0 if fell else 0.0
        total = fwd + 0.5 * upright + alive + food_approach + wall_penalty + fall_penalty
        return float(total), fell


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
class MockPipeline:
    """
    TEST DOUBLE — synthetic dynamics, NO MuJoCo, NOT for scientific claims.

    Produces deterministic pseudo-connectome torques and integrates a toy body:
    forward velocity responds to the mean *useful* applied torque, and the body
    tips over if torques are applied too aggressively. Lets us validate the full
    RL loop offline. There is a hidden "correct" gain pattern the policy can
    discover, so the smoke test shows learning is possible end-to-end.
    """

    def __init__(self, config: RLConfig):
        self.cfg = config
        self.joint_names = [f"joint_{i}" for i in range(config.n_joints)]
        self._rng = np.random.default_rng(config.seed)
        # fixed per-joint "true" connectome drive (the architecture)
        self._drive = self._rng.uniform(-0.6, 0.6, config.n_joints)
        # hidden ideal gain the RL layer should find (calibration target)
        self._ideal_gain = self._rng.uniform(0.5, 1.5, config.n_joints)
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._x = 0.0
        self._v = 0.0
        self._upright = 1.0
        self._fell = False
        self._food = 1.0
        self._last_applied = np.zeros(self.cfg.n_joints)

    def get_observation(self) -> np.ndarray:
        ang = np.tanh(self._last_applied)
        contact = (self._last_applied > 0).astype(float)
        extero = np.array([0.3, min(1.0, 0.5 + self._x * 0.0), self._food], dtype=float)
        return np.concatenate([ang, contact, extero]).astype(np.float32)

    def get_connectome_torques(self) -> Dict[str, float]:
        # mild time modulation so it's not perfectly static
        phase = 0.2 * np.sin(self._t * 0.1)
        return {n: float(self._drive[i] + phase * 0.1) for i, n in enumerate(self.joint_names)}

    def apply_torques(self, torques: Dict[str, float]) -> None:
        self._last_applied = np.array([torques.get(n, 0.0) for n in self.joint_names])

    def step_physics(self) -> None:
        self._t += 1
        applied = self._last_applied
        # "useful" component: alignment of applied torque with ideal calibration
        ideal = self._drive * self._ideal_gain
        useful = float(np.dot(applied, ideal) / (np.linalg.norm(ideal) + 1e-8) / self.cfg.n_joints)
        excess = float(np.mean(np.abs(applied)))
        self._v = 0.9 * self._v + 0.5 * useful
        self._x += self._v
        self._food = max(0.0, self._food - max(0.0, self._v) * 0.01)
        self._upright -= max(0.0, excess - 0.6) * 0.05
        self._upright = float(np.clip(self._upright + 0.01, 0.0, 1.0))
        if self._upright <= 0.0:
            self._fell = True

    def get_state(self) -> Dict[str, float]:
        return {
            "x_velocity": float(self._v),
            "upright": float(self._upright),
            "z_height": 0.06,
            "food_distance": float(self._food),
            "wall_min_distance": 1.0,
            "fell": bool(self._fell),
        }


class ShuffledConnectomePipeline:
    """
    HONESTY CONTROL. Wraps a real pipeline but permutes the connectome
    joint->torque assignment on every reset. Magnitudes (the multiset of torque
    values) and joint degree are preserved; the *structure* (which joint gets
    which signal) is destroyed. If the real connectome carries motor meaning,
    training on this control must do WORSE. If it doesn't, the connectome path
    is contributing no signal yet — and we must report that.
    """

    def __init__(self, base_pipeline: "PipelineProtocol", seed: int):
        self.base = base_pipeline
        self.cfg = base_pipeline.cfg
        self.joint_names = list(base_pipeline.joint_names)
        self._rng = np.random.default_rng(seed)
        self._perm = np.arange(len(self.joint_names))
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self._perm = self._rng.permutation(len(self.joint_names))

    def get_observation(self) -> np.ndarray:
        return self.base.get_observation()

    def get_connectome_torques(self) -> Dict[str, float]:
        base = self.base.get_connectome_torques()
        vals = [base.get(n, 0.0) for n in self.joint_names]
        # reassign the SAME multiset of torque values to permuted joints
        return {self.joint_names[self._perm[i]]: float(vals[i]) for i in range(len(self.joint_names))}

    def apply_torques(self, torques: Dict[str, float]) -> None:
        self.base.apply_torques(torques)

    def step_physics(self) -> None:
        self.base.step_physics()

    def get_state(self) -> Dict[str, float]:
        return self.base.get_state()


# ---------------------------------------------------------------------------
# PPO trainer (numpy, analytic gradients, GAE, clipped surrogate)
# ---------------------------------------------------------------------------
def train_ppo(env: "SimFlyRLEnv", policy: ConnectomeModulationPolicy, config: RLConfig,
              n_iterations: int, log_path: str) -> List[Dict[str, Any]]:
    """
    Train PPO. Logs one JSONL line per iteration and saves a weight snapshot
    every iteration (NEVER LOSE DATA). Returns the history list.
    """
    snap_dir = log_path + ".snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    env.set_squash(policy.squash)  # env.step receives RAW actions

    T = config.rollout_steps
    obs_dim = config.obs_dim
    act_dim = 2 * config.n_joints
    history: List[Dict[str, Any]] = []

    for it in range(n_iterations):
        obs_buf = np.zeros((T, obs_dim))
        raw_buf = np.zeros((T, act_dim))
        rew_buf = np.zeros(T)
        val_buf = np.zeros(T)
        logp_buf = np.zeros(T)
        done_buf = np.zeros(T)

        obs, _ = env.reset()
        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        cur_r, cur_l = 0.0, 0
        gains_acc, biases_acc = [], []

        for t in range(T):
            squashed, logp, value, raw = policy.act(obs)
            obs_buf[t] = obs
            raw_buf[t] = raw
            val_buf[t] = value
            logp_buf[t] = logp
            g = squashed[:config.n_joints]; b = squashed[config.n_joints:]
            gains_acc.append(g.mean()); biases_acc.append(b.mean())

            obs, rew, terminated, truncated, _ = env.step(raw)
            rew_buf[t] = rew
            cur_r += rew; cur_l += 1
            done = terminated or truncated
            done_buf[t] = 1.0 if done else 0.0
            if done:
                ep_rewards.append(cur_r); ep_lengths.append(cur_l)
                cur_r, cur_l = 0.0, 0
                obs, _ = env.reset()

        last_val = float(policy.critic.forward(obs)[0, 0])

        # ---- GAE ----
        adv = np.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            next_nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + config.gamma * next_val * next_nonterm - val_buf[t]
            last_gae = delta + config.gamma * config.lam * next_nonterm * last_gae
            adv[t] = last_gae
        returns = adv + val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO epochs over minibatches ----
        idx = np.arange(T)
        for _ in range(config.epochs):
            np.random.shuffle(idx)
            for s in range(0, T, config.minibatch):
                mb = idx[s:s + config.minibatch]
                o, r_act = obs_buf[mb], raw_buf[mb]
                A, R, old_lp = adv[mb], returns[mb], logp_buf[mb]

                mean = policy.actor.forward(o)            # (B, act_dim)
                std = np.exp(policy.log_std)
                var = std ** 2
                new_lp = policy._gauss_logp(r_act, mean, std)
                ratio = np.exp(new_lp - old_lp)

                # clipped surrogate; gradient flows only on the unclipped branch
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - config.clip, 1 + config.clip) * A
                use_unclipped = (unclipped <= clipped).astype(float)   # min() picks this
                # d(L_pi)/d(mean): L_pi = -mean_b( min(...) ); only unclipped depends on mean
                # d ratio/d mean_k = ratio * (r_act_k - mean_k)/var_k
                B = o.shape[0]
                coeff = -(use_unclipped * ratio * A) / B            # (B,)
                d_logp_d_mean = (r_act - mean) / var                # (B, act_dim)
                d_mean = coeff[:, None] * d_logp_d_mean             # (B, act_dim)
                actor_grads = policy.actor.backward(d_mean)

                # log_std gradient (entropy + likelihood); approx via likelihood term
                # dlogp/dlogstd_k = ((x-mean)^2/var_k - 1)
                d_logp_d_logstd = ((r_act - mean) ** 2 / var - 1.0)   # (B, act_dim)
                g_logstd = np.sum(coeff[:, None] * d_logp_d_logstd, axis=0)
                # entropy bonus pushes log_std up
                g_logstd -= config.ent_coef

                # ---- critic ----
                v_pred = policy.critic.forward(o)[:, 0]
                d_v = (config.vf_coef * 2.0 * (v_pred - R) / B)[:, None]   # (B,1)
                critic_grads = policy.critic.backward(d_v)

                policy.actor_opt.step(actor_grads)
                policy.critic_opt.step(critic_grads)
                policy._logstd_adam_step(g_logstd, config.lr)

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else float(rew_buf.sum())
        rec = {
            "iteration": it,
            "mean_episode_reward": mean_reward,
            "mean_step_reward": float(rew_buf.mean()),
            "n_episodes": len(ep_rewards),
            "mean_episode_len": float(np.mean(ep_lengths)) if ep_lengths else float(T),
            "mean_gain": float(np.mean(gains_acc)),
            "mean_bias": float(np.mean(biases_acc)),
            "mean_log_std": float(policy.log_std.mean()),
        }
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        policy.save(os.path.join(snap_dir, f"iter_{it:04d}.npz"))
        print(f"[iter {it:3d}] ep_reward={mean_reward:8.3f} "
              f"step_reward={rec['mean_step_reward']:6.3f} "
              f"gain={rec['mean_gain']:.3f} bias={rec['mean_bias']:+.3f} "
              f"logstd={rec['mean_log_std']:+.2f} eps={len(ep_rewards)}")
    return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke() -> int:
    cfg = RLConfig(n_joints=8, rollout_steps=512, minibatch=128, epochs=6,
                   max_ep_steps=120, seed=0)

    # ---- integrity assertion: identity calibration is a no-op ----
    names = [f"joint_{i}" for i in range(cfg.n_joints)]
    base = {n: float(v) for n, v in zip(names, np.linspace(-0.9, 0.9, cfg.n_joints))}
    ones = np.ones(cfg.n_joints); zeros = np.zeros(cfg.n_joints)
    passthrough = apply_modulation(base, names, ones, zeros)
    assert all(abs(passthrough[n] - base[n]) < 1e-9 for n in names), \
        "INTEGRITY FAIL: gain=1,bias=0 must pass connectome torque unchanged"
    print("[integrity] gain=1,bias=0 -> connectome torque passes through unchanged. OK")

    print("\n=== REAL (mock) connectome pipeline ===")
    real_env = SimFlyRLEnv(MockPipeline(cfg))
    real_policy = ConnectomeModulationPolicy(cfg)
    real_hist = train_ppo(real_env, real_policy, cfg, n_iterations=3,
                          log_path="/tmp/rl_smoke_real.jsonl")

    print("\n=== SHUFFLED control (connectome structure destroyed) ===")
    shuf_env = SimFlyRLEnv(ShuffledConnectomePipeline(MockPipeline(cfg), seed=7))
    shuf_policy = ConnectomeModulationPolicy(cfg)
    shuf_hist = train_ppo(shuf_env, shuf_policy, cfg, n_iterations=3,
                          log_path="/tmp/rl_smoke_shuffled.jsonl")

    print("\n=== COMPARISON (real vs shuffled mean_episode_reward) ===")
    for r, s in zip(real_hist, shuf_hist):
        print(f"  iter {r['iteration']}: real={r['mean_episode_reward']:8.3f}  "
              f"shuffled={s['mean_episode_reward']:8.3f}")
    print("\nNOTE: on the MockPipeline both share the same toy body, so the gap is\n"
          "illustrative only. The real scientific test runs this comparison with\n"
          "the actual FlyWire pipeline (server integration).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SimFly RL calibration bridge")
    ap.add_argument("--smoke", action="store_true", help="run offline smoke test")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke())
    print("rl_bridge.py loaded. Use --smoke to run the offline self-test.")
    print("Integrate with the live server via SimFlyRLEnv(pipeline) where the")
    print("pipeline exposes the connectome decoder output (see module docstring).")