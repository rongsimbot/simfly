"""
rl_bridge_gpu.py — GPU-accelerated RL calibration layer for SimFly.

Replaces numpy-only PPO with PyTorch GPU training.
Connectome pipeline + MuJoCo on CPU; policy/value networks on GPU.

INTEGRITY: gain=1,bias=0 ≡ connectome passthrough (tested in smoke).
"""
from __future__ import annotations

import json, os, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Device ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    _mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[RL-GPU] {torch.cuda.get_device_name(0)} | {_mem:.1f} GB | PyTorch {torch.__version__}",
          flush=True)
else:
    print(f"[RL-GPU] CPU-only mode", flush=True)


# ── Config ──────────────────────────────────────────────────────────────
@dataclass
class RLConfig:
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
    action_scale_min: float = 0.1
    action_scale_max: float = 3.0
    modulation_range: float = 0.5

    def __post_init__(self) -> None:
        self.obs_dim = 2 * self.n_joints + 3


# ── GPU MLP ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── GPU Policy ──────────────────────────────────────────────────────────
class ConnectomeModulationPolicy(nn.Module):
    """Actor-critic on GPU. Outputs raw means; log_std learned per-dim."""

    def __init__(self, config: RLConfig):
        super().__init__()
        self.cfg = config
        self.act_dim = 2 * config.n_joints
        self.actor = MLP(config.obs_dim, config.hidden, self.act_dim)
        self.critic = MLP(config.obs_dim, config.hidden, 1)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5))
        self.to(DEVICE)

    def squash(self, raw):
        was_numpy = isinstance(raw, np.ndarray)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=torch.float32, device=DEVICE)
        n = self.cfg.n_joints
        sig = torch.sigmoid(raw[..., :n])
        gains = self.cfg.action_scale_min + (
            self.cfg.action_scale_max - self.cfg.action_scale_min) * sig
        biases = self.cfg.modulation_range * torch.tanh(raw[..., n:])
        if was_numpy:
            return gains.detach().cpu().numpy(), biases.detach().cpu().numpy()
        return gains, biases

    # ── CPU→GPU→CPU (rollout) ───────────────────────────────────────
    def act(self, obs: np.ndarray):
        """Single observation: CPU→GPU→CPU."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(obs_t)[0]
            std = self.log_std.exp()
            dist = Normal(mean, std)
            raw = dist.sample()
            logp = dist.log_prob(raw).sum()
            value = self.critic(obs_t)[0, 0]
            gains, biases = self.squash(raw)
        squashed = torch.cat([gains, biases])
        return (squashed.cpu().numpy(), float(logp.cpu()),
                float(value.cpu()), raw.cpu().numpy())

    # ── Pure GPU batch (PPO update) ─────────────────────────────────
    def evaluate_batch(self, obs_gpu: torch.Tensor, raw_gpu: torch.Tensor):
        """Pure GPU: returns logp, values, entropy — all torch tensors."""
        mean = self.actor(obs_gpu)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        logp = dist.log_prob(raw_gpu).sum(-1)
        values = self.critic(obs_gpu).squeeze(-1)
        entropy = dist.entropy().sum(-1).mean()
        return logp, values, entropy

    # ── Snapshot I/O ────────────────────────────────────────────────
    def get_params_dict(self):
        return {n: p.data.cpu().numpy().copy()
                for n, p in self.named_parameters()}

    def set_params_dict(self, d):
        for n, p in self.named_parameters():
            if n in d:
                p.data.copy_(torch.as_tensor(d[n], device=DEVICE))

    def save(self, path):
        np.savez(path, **self.get_params_dict())

    def load(self, path):
        data = np.load(path)
        self.set_params_dict({k: data[k] for k in data.files})


# ── Modulation (CPU) ────────────────────────────────────────────────────
def apply_modulation(torques: Dict[str, float], joint_names: List[str],
                     gains: np.ndarray, biases: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, name in enumerate(joint_names):
        base = float(torques.get(name, 0.0))
        out[name] = float(np.clip(base * float(gains[i]) + float(biases[i]), -1.0, 1.0))
    return out


# ── Gym Environment (re-exports rl_bridge SimFlyRLEnv) ──────────────────
# We can't easily redefine the env because the pipeline interface is complex.
# Import the original env class and use it as-is.
_HAVE_GYM = True
try:
    import gymnasium as gym
except Exception:
    gym = None
    _HAVE_GYM = False

# Re-use the original SimFlyRLEnv — it just wraps a pipeline and doesn't
# contain any numpy MLP code. We inject our GPU policy's squash.
from rl_bridge import SimFlyRLEnv


# ── GPU PPO Trainer ─────────────────────────────────────────────────────
def train_ppo(env, policy: ConnectomeModulationPolicy, config: RLConfig,
              n_iterations: int, log_path: str) -> List[Dict[str, Any]]:
    """
    GPU PPO: rollouts on CPU (MuJoCo/connectome), updates on GPU.
    Logs JSONL per iteration + weight snapshots. Never loses data.
    """
    snap_dir = log_path + ".snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    env.set_squash(policy.squash)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)
    T = config.rollout_steps
    obs_dim, act_dim = config.obs_dim, 2 * config.n_joints
    history = []

    for it in range(n_iterations):
        t_iter = time.time()

        # ── 1. Rollout (CPU-bound: MuJoCo + connectome) ─────────────
        obs_buf = np.zeros((T, obs_dim), dtype=np.float32)
        raw_buf = np.zeros((T, act_dim), dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)
        val_buf = np.zeros(T, dtype=np.float32)
        logp_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)

        obs, _ = env.reset()
        ep_rewards, ep_lengths = [], []
        cur_r, cur_l = 0.0, 0
        gains_acc, biases_acc = [], []

        for t in range(T):
            squashed, logp, value, raw = policy.act(obs)
            obs_buf[t], raw_buf[t] = obs, raw
            val_buf[t], logp_buf[t] = value, logp
            g = squashed[:config.n_joints]; b = squashed[config.n_joints:]
            gains_acc.append(float(g.mean())); biases_acc.append(float(b.mean()))

            obs, rew, terminated, truncated, _ = env.step(raw)
            rew_buf[t] = rew
            cur_r += rew; cur_l += 1
            done = terminated or truncated
            done_buf[t] = 1.0 if done else 0.0
            if done:
                ep_rewards.append(cur_r); ep_lengths.append(cur_l)
                cur_r, cur_l = 0.0, 0
                obs, _ = env.reset()

        t_rollout = time.time() - t_iter

        # ── 2. GAE (CPU numpy, fast) ────────────────────────────────
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            last_val = float(policy.critic(obs_t)[0, 0].cpu())

        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t_rev in reversed(range(T)):
            nv = last_val if t_rev == T - 1 else val_buf[t_rev + 1]
            nonterm = 1.0 - done_buf[t_rev]
            delta = rew_buf[t_rev] + config.gamma * nv * nonterm - val_buf[t_rev]
            last_gae = delta + config.gamma * config.lam * nonterm * last_gae
            adv[t_rev] = last_gae
        returns = adv + val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── 3. Upload to GPU ────────────────────────────────────────
        obs_g = torch.as_tensor(obs_buf, device=DEVICE)
        raw_g = torch.as_tensor(raw_buf, device=DEVICE)
        adv_g = torch.as_tensor(adv, device=DEVICE)
        ret_g = torch.as_tensor(returns, device=DEVICE)
        old_lp_g = torch.as_tensor(logp_buf, device=DEVICE)

        # ── 4. PPO Update (GPU) ─────────────────────────────────────
        t_update = time.time()
        for _ in range(config.epochs):
            perm = torch.randperm(T, device=DEVICE)
            for s in range(0, T, config.minibatch):
                mb = perm[s:s + config.minibatch]
                new_lp, values_g, entropy_g = policy.evaluate_batch(
                    obs_g[mb], raw_g[mb])

                ratio = (new_lp - old_lp_g[mb]).exp()
                surr1 = ratio * adv_g[mb]
                surr2 = torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv_g[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values_g, ret_g[mb])
                loss = (actor_loss + config.vf_coef * critic_loss
                        - config.ent_coef * entropy_g)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        t_update = time.time() - t_update
        t_total = time.time() - t_iter

        # ── 5. Log ──────────────────────────────────────────────────
        with torch.no_grad():
            ls = float(policy.log_std.mean().cpu())

        mr = float(np.mean(ep_rewards)) if ep_rewards else float(rew_buf.sum())
        rec = {
            "iteration": it,
            "mean_episode_reward": mr,
            "mean_step_reward": float(rew_buf.mean()),
            "n_episodes": len(ep_rewards),
            "mean_episode_len": float(np.mean(ep_lengths)) if ep_lengths else float(T),
            "mean_gain": float(np.mean(gains_acc)),
            "mean_bias": float(np.mean(biases_acc)),
            "mean_log_std": ls,
            "rollout_sec": round(t_rollout, 1),
            "update_sec": round(t_update, 1),
            "total_sec": round(t_total, 1),
        }
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        policy.save(os.path.join(snap_dir, f"iter_{it:04d}.npz"))

        # Estimate ETA
        eta = t_total * (n_iterations - it - 1)
        print(f"[iter {it:3d}] reward={mr:8.3f} gain={rec['mean_gain']:.3f} "
              f"bias={rec['mean_bias']:+.3f} logstd={ls:+.2f} "
              f"rollout={t_rollout:.1f}s update={t_update:.3f}s "
              f"ETA={eta/60:.0f}min eps={len(ep_rewards)}", flush=True)

    return history


# ── Smoke test ──────────────────────────────────────────────────────────
def _smoke():
    cfg = RLConfig(n_joints=8, rollout_steps=512, minibatch=128, epochs=6,
                   max_ep_steps=120, seed=0)

    # Integrity
    names = [f"joint_{i}" for i in range(cfg.n_joints)]
    base = {n: float(v) for n, v in zip(names, np.linspace(-0.9, 0.9, cfg.n_joints))}
    passthrough = apply_modulation(base, names, np.ones(cfg.n_joints), np.zeros(cfg.n_joints))
    assert all(abs(passthrough[n] - base[n]) < 1e-9 for n in names), "INTEGRITY FAIL"
    print("[✓] Integrity: gain=1,bias=0 ≡ connectome passthrough", flush=True)

    from rl_bridge import MockPipeline
    env = SimFlyRLEnv(MockPipeline(cfg))
    policy = ConnectomeModulationPolicy(cfg)
    print(f"[✓] Policy on {DEVICE}, params: {sum(p.numel() for p in policy.parameters())}",
          flush=True)

    hist = train_ppo(env, policy, cfg, n_iterations=3,
                     log_path="/tmp/rl_gpu_smoke.jsonl")
    print(f"\n[✓] GPU PPO smoke: {len(hist)} iters", flush=True)
    for h in hist:
        print(f"  iter {h['iteration']}: reward={h['mean_episode_reward']:.3f} "
              f"({h['rollout_sec']}s rollout + {h['update_sec']:.3f}s GPU update)", flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke())
    print("rl_bridge_gpu.py ready. Import or use --smoke.", flush=True)
