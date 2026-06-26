#!/usr/bin/env python3
"""Quick smoke test for Brain2Env."""
from training_env import Brain2Env
import numpy as np

env = Brain2Env(stabilize_seconds=2.0, sample_steps=5, sample_interval=0.2)

print("Reset...")
obs, _ = env.reset()
print(f"  State dim: {len(obs)}")
print(f"  Food dist: {obs[36]:.3f}")

# Test with uniform multiplier = 1.0 (sigmoid(-0.693) = 0.333, so 0.5+1.5*0.333=1.0)
action = np.full(36, -0.693, dtype=np.float32)
print("\nStep with uniform 1.0x scales...")
obs, reward, done, trunc, info = env.step(action)
print(f"  Reward: {reward:.4f}")
print(f"  Torque CV: {info.get('torque_cv', 0):.4f}")
print(f"  Food dist: {info.get('food_distance', 0):.4f}")
print(f"  Saturation: {info.get('sat_ratio', 0):.4f}")
print(f"  Multiplier mean: {info.get('multiplier_mean', 0):.4f}")
print(f"  Multiplier std: {info.get('multiplier_std', 0):.4f}")
print("\nEnvironment smoke test PASSED")
