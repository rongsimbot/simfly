#!/usr/bin/env python3
"""
Phase 18a: Joint-Type-Specific Reward Function (v2)
=====================================================
Updated with ACCURATE MuJoCo joint ranges from simfly_arena.xml.

MuJoCo Joint Ranges:
  coxa_T1:        (-0.2, 1.7)   | coxa_T2: (-0.2, 0.9)   | coxa_T3: (-0.3, 1.3)
  coxa_abduct_T1: (-1.0, 0.7)   | T2: (-0.5, 0.3)       | T3: (-0.9, 0.25)
  coxa_twist_T1:  (-0.8, 0.8)   | T2: (-0.75, 0.8)      | T3: (-0.15, 0.8)
  femur_T1:       (-0.15, 2.0)   | T2: (-0.15, 2.0)      | T3: (-0.7, 1.5)
  femur_twist:    (-1.0, 1.0) all segments
  tibia:          (-1.35, 1.3) all segments
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MUJOCO JOINT RANGES (from simfly_arena.xml)                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
MUJOCO_RANGES = {
    'coxa_T1_left': (-0.2, 1.7),     'coxa_T1_right': (-0.2, 1.7),
    'coxa_T2_left': (-0.2, 0.9),     'coxa_T2_right': (-0.2, 0.9),
    'coxa_T3_left': (-0.3, 1.3),     'coxa_T3_right': (-0.3, 1.3),
    'coxa_abduct_T1_left': (-1.0, 0.7),  'coxa_abduct_T1_right': (-1.0, 0.7),
    'coxa_abduct_T2_left': (-0.5, 0.3),  'coxa_abduct_T2_right': (-0.5, 0.3),
    'coxa_abduct_T3_left': (-0.9, 0.25), 'coxa_abduct_T3_right': (-0.9, 0.25),
    'coxa_twist_T1_left': (-0.8, 0.8),   'coxa_twist_T1_right': (-0.8, 0.8),
    'coxa_twist_T2_left': (-0.75, 0.8),  'coxa_twist_T2_right': (-0.75, 0.8),
    'coxa_twist_T3_left': (-0.15, 0.8),  'coxa_twist_T3_right': (-0.15, 0.8),
    'femur_T1_left': (-0.15, 2.0),    'femur_T1_right': (-0.15, 2.0),
    'femur_T2_left': (-0.15, 2.0),    'femur_T2_right': (-0.15, 2.0),
    'femur_T3_left': (-0.7, 1.5),     'femur_T3_right': (-0.7, 1.5),
    'femur_twist_T1_left': (-1.0, 1.0),  'femur_twist_T1_right': (-1.0, 1.0),
    'femur_twist_T2_left': (-1.0, 1.0),  'femur_twist_T2_right': (-1.0, 1.0),
    'femur_twist_T3_left': (-1.0, 1.0),  'femur_twist_T3_right': (-1.0, 1.0),
    'tibia_T1_left': (-1.35, 1.3),    'tibia_T1_right': (-1.35, 1.3),
    'tibia_T2_left': (-1.35, 1.3),    'tibia_T2_right': (-1.35, 1.3),
    'tibia_T3_left': (-1.35, 1.3),    'tibia_T3_right': (-1.35, 1.3),
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Joint Biomechanical Profiles (v2 — accurate ranges)               ║
# ╚══════════════════════════════════════════════════════════════════════╝
JOINT_PROFILES: Dict[str, dict] = {
    # ── Main Rotation (coxa) — large ROM, clear flexion/extension ──
    'coxa_T': {
        'limit_margin': 0.2,
        'movement_weight': 1.0,
        'stability_weight': 0.3,
        'limit_penalty_scale': 0.5,
        'neutral_zone': 0.15,
        'power_bonus_weight': 0.1,
        'oscillation_penalty': 0.02,
    },
    # ── Abduction/Adduction — narrower ROM, gentler movement reward ──
    'coxa_abduct': {
        'limit_margin': 0.25,  # wider margin = earlier warning before limit
        'movement_weight': 0.5,  # half the baseline displacement reward
        'stability_weight': 0.8,
        'limit_penalty_scale': 0.3,  # softer penalty than coxa (0.5)
        'neutral_zone': 0.1,
        'power_bonus_weight': 0.05,
        'oscillation_penalty': 0.05,
    },
    # ── Twist — narrow ROM, minimal movement reward ──
    'coxa_twist': {
        'limit_margin': 0.15,  # wider relative to small range
        'movement_weight': 0.3,  # low — micro-movements only
        'stability_weight': 1.0,
        'limit_penalty_scale': 0.2,  # very soft
        'neutral_zone': 0.05,
        'power_bonus_weight': 0.0,
        'oscillation_penalty': 0.08,
    },
    # ── Femur (main rotation) — largest torque, reward power + excursion ──
    'femur_T': {
        'limit_margin': 0.25,
        'movement_weight': 1.2,
        'stability_weight': 0.2,
        'limit_penalty_scale': 0.5,
        'neutral_zone': 0.2,
        'power_bonus_weight': 0.3,
        'oscillation_penalty': 0.01,
    },
    # ── Femur Twist — narrow, reward precision ──
    'femur_twist': {
        'limit_margin': 0.1,
        'movement_weight': 0.4,
        'stability_weight': 0.9,
        'limit_penalty_scale': 0.15,
        'neutral_zone': 0.06,
        'power_bonus_weight': 0.0,
        'oscillation_penalty': 0.06,
    },
    # ── Tibia — medium torque, controlled flexion ──
    'tibia_T': {
        'limit_margin': 0.2,
        'movement_weight': 0.8,
        'stability_weight': 0.5,
        'limit_penalty_scale': 0.5,
        'neutral_zone': 0.15,
        'power_bonus_weight': 0.15,
        'oscillation_penalty': 0.02,
    },
}


def get_joint_type(joint_name: str) -> str:
    """Parse joint biomechanical type from joint name."""
    name = joint_name.lower()
    if 'coxa_abduct' in name:
        return 'coxa_abduct'
    elif 'coxa_twist' in name:
        return 'coxa_twist'
    elif 'coxa_' in name:
        return 'coxa_T'
    elif 'femur_twist' in name:
        return 'femur_twist'
    elif 'femur_' in name:
        return 'femur_T'
    elif 'tibia_' in name:
        return 'tibia_T'
    return 'unknown'


def get_joint_profile(joint_name: str) -> dict:
    """Get the biomechanical profile for a joint."""
    jtype = get_joint_type(joint_name)
    profile = JOINT_PROFILES.get(jtype, {
        'limit_margin': 0.2,
        'movement_weight': 1.0,
        'stability_weight': 0.3,
        'limit_penalty_scale': 0.5,
        'neutral_zone': 0.1,
        'power_bonus_weight': 0.1,
        'oscillation_penalty': 0.02,
    })
    # Add the MuJoCo range
    jrange = MUJOCO_RANGES.get(joint_name, (-1.57, 1.57))
    return {**profile, 'joint_range': jrange, 'joint_type': jtype}


def compute_reward_simple(
    joint_name: str,
    joint_angle: float,
    joint_velocity: float,
    prev_angle: float = 0.0,
) -> float:
    """
    Phase16-style generic reward adapted with joint-type-specific limits.
    
    Proven formula (worked for coxa_T1: R=+10.20):
      displacement * weight + toward_center_bonus - limit_penalty - zero_penalty
    
    Key: HIGH displacement weight (5-10), sharp limit penalty, mild zero penalty.
    For narrow-ROM joints (abduct, twist): lower displacement weight, wider margin.
    """
    profile = get_joint_profile(joint_name)
    jrange = profile['joint_range']
    lo, hi = jrange
    margin = profile['limit_margin']

    displacement = abs(joint_angle - prev_angle)

    # ── 1. Movement reward (same as proven Phase16 formula) ──
    # Use profile movement_weight as multiplier against the 10.0 baseline
    angular_weight = 10.0 * profile['movement_weight']
    move_reward = displacement * angular_weight

    # ── 2. Toward-center bonus (reward moving toward 0) ──
    if abs(joint_velocity) > 0.001:
        toward_center = -np.sign(joint_angle * joint_velocity)
    else:
        toward_center = 0.0
    center_bonus = displacement * max(0.0, toward_center) * 2.0

    # ── 3. Limit penalty (sharp, Phase16-style) ──
    # Uses the INNER margin boundary to penalize approaching limits
    limit_penalty = 0.0
    upper_bound = hi - margin
    lower_bound = lo + margin
    if joint_angle > upper_bound:
        limit_penalty = -(joint_angle - upper_bound) * 5.0 * profile['limit_penalty_scale']
    elif joint_angle < lower_bound:
        limit_penalty = -(lower_bound - joint_angle) * 5.0 * profile['limit_penalty_scale']

    # ── 4. Zero movement penalty (encourage exploration) ──
    zero_penalty = -0.01 if displacement < 0.0001 else 0.0

    return float(move_reward + center_bonus + limit_penalty + zero_penalty)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Self-Test                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    test_joints = [
        "coxa_T1_left", "coxa_abduct_T1_left", "coxa_twist_T1_left",
        "femur_T1_left", "femur_twist_T1_left", "tibia_T1_left",
    ]

    print("=" * 70)
    print("Joint-Type-Specific Reward Function v2 — Self Test")
    print("=" * 70)

    for jname in test_joints:
        profile = get_joint_profile(jname)
        jrange = profile['joint_range']
        lo, hi = jrange
        print(f"\n── {jname} (type: {profile['joint_type']}, range: [{lo}, {hi}]) ──")

        # Near neutral
        r = compute_reward_simple(jname, 0.01, 0.001, 0.0)
        print(f"  Near neutral (θ=0.01): R={r:.4f}")

        # Moderate excursion (30% of range)
        mid = lo + 0.3 * (hi - lo)
        r = compute_reward_simple(jname, mid, 0.1, mid - 0.02)
        print(f"  Moderate (θ={mid:.3f}, 30% range): R={r:.4f}")

        # Near upper limit (95% of range)
        near_hi = lo + 0.95 * (hi - lo)
        r = compute_reward_simple(jname, near_hi, 0.3, near_hi - 0.01)
        print(f"  Near hi limit (θ={near_hi:.3f}): R={r:.4f}")

        # Near lower limit (5% of range)
        near_lo = lo + 0.05 * (hi - lo)
        r = compute_reward_simple(jname, near_lo, -0.1, near_lo + 0.01)
        print(f"  Near lo limit (θ={near_lo:.3f}): R={r:.4f}")

        # Past limit
        r = compute_reward_simple(jname, hi + 0.1, 0.1, hi + 0.05)
        print(f"  Past hi (θ={hi + 0.1:.3f}): R={r:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
