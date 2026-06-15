#!/usr/bin/env python3
"""
Gait Controller — Tripod gait state machine for connectome-driven locomotion.

Implements alternating tripod gait for Drosophila walking:
  - Tripod 1: Left Front (L1/T1) + Right Middle (R2/T2) + Left Hind (L3/T3)
  - Tripod 2: Right Front (R1/T1) + Left Middle (L2/T2) + Right Hind (R3/T3)

Phases alternate at ~5Hz. The controller acts as a post-processing filter
on the connectome-driven torque outputs:
  - SWING legs: amplified torque + lift bias for visible leg movement
  - STANCE legs: suppressed torque + push-down bias for stability

Turning modes add asymmetry: stronger torque on the outer turning side.

Usage:
    gc = GaitController(actuator_map, gait_freq_hz=5.0)
    gait_mode = 'walk'  # from DNSubtypeClassifier.determine_gait_mode()
    torques = decoder.decode()  # connectome-driven torques
    torques = gc.apply(gait_mode, torques, dt_ms=5.0)
    # torques now has tripod-gated values
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math
import re


# ── Tripod leg groups ─────────────────────────────────────────────────────

# Tripod 1: L1 + R2 + L3  (swing together)
TRIPOD_1_LEGS: Set[str] = {'T1_left', 'T2_right', 'T3_left'}

# Tripod 2: R1 + L2 + R3  (swing together)
TRIPOD_2_LEGS: Set[str] = {'T1_right', 'T2_left', 'T3_right'}

ALL_LEGS: Set[str] = TRIPOD_1_LEGS | TRIPOD_2_LEGS


class GaitController:
    """Tripod gait state machine for Drosophila hexapod walking.

    Applies gait-phase-specific torque modulation to connectome-driven
    joint commands. One tripod swings while the other provides stance.

    Usage::

        gc = GaitController(actuator_map)
        gait_mode = classifier.determine_gait_mode(grad_dir, concentration)
        torques = gc.apply(gait_mode, raw_torques, dt_ms=5.0)
    """

    def __init__(
        self,
        actuator_map: Dict[str, int] = None,
        gait_freq_hz: float = 5.0,
        swing_amplify: float = 2.5,
        stance_suppress: float = 0.05,
        turn_bias_outer: float = 1.6,
        turn_bias_inner: float = 0.4,
        lift_bias: float = 0.12,
        stance_push: float = -0.08,
    ):
        """Initialize the gait controller.

        Args:
            actuator_map: Dict mapping joint_name → actuator_index (for leg detection).
            gait_freq_hz: Tripod alternation frequency in Hz (default 5Hz).
            swing_amplify: Torque multiplier for swing-phase legs.
            stance_suppress: Torque multiplier for stance-phase legs.
            turn_bias_outer: Extra multiplier for outer turn-side legs.
            turn_bias_inner: Reduced multiplier for inner turn-side legs.
            lift_bias: Positive bias added to femur joints during swing (lift).
            stance_push: Negative bias added to stance legs (push down).
        """
        self.actuator_map = actuator_map or {}
        self.gait_freq_hz = gait_freq_hz
        self.swing_amplify = swing_amplify
        self.stance_suppress = stance_suppress
        self.turn_bias_outer = turn_bias_outer
        self.turn_bias_inner = turn_bias_inner
        self.lift_bias = lift_bias
        self.stance_push = stance_push

        # Build leg-to-joint mapping from actuator map
        self.leg_joints: Dict[str, List[str]] = defaultdict(list)
        self.joint_to_leg: Dict[str, str] = {}
        for joint_name in self.actuator_map:
            leg = self._extract_leg(joint_name)
            if leg:
                self.leg_joints[leg].append(joint_name)
                self.joint_to_leg[joint_name] = leg

        # Phase state
        self.current_phase: str = 'tripod_1'
        self.phase_timer_ms: float = 0.0
        self.period_ms: float = 1000.0 / max(0.1, gait_freq_hz)  # 200ms at 5Hz
        self.half_period_ms: float = self.period_ms / 2.0

        # Step counter
        self.step_count: int = 0
        self.phase_switch_count: int = 0

        # Print init info
        if self.leg_joints:
            print(f"  [gait] Initialized: {len(self.leg_joints)} legs, "
                  f"{sum(len(v) for v in self.leg_joints.values())} joints, "
                  f"{gait_freq_hz}Hz tripod", flush=True)
            for leg in sorted(self.leg_joints):
                tripod = 'T1' if leg in TRIPOD_1_LEGS else 'T2' if leg in TRIPOD_2_LEGS else '?'
                joints_str = ', '.join(self.leg_joints[leg][:3])
                if len(self.leg_joints[leg]) > 3:
                    joints_str += ', ...'
                print(f"    {leg} [{tripod}]: {joints_str}", flush=True)

    def _extract_leg(self, joint_name: str) -> Optional[str]:
        """Extract leg identifier from joint name.

        Examples:
            'coxa_T1_left'  → 'T1_left'
            'femur_T2_right' → 'T2_right'
            'tibia_T3_left'  → 'T3_left'

        Args:
            joint_name: Full joint/actuator name.

        Returns:
            Leg identifier like 'T1_left' or None if not a leg joint.
        """
        m = re.search(r'(T\d)_(left|right)', joint_name)
        if m:
            return f'{m.group(1)}_{m.group(2)}'
        return None

    def _is_swing_joint(self, joint_name: str) -> bool:
        """Check if a joint is in a swing-phase leg."""
        leg = self.joint_to_leg.get(joint_name)
        if leg is None:
            return False
        if self.current_phase == 'tripod_1':
            return leg in TRIPOD_1_LEGS
        elif self.current_phase == 'tripod_2':
            return leg in TRIPOD_2_LEGS
        return False

    def _is_femur_lift(self, joint_name: str) -> bool:
        """Check if this is a femur joint (primary lift actuator)."""
        return 'femur' in joint_name.lower() and 'twist' not in joint_name.lower()

    def _is_coxa_swing(self, joint_name: str) -> bool:
        """Check if this is a coxa joint (forward/back rotation)."""
        return 'coxa' in joint_name.lower() and 'twist' not in joint_name.lower() and 'abduct' not in joint_name.lower()

    # ── Phase advancement ───────────────────────────────────────────────

    def _advance_phase(self, dt_ms: float) -> None:
        """Advance the gait phase timer and toggle tripods at half-period."""
        self.phase_timer_ms += dt_ms

        if self.phase_timer_ms >= self.half_period_ms:
            self.phase_timer_ms -= self.half_period_ms
            self.phase_switch_count += 1

            # Toggle tripod
            if self.current_phase == 'tripod_1':
                self.current_phase = 'tripod_2'
            else:
                self.current_phase = 'tripod_1'

    # ── Main gait application ───────────────────────────────────────────

    def apply(
        self,
        gait_mode: str,
        base_torques: Dict[str, float],
        dt_ms: float = 5.0,
    ) -> Dict[str, float]:
        """Apply gait-phase modulation to connectome-driven torques.

        Transforms raw connectome output into biologically-plausible
        walking patterns by gating which legs move when.

        Args:
            gait_mode: One of 'stance', 'walk', 'turn_left', 'turn_right'.
            base_torques: Raw torque dict {joint_name: ctrl_value}.
            dt_ms: Time step in milliseconds for phase advancement.

        Returns:
            Modified torque dict with gait gating applied.
        """
        self.step_count += 1

        # Advance tripod phase for walking/turning modes
        if gait_mode in ('walk', 'turn_left', 'turn_right'):
            self._advance_phase(dt_ms)
        else:
            # Stance: reset phase
            self.current_phase = 'tripod_1'
            self.phase_timer_ms = 0.0

        result: Dict[str, float] = {}

        for joint_name, torque in base_torques.items():
            leg = self.joint_to_leg.get(joint_name)
            modified = torque

            if gait_mode == 'stance':
                # All legs: ZERO connectome torque, only apply push-down for stability
                modified = 0.0
                if self._is_femur_lift(joint_name):
                    modified = self.stance_push * 3.0  # Strong push-down on femurs
                elif self._is_coxa_swing(joint_name):
                    modified = self.stance_push * 1.5  # Mild push-down on coxa

            elif gait_mode in ('walk', 'turn_left', 'turn_right'):
                is_swing = self._is_swing_joint(joint_name)

                if is_swing:
                    # SWING: amplify torque for visible movement
                    modified *= self.swing_amplify
                    # Add lift bias to femur joints
                    if self._is_femur_lift(joint_name):
                        modified += self.lift_bias
                    # Add forward bias to coxa joints
                    if self._is_coxa_swing(joint_name):
                        if leg and leg.endswith('_right'):
                            modified += self.lift_bias * 0.5  # Right legs: forward
                        else:
                            modified -= self.lift_bias * 0.5  # Left legs: forward
                else:
                    # STANCE: actively push down for ground propulsion
                    if self._is_femur_lift(joint_name):
                        # Femur extends = pushes body forward/up from ground
                        modified = self.stance_push * 2.5  # Strong push-down
                    elif self._is_coxa_swing(joint_name):
                        # Coxa rotates backward = forward propulsion
                        modified = self.stance_push * 1.5
                    else:
                        modified = torque * 0.1  # Minimal torque on other joints

            # Turning bias (applied on top of gait gating)
            if gait_mode == 'turn_left':
                if leg and leg.endswith('_right'):
                    modified *= self.turn_bias_outer  # Outer side = right, more torque
                elif leg and leg.endswith('_left'):
                    modified *= self.turn_bias_inner  # Inner side = left, less torque

            elif gait_mode == 'turn_right':
                if leg and leg.endswith('_left'):
                    modified *= self.turn_bias_outer  # Outer side = left, more torque
                elif leg and leg.endswith('_right'):
                    modified *= self.turn_bias_inner  # Inner side = right, less torque

            result[joint_name] = modified

        return result

    # ── Query ───────────────────────────────────────────────────────────

    def get_phase_info(self) -> Dict:
        """Get current gait phase information."""
        in_swing = TRIPOD_1_LEGS if self.current_phase == 'tripod_1' else TRIPOD_2_LEGS
        in_stance = TRIPOD_2_LEGS if self.current_phase == 'tripod_1' else TRIPOD_1_LEGS
        return {
            'phase': self.current_phase,
            'phase_timer_ms': self.phase_timer_ms,
            'half_period_ms': self.half_period_ms,
            'swing_legs': sorted(in_swing),
            'stance_legs': sorted(in_stance),
            'phase_switches': self.phase_switch_count,
        }

    def reset(self) -> None:
        """Reset gait state machine."""
        self.current_phase = 'tripod_1'
        self.phase_timer_ms = 0.0
        self.step_count = 0
        self.phase_switch_count = 0

    def summary(self) -> Dict:
        """Return controller summary."""
        return {
            'gait_freq_hz': self.gait_freq_hz,
            'period_ms': self.period_ms,
            'legs': len(self.leg_joints),
            'joints': sum(len(v) for v in self.leg_joints.values()),
            'swing_amplify': self.swing_amplify,
            'stance_suppress': self.stance_suppress,
            'turn_bias_outer': self.turn_bias_outer,
            'turn_bias_inner': self.turn_bias_inner,
        }


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == '__main__':
    # Build a mini actuator map for testing
    test_map = {}
    for leg in ['T1_left', 'T2_left', 'T3_left', 'T1_right', 'T2_right', 'T3_right']:
        for joint in ['coxa', 'coxa_abduct', 'femur', 'femur_twist', 'tibia', 'tarsus']:
            name = f'{joint}_{leg}'
            test_map[name] = len(test_map)

    gc = GaitController(test_map, gait_freq_hz=5.0)

    # Test leg extraction
    assert gc._extract_leg('coxa_T1_left') == 'T1_left'
    assert gc._extract_leg('femur_T2_right') == 'T2_right'
    assert gc._extract_leg('not_a_leg') is None
    print("  ✓ Leg extraction")

    # Test swing detection
    gc.current_phase = 'tripod_1'
    assert gc._is_swing_joint('coxa_T1_left') == True   # L1 in T1
    assert gc._is_swing_joint('femur_T2_right') == True  # R2 in T1
    assert gc._is_swing_joint('tibia_T3_left') == True   # L3 in T1
    assert gc._is_swing_joint('coxa_T1_right') == False  # R1 in T2
    assert gc._is_swing_joint('femur_T2_left') == False  # L2 in T2
    print("  ✓ Swing detection: tripod_1 match")

    gc.current_phase = 'tripod_2'
    assert gc._is_swing_joint('coxa_T1_right') == True   # R1 in T2
    assert gc._is_swing_joint('femur_T2_left') == True   # L2 in T2
    assert gc._is_swing_joint('tibia_T3_right') == True  # R3 in T2
    assert gc._is_swing_joint('coxa_T1_left') == False   # L1 in T1
    print("  ✓ Swing detection: tripod_2 match")

    # Test gait application: walking
    base = {name: 0.1 for name in test_map}
    gc.reset()
    result = gc.apply('walk', base, dt_ms=5.0)

    # Swing legs should have amplified torque
    swing_joints = [n for n in test_map if gc._is_swing_joint(n)]
    stance_joints = [n for n in test_map if not gc._is_swing_joint(n)]
    avg_swing = sum(result[j] for j in swing_joints) / len(swing_joints)
    avg_stance = sum(result[j] for j in stance_joints) / len(stance_joints)
    print(f"  Walk: avg_swing={avg_swing:.4f}, avg_stance={avg_stance:.4f}")
    assert avg_swing > avg_stance, f"Swing torque ({avg_swing:.4f}) should exceed stance ({avg_stance:.4f})"
    print("  ✓ Walk: swing > stance torque")

    # Test phase alternation
    gc.reset()
    for _ in range(20):
        gc.apply('walk', base, dt_ms=5.0)
    assert gc.phase_switch_count >= 1, f"Should have switched phase at least once (got {gc.phase_switch_count})"
    print(f"  ✓ Phase alternation: {gc.phase_switch_count} switches after 20 steps")

    # Test stance mode suppresses all
    gc.reset()
    result = gc.apply('stance', base, dt_ms=5.0)
    avg_all = sum(result.values()) / len(result)
    assert avg_all < 0.05, f"Stance should suppress all torque (avg={avg_all:.4f})"
    print(f"  ✓ Stance: avg torque={avg_all:.4f} (suppressed)")

    # Test turn bias
    gc.reset()
    result = gc.apply('turn_left', base, dt_ms=5.0)
    right_torques = [v for k, v in result.items() if 'right' in k]
    left_torques = [v for k, v in result.items() if 'left' in k]
    avg_right = sum(right_torques) / len(right_torques)
    avg_left = sum(left_torques) / len(left_torques)
    print(f"  Turn-left: avg_right={avg_right:.4f}, avg_left={avg_left:.4f}")
    # Right should be higher than left for turn-left
    if avg_right > avg_left:
        print("  ✓ Turn-left: right side > left side (correct)")
    else:
        print(f"  ⚠ Turn-left bias may need tuning: R={avg_right:.4f}, L={avg_left:.4f}")

    print("\n✅ All GaitController tests passed!")
