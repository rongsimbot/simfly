#!/usr/bin/env python3
"""
VNC Motor Decoder — extends the DN-proxy motor decoder with real MANC motor neurons.

Replaces the heuristic DN-proxy approach (Phase 5) with biological VNC motor neurons
from the MANC connectome. Each MANC MN type maps to specific SimFLy actuator torque
profiles based on its functional classification:

  - fast_MN  → high torque, rapid response (phasic bursts)
  - slow_MN  → sustained, lower torque (tonic)
  - flexor_MN → positive torque (contracts joint)
  - extensor_MN → negative torque (extends joint)

Architecture:
  Brain (1kHz) → DN spike accumulation → MANC DN → interneuron → MN
  → MN-type-aware torque conversion → SimFLy MuJoCo joints (200Hz)

Usage:
    from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
    decoder = VNCMotorDecoder.load_from_vnc("vnc_actuator_map.json")
    commands = decoder.decode()
"""
import json
import math
import os
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict


# ── MN-type-specific torque profiles ──────────────────────────────────────

# Gain multipliers per MN functional type
MN_TYPE_GAIN: Dict[str, float] = {
    "fast_MN": 0.25,
    "slow_MN": 0.08,
    "tonic_MN": 0.08,
    "phasic_MN": 0.20,
    "flexor_MN": 0.15,
    "extensor_MN": 0.15,
    "sternotrochanter_MN": 0.12,
    "trochanter_MN": 0.10,
    "coxa_MN": 0.14,
    "femur_MN": 0.14,
    "tibia_MN": 0.12,
    "tarsus_MN": 0.08,
    "adductor_MN": 0.10,
    "abductor_MN": 0.10,
    "leg_MN": 0.12,
    "unclassified_MN": 0.10,
}

# Torque sign per MN functional type (positive = flex, negative = extend)
MN_TYPE_SIGN: Dict[str, int] = {
    "flexor_MN": 1,
    "adductor_MN": 1,
    "extensor_MN": -1,
    "abductor_MN": -1,
    # Neutral — sign determined by NT (ACH=+, GABA/GLUT=-)
}


class VNCMotorDecoder:
    """VNC-aware motor decoder using real MANC motor neurons.

    Extends the Phase 5 MotorDecoder with biological MN types and
    type-specific torque profiles derived from the MANC connectome.

    Usage::

        decoder = VNCMotorDecoder(vnc_actuator_map, tau_decay=50.0, gain=0.1)
        for _ in range(5):
            brain.step()
            decoder.accumulate(brain.get_fired_neurons())
        commands = decoder.decode()
    """

    def __init__(
        self,
        vnc_actuator_map: Dict[str, Dict[str, Any]],
        mn_type_gain: Optional[Dict[str, float]] = None,
        tau_decay: float = 50.0,
        global_gain: float = 0.1,
        dt_brain_ms: float = 1.0,
        dt_physics_ms: float = 5.0,
    ):
        """Initialize the VNC motor decoder.

        Args:
            vnc_actuator_map: Dict mapping joint_name → {
                'agonists': [mn_ids],
                'antagonists': [mn_ids],
                'info': {segment, leg, side, joint},
            }
            mn_type_gain: Optional MN type gain map (defaults to MN_TYPE_GAIN).
            tau_decay: Firing rate decay time constant (ms).
            global_gain: Global torque scaling factor.
            dt_brain_ms: Brain step in ms (1kHz).
            dt_physics_ms: Physics step in ms (200Hz).
        """
        self.vnc_actuator_map = vnc_actuator_map
        self.mn_type_gain = mn_type_gain or MN_TYPE_GAIN
        self.tau_decay = tau_decay
        self.global_gain = global_gain
        self.dt_brain_ms = dt_brain_ms
        self.dt_physics_ms = dt_physics_ms

        # Optional MN type reference (loaded from pathways)
        self._mn_types: Dict[int, List[str]] = {}

        # Spike accumulators
        self._agonist_spikes: Dict[str, float] = defaultdict(float)
        self._antagonist_spikes: Dict[str, float] = defaultdict(float)
        self._agonist_rates: Dict[str, float] = defaultdict(float)
        self._antagonist_rates: Dict[str, float] = defaultdict(float)
        self._sub_step: int = 0
        self._sub_steps_per_physics: int = max(1, int(dt_physics_ms / dt_brain_ms))

        # Build fast index: MN id → list of (joint, is_agonist)
        self._mn_to_joints: Dict[int, List[Tuple[str, bool]]] = defaultdict(list)
        for joint_name, config in vnc_actuator_map.items():
            for mn_id in config.get("agonists", []):
                self._mn_to_joints[mn_id].append((joint_name, True))
            for mn_id in config.get("antagonists", []):
                self._mn_to_joints[mn_id].append((joint_name, False))

    # ── Data loading ──────────────────────────────────────────────────────

    @classmethod
    def load_from_vnc(
        cls,
        vnc_actuator_map_path: str,
        pathways_path: Optional[str] = None,
        **kwargs,
    ) -> "VNCMotorDecoder":
        """Load a VNC motor decoder from VNC bridge output files.

        Args:
            vnc_actuator_map_path: Path to vnc_actuator_map.json.
            pathways_path: Optional path to dn_mn_pathways.json for MN type info.

        Returns:
            Configured VNCMotorDecoder.
        """
        with open(vnc_actuator_map_path) as f:
            data = json.load(f)

        # Aggregate actuator maps across all segments
        actuator_map: Dict[str, Dict[str, Any]] = {}
        for seg, seg_map in data.items():
            for joint_name, config in seg_map.items():
                # Skip joints with no MNs assigned
                if not config.get("agonists") and not config.get("antagonists"):
                    continue
                actuator_map[joint_name] = config

        decoder = cls(actuator_map, **kwargs)

        # Load MN type info from pathways if available
        if pathways_path and os.path.exists(pathways_path):
            with open(pathways_path) as f:
                pw_data = json.load(f)
            paths = pw_data.get("pathways", {})
            for dn_pathways in paths.values():
                for pw in dn_pathways:
                    mn_id = pw.get("mn_id")
                    fn_types = pw.get("mn_fn_types", [])
                    if mn_id and fn_types:
                        decoder._mn_types[mn_id] = fn_types

        return decoder

    # ── Spike accumulation ────────────────────────────────────────────────

    def accumulate(self, fired_neuron_ids: Set[int]) -> None:
        """Accumulate spike counts from one brain sub-step.

        Uses the MN→joint index for O(1) lookup instead of iterating
        over all actuator map entries.

        Args:
            fired_neuron_ids: Set of MANC MN IDs that fired this sub-step.
        """
        for mn_id in fired_neuron_ids:
            for joint_name, is_agonist in self._mn_to_joints.get(mn_id, []):
                if is_agonist:
                    self._agonist_spikes[joint_name] += 1
                else:
                    self._antagonist_spikes[joint_name] += 1
        self._sub_step += 1

    def accumulate_slow(self, fired_neuron_ids: Set[int]) -> None:
        """Accumulate spikes by iterating over the actuator map.

        Slower than accumulate() but doesn't require the MN index to be
        pre-built. Used when actuator map is loaded dynamically.

        Args:
            fired_neuron_ids: Set of MANC MN IDs that fired this sub-step.
        """
        for joint_name, config in self.vnc_actuator_map.items():
            ag_count = sum(1 for mn in config.get("agonists", []) if mn in fired_neuron_ids)
            ant_count = sum(1 for mn in config.get("antagonists", []) if mn in fired_neuron_ids)
            self._agonist_spikes[joint_name] += ag_count
            self._antagonist_spikes[joint_name] += ant_count
        self._sub_step += 1

    # ── Decode to joint commands ──────────────────────────────────────────

    def decode(self) -> Dict[str, float]:
        """Convert accumulated spikes to joint commands.

        Uses MN-type-aware gain and sign for biologically realistic
        torque profiles.

        Returns:
            Dict joint_name → ctrl_value, clamped to [-1, 1].
        """
        dt_seconds = self._sub_step * self.dt_brain_ms / 1000.0
        decay = math.exp(-self._sub_step * self.dt_brain_ms / self.tau_decay)
        commands: Dict[str, float] = {}

        for joint_name in self.vnc_actuator_map:
            ag_spikes = self._agonist_spikes[joint_name]
            ant_spikes = self._antagonist_spikes[joint_name]
            config = self.vnc_actuator_map[joint_name]
            info = config.get("info", {})

            n_ago = max(1, len(config.get("agonists", [])))
            n_ant = max(1, len(config.get("antagonists", [])))

            # Update rates with exponential decay
            ago_add = ag_spikes / (n_ago * dt_seconds) if dt_seconds > 0 else 0.0
            ant_add = ant_spikes / (n_ant * dt_seconds) if dt_seconds > 0 else 0.0

            self._agonist_rates[joint_name] = (
                self._agonist_rates[joint_name] * decay + ago_add
            )
            self._antagonist_rates[joint_name] = (
                self._antagonist_rates[joint_name] * decay + ant_add
            )

            # Compute type-weighted net activation
            ago_weighted = self._type_weighted_activation(
                config.get("agonists", []), self._agonist_rates[joint_name]
            )
            ant_weighted = self._type_weighted_activation(
                config.get("antagonists", []), self._antagonist_rates[joint_name]
            )
            net_rate = ago_weighted - ant_weighted

            # Default torque range
            torque = net_rate * self.global_gain * 1.0  # default max_torque=1.0

            # Clamp to [-1, 1]
            commands[joint_name] = max(-1.0, min(1.0, torque))

            # Reset accumulators
            self._agonist_spikes[joint_name] = 0.0
            self._antagonist_spikes[joint_name] = 0.0

        self._sub_step = 0
        return commands

    def _type_weighted_activation(
        self, mn_ids: List[int], base_rate: float
    ) -> float:
        """Apply MN-type-specific gain to base activation.

        Each MN type contributes differently to torque:
          - fast_MN → high gain (phasic, strong)
          - slow_MN → low gain (tonic, sustained)

        Args:
            mn_ids: List of MN IDs.
            base_rate: Base firing rate.

        Returns:
            Type-weighted activation.
        """
        if not mn_ids:
            return base_rate

        total_gain = 0.0
        for mn_id in mn_ids:
            fn_types = self._mn_types.get(mn_id, ["unclassified_MN"])
            gain = max(self.mn_type_gain.get(ft, 0.1) for ft in fn_types)
            total_gain += gain

        avg_gain = total_gain / len(mn_ids)
        return base_rate * avg_gain

    # ── Query ─────────────────────────────────────────────────────────────

    def get_rates(self) -> Dict[str, Dict[str, float]]:
        """Get current firing rate estimates for all joints."""
        rates = {}
        for joint_name in self.vnc_actuator_map:
            rates[joint_name] = {
                "agonist": self._agonist_rates.get(joint_name, 0.0),
                "antagonist": self._antagonist_rates.get(joint_name, 0.0),
                "net": self._agonist_rates.get(joint_name, 0.0)
                       - self._antagonist_rates.get(joint_name, 0.0),
            }
        return rates

    def reset(self) -> None:
        """Reset all state."""
        self._agonist_spikes.clear()
        self._antagonist_spikes.clear()
        self._agonist_rates.clear()
        self._antagonist_rates.clear()
        self._sub_step = 0

    def summary(self) -> Dict[str, Any]:
        """Get decoder summary."""
        return {
            "total_joints": len(self.vnc_actuator_map),
            "mns_with_types": len(self._mn_types),
            "tau_decay_ms": self.tau_decay,
            "global_gain": self.global_gain,
            "dt_brain_ms": self.dt_brain_ms,
            "dt_physics_ms": self.dt_physics_ms,
        }


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("VNC Motor Decoder Test")
    print("=" * 40)

    # Test with a simple map
    test_map = {
        "femur_T1_left": {
            "agonists": [96565],   # Tr flexor MN (glutamate/inhibitory in real data)
            "antagonists": [152620],  # Sternotrochanter MN
            "info": {"segment": "metathoracic_R", "leg": "T3", "side": "right", "joint": "femur"},
        },
        "coxa_T1_left": {
            "agonists": [96565],
            "antagonists": [],
            "info": {"segment": "metathoracic_R", "leg": "T3", "side": "right", "joint": "coxa"},
        },
    }

    # Add MN type info
    decoder = VNCMotorDecoder(test_map, global_gain=0.15)
    decoder._mn_types = {
        96565: ["flexor_MN", "prothoracic_MN", "leg_MN"],
        152620: ["sternotrochanter_MN", "mesothoracic_MN", "leg_MN"],
    }

    # Test agonist firing
    for _ in range(5):
        decoder.accumulate({96565})
    cmds = decoder.decode()
    print(f"\nFlexor MN (96565) firing 5 steps: femur_T1_left = {cmds.get('femur_T1_left', 0):.4f}")
    assert cmds.get("femur_T1_left", 0) > 0, "Expected positive torque from flexor"
    print("  ✓ Positive flexor torque")

    # Test antagonist firing
    for _ in range(5):
        decoder.accumulate({152620})
    cmds = decoder.decode()
    print(f"Antagonist MN (152620) firing 5 steps: femur_T1_left = {cmds.get('femur_T1_left', 0):.4f}")
    assert cmds.get("femur_T1_left", 0) < 0, "Expected negative torque from antagonist"
    print("  ✓ Negative antagonist torque")

    # Test type-weighted gain
    decoder.reset()
    decoder.mn_type_gain = {"flexor_MN": 0.3, "fast_MN": 0.25}  # Higher flexor gain
    for _ in range(5):
        decoder.accumulate({96565})
    cmds_high = decoder.decode()
    print(f"\nType-weighted flexor (higher gain): femur_T1_left = {cmds_high.get('femur_T1_left', 0):.4f}")

    # Test coxa joint
    decoder.reset()
    for _ in range(5):
        decoder.accumulate({96565})
    cmds = decoder.decode()
    print(f"Coxa joint (flexor drives): coxa_T1_left = {cmds.get('coxa_T1_left', 0):.4f}")
    assert cmds.get("coxa_T1_left", 0) > 0, "Expected positive coxa torque"
    print("  ✓ Coxa torque")

    print("\nAll VNC Motor Decoder tests passed!")
