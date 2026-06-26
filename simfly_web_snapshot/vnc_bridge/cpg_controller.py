#!/usr/bin/env python3
"""
Phase 20: VNC Central Pattern Generator (CPG)
==============================================
Biologically-inspired tripod gait CPG for the VNC bridge.

Architecture:
  Brain DNs → CPG onset/offset → Rhythmic MN modulation → Decoder → Joints

Each of 6 legs has a CPG oscillator pair (flexor/extensor phase).
CPGs are phase-locked: T1_left + T2_right + T3_left fire together (tripod A)
                        T1_right + T2_left + T3_right fire together (tripod B)

DN activity from the connectome provides:
  - Onset trigger: "start walking"
  - Speed control: DN firing rate → CPG frequency
  - Direction bias: asymmetric DN activity → turn left/right

Sensory feedback modulates:
  - Ground contact → phase reset
  - Leg load → stance phase duration
"""

import math
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ── Tripod groupings (biologically accurate for Drosophila) ──
TRIPOD_A = ["T1_left", "T2_right", "T3_left"]   # L1 + R2 + L3
TRIPOD_B = ["T1_right", "T2_left", "T3_right"]   # R1 + L2 + R3

# Joint groups per leg
LEG_JOINTS = {
    "T1_left":  ["coxa_T1_left", "coxa_abduct_T1_left", "coxa_twist_T1_left",
                 "femur_T1_left", "femur_twist_T1_left", "tibia_T1_left"],
    "T1_right": ["coxa_T1_right", "coxa_abduct_T1_right", "coxa_twist_T1_right",
                 "femur_T1_right", "femur_twist_T1_right", "tibia_T1_right"],
    "T2_left":  ["coxa_T2_left", "coxa_abduct_T2_left", "coxa_twist_T2_left",
                 "femur_T2_left", "femur_twist_T2_left", "tibia_T2_left"],
    "T2_right": ["coxa_T2_right", "coxa_abduct_T2_right", "coxa_twist_T2_right",
                 "femur_T2_right", "femur_twist_T2_right", "tibia_T2_right"],
    "T3_left":  ["coxa_T3_left", "coxa_abduct_T3_left", "coxa_twist_T3_left",
                 "femur_T3_left", "femur_twist_T3_left", "tibia_T3_left"],
    "T3_right": ["coxa_T3_right", "coxa_abduct_T3_right", "coxa_twist_T3_right",
                 "femur_T3_right", "femur_twist_T3_right", "tibia_T3_right"],
}

# Joint type roles in gait
STANCE_JOINTS = ["coxa", "femur", "tibia"]       # extend during stance
SWING_JOINTS = ["coxa_abduct", "femur_twist"]     # lift during swing
ROTATION_JOINTS = ["coxa_twist"]                   # subtle rotation

def joint_gait_role(joint_name: str) -> str:
    """Determine a joint's role in the gait cycle."""
    for role, patterns in [("stance", STANCE_JOINTS), ("swing", SWING_JOINTS), ("rotation", ROTATION_JOINTS)]:
        for pat in patterns:
            if pat in joint_name:
                return role
    return "stance"


class MatsuokaOscillator:
    """Single Matsuoka neural oscillator — two neurons with mutual inhibition.
    
    Produces rhythmic output with:
    - Tunable frequency
    - Adjustable duty cycle
    - Smooth sinusoidal-like waveform
    - Natural response to tonic input
    """
    
    def __init__(self, tau: float = 0.05, tau_adapt: float = 0.5, 
                 beta: float = 2.5, w_inhibit: float = 2.0):
        self.tau = tau              # membrane time constant (s)
        self.tau_adapt = tau_adapt  # adaptation time constant (s)
        self.beta = beta            # adaptation strength
        self.w_inhibit = w_inhibit  # mutual inhibition weight
        
        # State variables
        self.u1 = 0.1   # flexor neuron potential
        self.u2 = 0.0   # extensor neuron potential
        self.v1 = 0.0   # flexor adaptation
        self.v2 = 0.0   # extensor adaptation
        self.y1 = 0.0   # flexor output
        self.y2 = 0.0   # extensor output
    
    def step(self, tonic_input: float, dt: float = 0.005) -> Tuple[float, float]:
        """Advance oscillator by dt seconds.
        
        Args:
            tonic_input: Constant excitatory drive (0=stop, 1=fast walk)
            dt: Time step in seconds (match physics rate)
        
        Returns:
            (flexor_output, extensor_output) — each in [0, 1]
        """
        # Clamp tonic input
        c = max(0.0, min(1.0, tonic_input))
        
        # Neuron 1 (flexor phase)
        du1 = (-self.u1 - self.beta * self.v1 
               - self.w_inhibit * max(0.0, self.y2) + c) / self.tau
        dv1 = (-self.v1 + max(0.0, self.y1)) / self.tau_adapt
        
        # Neuron 2 (extensor phase)
        du2 = (-self.u2 - self.beta * self.v2 
               - self.w_inhibit * max(0.0, self.y1) + c) / self.tau
        dv2 = (-self.v2 + max(0.0, self.y2)) / self.tau_adapt
        
        # Euler integration
        self.u1 += du1 * dt
        self.u2 += du2 * dt
        self.v1 += dv1 * dt
        self.v2 += dv2 * dt
        
        # Rectified output
        self.y1 = max(0.0, self.u1)
        self.y2 = max(0.0, self.u2)
        
        return self.y1, self.y2
    
    def reset(self):
        """Reset oscillator state."""
        self.u1 = 0.1
        self.u2 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0


class VNC_CPG:
    """Central Pattern Generator for 6-leg tripod gait.
    
    Maps brain DN activity to rhythmic leg commands:
    - Tripod A (L1+R2+L3) and Tripod B (R1+L2+R3) alternate
    - DN firing rate controls walking speed/frequency
    - Sensory feedback modulates phase timing
    """
    
    def __init__(self, dt_physics: float = 0.005):
        self.dt = dt_physics
        
        # One oscillator per leg (flexor=stance, extensor=swing)
        self.oscillators: Dict[str, MatsuokaOscillator] = {}
        for leg in LEG_JOINTS:
            self.oscillators[leg] = MatsuokaOscillator()
        
        # Phase coupling: tripod A and B are anti-phase
        self._phase_offset = math.pi  # 180 degrees between tripods
        
        # Current DN activity tracking
        self.dn_walk_activity: float = 0.0
        self.dn_speed_mod: float = 0.0
        self.dn_turn_bias: float = 0.0  # -1=left, +1=right
        
        # Gait state
        self.is_walking: bool = False
        self.walk_onset_threshold: float = 0.3  # DN activity to start walking
        self.walk_offset_threshold: float = 0.1  # DN activity to stop
        
        # Output cache
        self.leg_phases: Dict[str, float] = {}     # 0=stance, 1=swing peak
        self.joint_modulations: Dict[str, float] = {}  # per-joint MN modulation factor
    
    def update_dn_input(self, dn_firing_rate: float, turn_bias: float = 0.0):
        """Update brain signals to the CPG.
        
        Args:
            dn_firing_rate: Normalized DN activity [0, 1] from connectome
                           (fraction of walk-related DNs firing)
            turn_bias: Asymmetric DN activity for turning [-1, 1]
        """
        self.dn_walk_activity = dn_firing_rate
        self.dn_turn_bias = turn_bias
        
        # Walking state transition
        if not self.is_walking and dn_firing_rate >= self.walk_onset_threshold:
            self.is_walking = True
            self._reset_all_oscillators()
        elif self.is_walking and dn_firing_rate < self.walk_offset_threshold:
            self.is_walking = False
    
    def step(self, sensory_feedback: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict[str, float]]:
        """Advance CPG one physics step.
        
        Args:
            sensory_feedback: Dict leg_id → {on_ground: bool, load: float}
        
        Returns:
            Dict mapping joint_name → {modulation: float, phase: str}
            modulation is a multiplier on MN activation [0, 2]
        """
        if not self.is_walking:
            # Not walking: return neutral modulation (1.0 = no change)
            result = {}
            for leg, joints in LEG_JOINTS.items():
                for j in joints:
                    result[j] = {"modulation": 1.0, "phase": "stand"}
            return result
        
        # Walking: compute CPG rhythm
        # Base tonic drive from DN activity
        tonic_base = self.dn_walk_activity
        
        # Apply turn bias (asymmetric left/right drive)
        left_drive = tonic_base * (1.0 - max(0, self.dn_turn_bias))  # slow left legs when turning left
        right_drive = tonic_base * (1.0 + max(0, self.dn_turn_bias))  # speed up right legs
        
        # Step each leg's oscillator
        for leg_name, osc in self.oscillators.items():
            if "left" in leg_name:
                tonic = left_drive
            else:
                tonic = right_drive
            
            # Sensory feedback: ground contact prolongs stance
            if sensory_feedback and leg_name in sensory_feedback:
                fb = sensory_feedback[leg_name]
                if fb.get("on_ground", True):
                    tonic *= 0.9  # Slightly reduce drive during ground contact
                if fb.get("load", 0) > 0.5:
                    tonic *= 0.8  # Heavy load → slower oscillation
            
            osc.step(tonic, self.dt)
        
        # Compute leg phases and joint modulations
        result = {}
        for leg_name, osc in self.oscillators.items():
            flexor_out, extensor_out = osc.y1, osc.y2
            
            # Phase: 0 = full stance (flexor active), 1 = full swing (extensor active)
            phase = extensor_out / max(flexor_out + extensor_out, 0.001)
            self.leg_phases[leg_name] = phase
            
            # Determine if tripod A or B
            is_tripod_a = leg_name in TRIPOD_A
            
            # Joint-level modulation
            for joint_name in LEG_JOINTS[leg_name]:
                role = joint_gait_role(joint_name)
                
                if role == "stance":
                    # Stance joints extend during flexor phase
                    modulation = 1.0 + flexor_out * 1.5
                elif role == "swing":
                    # Swing joints activate during extensor phase
                    modulation = 1.0 + extensor_out * 1.5
                else:
                    # Rotation joints have subtle modulation
                    modulation = 1.0 + (flexor_out - extensor_out) * 0.3
                
                self.joint_modulations[joint_name] = modulation
                
                result[joint_name] = {
                    "modulation": round(modulation, 4),
                    "phase": "stance" if phase < 0.5 else "swing",
                    "leg": leg_name,
                    "tripod": "A" if is_tripod_a else "B",
                }
        
        return result
    
    def apply_to_decoder(self, decoder_output: Dict[str, float]) -> Dict[str, float]:
        """Apply CPG modulation to decoder output torques.
        
        The CPG modulates (doesn't replace) the connectome's torque signals.
        During swing phase, extensor torque is reduced and flexor enhanced.
        During stance phase, extensor torque is maintained.
        """
        modulated = {}
        for joint_name, torque in decoder_output.items():
            mod_info = self.joint_modulations.get(joint_name, 1.0)
            role = joint_gait_role(joint_name)
            
            if role == "stance":
                # Stance: maintain torque, slight enhancement
                modulated[joint_name] = torque * mod_info
            elif role == "swing":
                # Swing: reduce anti-swing torque, enhance swing direction
                modulated[joint_name] = torque * mod_info
            else:
                modulated[joint_name] = torque * mod_info
        
        return modulated
    
    def _reset_all_oscillators(self):
        """Reset all oscillators with tripod phase offsets."""
        for i, (leg_name, osc) in enumerate(self.oscillators.items()):
            osc.reset()
            # Initialize tripod A at different phase than tripod B
            if leg_name in TRIPOD_A:
                # Tripod A starts in stance (flexor active)
                osc.u1 = 0.5
                osc.y1 = 0.5
            else:
                # Tripod B starts in swing (extensor active)
                osc.u2 = 0.5
                osc.y2 = 0.5
    
    def get_state(self) -> Dict:
        """Get current CPG state for monitoring."""
        return {
            "is_walking": self.is_walking,
            "dn_walk_activity": round(self.dn_walk_activity, 3),
            "dn_turn_bias": round(self.dn_turn_bias, 3),
            "leg_phases": {k: round(v, 3) for k, v in self.leg_phases.items()},
            "tripod_a_active": all(
                self.leg_phases.get(leg, 0) < 0.5 for leg in TRIPOD_A
            ),
            "tripod_b_active": all(
                self.leg_phases.get(leg, 0) < 0.5 for leg in TRIPOD_B
            ),
        }


# ── Integration with Brain2 server ──

def extract_dn_walk_activity(status: Dict, dn_matches: int) -> float:
    """Extract walking-related DN activity from connectome status.
    
    Maps DN match count and firing patterns to a walking drive signal.
    953 total DNs in the bridge — a subset are locomotion-related.
    """
    total_dns = status.get("dns", 953)
    fired_neurons = status.get("metrics", {}).get("fired_neurons", 0)
    
    # Walking DNs are ~15% of total DNs in Drosophila
    walk_dn_fraction = dn_matches / max(total_dns * 0.15, 1)
    
    # Normalize: typical walk onset at 20-30% of walk DNs firing
    activity = min(1.0, walk_dn_fraction)
    
    return activity


def extract_turn_bias(status: Dict) -> float:
    """Extract turning bias from asymmetric DN/visual activity.
    
    Uses looming and contrast asymmetry as proxies for turn direction.
    """
    metrics = status.get("metrics", {})
    looming = metrics.get("looming", 0)
    contrast = metrics.get("contrast", 0)
    
    # Simple proxy: use vision to determine turn direction
    # In a real fly, visual asymmetry drives turning
    # For now, introduce small random exploration bias if DN activity high
    import random
    if metrics.get("fired_neurons", 0) > 3000:
        return random.uniform(-0.3, 0.3)
    return 0.0


if __name__ == "__main__":
    # Demo: test CPG with simulated DN input
    print("=" * 60)
    print("VNC CPG Test — Simulated Walking")
    print("=" * 60)
    
    cpg = VNC_CPG(dt_physics=0.005)
    
    # Simulate 5 seconds of walking
    steps = 1000  # 1000 steps * 5ms = 5s
    phases_log = defaultdict(list)
    
    cpg.update_dn_input(0.5, 0.0)  # Walk at half speed, straight
    
    for i in range(steps):
        result = cpg.step()
        
        if i % 100 == 0:
            state = cpg.get_state()
            print(f"\nStep {i} ({i*5}ms): walking={state['is_walking']}")
            print(f"  Tripod A (L1+R2+L3): {'STANCE' if state['tripod_a_active'] else 'SWING'}")
            print(f"  Tripod B (R1+L2+R3): {'STANCE' if state['tripod_b_active'] else 'SWING'}")
            
            # Show sample joint modulations
            for j in ["femur_T1_left", "femur_T1_right", "femur_T2_left", "femur_T2_right"]:
                mod = result[j]["modulation"]
                phase = result[j]["phase"]
                bar = "█" * int(mod * 10)
                print(f"  {j:20s} {phase:6s} mod={mod:.2f} {bar}")
    
    print(f"\n  CPG test complete — {steps} steps simulated")
