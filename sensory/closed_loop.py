#!/usr/bin/env python3
"""
Closed-Loop Sensory-Motor Loop — the main sense→think→act coordinator.

Phase 8: Coordinates the full connectome-driven simulation pipeline:
  1. Read MuJoCo sensors (vision, mechano, proprio, chemo)
  2. Encode sensory data as spike trains
  3. Inject sensory spikes into NIRON engine
  4. Run NIRON fire cycle (Fire1 → Fire2 → Fire3)
  5. Decode motor neuron output
  6. Convert to MuJoCo actuator commands
  7. Step physics simulation
  8. Repeat

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │                  ClosedLoop                           │
  │                                                       │
  │  [1] read_sensors()        ← MuJoCo data/sensors      │
  │  [2] sensory_injector      → encode to spikes         │
  │  [3] inject → NIRON        → add to fire_list_1       │
  │  [4] engine.fire()          → Fire1 → Fire2 → Fire3   │
  │  [5] read_motor_output()   ← engine fire_list_2       │
  │  [6] vnc_decoder.decode()  → joint commands           │
  │  [7] apply_to_mujoco()     → data.ctrl[:]             │
  │  [8] mujoco.mj_step()      → physics update           │
  │  [9] render_frame()        → optional video frame     │
  └──────────────────────────────────────────────────────┘
"""

import json
import math
import os
import sys
import numpy as np
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


class ClosedLoop:
    """Main closed-loop coordinator for connectome-driven simulation.
    
    Orchestrates the full pipeline: sensory input → brain processing →
    motor output → physics update.
    
    Usage:
        loop = ClosedLoop(engine, injector, decoder, mujoco_model, mujoco_data)
        for step in range(1000):
            loop.step()
    """
    
    def __init__(
        self,
        engine,               # NIRON NeuronArrayBase
        injector,             # SensoryInjector
        vnc_decoder,          # VNCMotorDecoder
        model,                # MuJoCo model
        data,                 # MuJoCo data
        renderer=None,        # Optional renderer
        actuator_map: Optional[Dict[str, int]] = None,  # joint_name → ctrl_idx
        bridge=None,          # Phase 9: DnMnBridge (optional)
        brain_rate_hz: int = 1000,
        physics_rate_hz: int = 200,
        w_syn: float = 1.0,
        sensory_feedback_enabled: bool = True,
        enable_vision: bool = True,
        enable_proprioception: bool = True,
        enable_touch: bool = True,
        enable_chemo: bool = False,
    ):
        """Initialize the closed-loop coordinator.
        
        Args:
            engine: NIRON NeuronArrayBase instance.
            injector: SensoryInjector instance.
            vnc_decoder: VNCMotorDecoder instance.
            model: MuJoCo model (from mujoco.MjModel.from_xml_path).
            data: MuJoCo data (from mujoco.MjData).
            renderer: Optional MuJoCo renderer instance.
            actuator_map: Dict mapping joint_name → ctrl array index.
            brain_rate_hz: Brain simulation rate (Hz).
            physics_rate_hz: Physics simulation rate (Hz).
            w_syn: Global synaptic weight scalar.
            sensory_feedback_enabled: Enable sensory feedback loop.
            enable_vision: Enable visual input.
            enable_proprioception: Enable proprioceptive feedback.
            enable_touch: Enable touch/contact feedback.
            enable_chemo: Enable chemosensory input.
        """
        self.engine = engine
        self.injector = injector
        self.vnc_decoder = vnc_decoder
        self.model = model
        self.data = data
        self.renderer = renderer
        
        # Actuator mapping
        self.actuator_map = actuator_map or {}
        # Phase 9: DN->MN bridge (optional)
        self.bridge = bridge
        
        # Timing
        self.brain_rate_hz = brain_rate_hz
        self.physics_rate_hz = physics_rate_hz
        self.dt_brain_ms = 1000.0 / brain_rate_hz
        self.dt_physics_ms = 1000.0 / physics_rate_hz
        self.brain_steps_per_physics = max(1, brain_rate_hz // physics_rate_hz)
        
        # Free parameter (Nature 2024 paper)
        self.w_syn = w_syn
        
        # Feedback flags
        self.sensory_feedback_enabled = sensory_feedback_enabled
        self.enable_vision = enable_vision
        self.enable_proprioception = enable_proprioception
        self.enable_touch = enable_touch
        self.enable_chemo = enable_chemo
        
        # State
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.brain_steps_this_physics = 0
        self.bridge_torque_buffer = None  # Phase 9 bridge
        
        # Metrics
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._step_start_times: List[float] = []
        
        # Joint/sensor cache
        self._joint_names: List[str] = []
        self._joint_indices: Dict[str, int] = {}
        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu
        
        # Initial joint angles for proprioceptive baseline
        self._initial_joint_angles: List[float] = []
    
    def initialize(self) -> None:
        """Initialize state after construction. Call once before stepping."""
        # Cache initial joint positions
        if hasattr(self.data, 'qpos') and self._nq > 0:
            self._initial_joint_angles = list(self.data.qpos[:self._nq])
        
        # Build joint name index
        try:
            import mujoco
            for i in range(min(self._nv, self.model.nv)):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name:
                    self._joint_names.append(name)
                    self._joint_indices[name] = i
        except Exception:
            pass  # Will work with indices directly
    
    def read_sensors(self) -> Dict[str, Any]:
        """Read all sensor data from MuJoCo simulation state.
        
        Returns:
            Dict with keys: joint_angles, joint_velocities, contact_forces,
            body_position, body_orientation, sensor_data.
        """
        sensors: Dict[str, Any] = {}
        
        try:
            # Joint positions (proprioception)
            if hasattr(self.data, 'qpos') and self._nq > 0:
                # Skip the free joint (first 7 for free joint: 3 pos + 4 quat)
                start = 7 if self._nq > 7 else 0
                joint_pos = list(self.data.qpos[start:self._nq])
                sensors['joint_angles'] = joint_pos
            else:
                sensors['joint_angles'] = []
            
            # Joint velocities
            if hasattr(self.data, 'qvel') and self._nv > 0:
                # Skip free joint velocities (first 6)
                start = 6 if self._nv > 6 else 0
                sensors['joint_velocities'] = list(self.data.qvel[start:self._nv])
            else:
                sensors['joint_velocities'] = []
            
            # Contact forces
            if hasattr(self.data, 'contact') and hasattr(self.data, 'ncon'):
                contacts = []
                for i in range(min(50, self.data.ncon)):  # Cap at 50 contacts
                    try:
                        contact = self.data.contact[i]
                        contacts.append((contact.frame[0], contact.frame[1], contact.frame[2]))
                    except (IndexError, AttributeError):
                        break
                sensors['contact_forces'] = contacts
            else:
                sensors['contact_forces'] = []
            
            # Body position (for vision/motion)
            if hasattr(self.data, 'qpos') and self._nq >= 3:
                sensors['body_position'] = list(self.data.qpos[:3])
            
            # Acceleration
            if hasattr(self.data, 'qacc') and len(self.data.qacc) >= 3:
                sensors['body_acceleration'] = list(self.data.qacc[:3])
            
            # Sensor data (if any named sensors)
            if hasattr(self.data, 'sensordata') and len(self.data.sensordata) > 0:
                sensors['sensor_data'] = list(self.data.sensordata)
            
        except Exception as e:
            print(f"  [WARNING] Sensor read error: {e}")
        
        return sensors
    
    def update_sensory(self, sensors: Dict[str, Any]) -> int:
        """Update sensory models and inject spikes into NIRON engine.
        
        Args:
            sensors: Dict from read_sensors().
            
        Returns:
            Number of sensory spikes injected.
        """
        total_spikes = 0
        
        if not self.sensory_feedback_enabled:
            return 0
        
        # Vision — derived from body position/height (simple proxy)
        if self.enable_vision:
            body_pos = sensors.get('body_position', [0, 0, 0])
            # Height gives brightness proxy (closer to ground = darker)
            height = abs(body_pos[2]) if len(body_pos) > 2 else 0.3
            brightness = min(1.0, max(0.1, height * 3.0))
            
            self.injector.update_vision(
                scene_brightness=brightness,
                dt_ms=self.dt_brain_ms,
            )
        
        # Mechanosensory (proprioception + touch)
        if self.enable_proprioception or self.enable_touch:
            joint_angles = sensors.get('joint_angles', [])
            joint_velocities = sensors.get('joint_velocities', [])
            contact_forces = sensors.get('contact_forces', []) if self.enable_touch else None
            
            self.injector.update_mechanosensory(
                joint_angles=joint_angles if joint_angles else [0.0],
                joint_velocities=joint_velocities if joint_velocities else [0.0],
                contact_forces=contact_forces if contact_forces else None,
                dt_ms=self.dt_brain_ms,
            )
        
        # Chemosensory (disabled by default — no odor field)
        if self.enable_chemo:
            self.injector.update_chemosensory(
                odor_concentration=0.0,
                dt_ms=self.dt_brain_ms,
            )
        
        # Inject all sensory spikes
        total_spikes = self.injector.inject_spikes(self.dt_brain_ms)
        
        return total_spikes
    
    def read_motor_output(self) -> Set[int]:
        """Read which motor neurons fired in the current NIRON cycle.
        
        Returns:
            Set of neuron indices that fired.
        """
        fired: Set[int] = set()
        
        if self.engine is None or not self.engine.neurons:
            return fired
        
        # Check fire_list_2 for fired neurons
        for word_idx, word_val in enumerate(self.engine._fire_list_2):
            if word_val == 0:
                continue
            
            base_id = word_idx * 64
            remaining = word_val
            while remaining:
                bit_pos = (remaining & -remaining).bit_length() - 1
                neuron_id = base_id + bit_pos
                if neuron_id < len(self.engine.neurons):
                    fired.add(neuron_id)
                remaining &= remaining - 1
        
        return fired
    
    def apply_motor_commands(self, joint_commands: Dict[str, float]) -> None:
        """Apply joint commands to MuJoCo control array.
        
        Args:
            joint_commands: Dict mapping joint_name → ctrl_value [-1, 1].
        """
        if hasattr(self.data, 'ctrl'):
            for joint_name, value in joint_commands.items():
                if joint_name in self.actuator_map:
                    ctrl_idx = self.actuator_map[joint_name]
                    if 0 <= ctrl_idx < len(self.data.ctrl):
                        # Clamp to control range
                        clamped = max(-1.0, min(1.0, value))
                        self.data.ctrl[ctrl_idx] = clamped
    
    def step(self) -> Dict[str, Any]:
        """Execute one full closed-loop step.
        
        One step = brain_steps_per_physics brain cycles followed by
        one physics step. At 1kHz brain / 200Hz physics, that's 5 brain
        steps per physics step.
        
        Returns:
            Dict with step metrics.
        """
        t_start = time.perf_counter()
        
        # Read sensors (once per physics step)
        sensors = self.read_sensors()
        
        # Brain sub-steps (multiple brain cycles per physics step)
        total_fired = 0
        total_sensory_spikes = 0
        
        for _ in range(self.brain_steps_per_physics):
            # Update sensory models and inject spikes
            sensory_spikes = self.update_sensory(sensors)
            total_sensory_spikes += sensory_spikes
            
            # Run NIRON fire cycle
            fired_count, cycle = self.engine.fire()
            total_fired += fired_count
            
            # Read motor output from fired neurons
            fired_motor_neurons = self.read_motor_output()
            
            # Phase 9: DN->MN Bridge — translate if bridge present
            if self.bridge is not None:
                bt = self.bridge.translate(fired_motor_neurons, dt=0.001)
                if bt is not None:
                    self.bridge_torque_buffer = bt
            else:
                # Accumulate in VNC decoder
                if self.vnc_decoder is not None:
                    self.vnc_decoder.accumulate(fired_motor_neurons)
            
            self.sim_time_ms += self.dt_brain_ms
        
        # Decode accumulated motor commands
        # Phase 9: Prefer bridge torque buffer if available
        if self.bridge is not None and self.bridge_torque_buffer is not None:
            if hasattr(self.data, 'ctrl') and len(self.data.ctrl) == len(self.bridge_torque_buffer):
                self.data.ctrl[:] = self.bridge_torque_buffer
                joint_commands = {}  # bridge handles it
            self.bridge_torque_buffer = None  # clear for next step
        else:
            joint_commands: Dict[str, float] = {}
            if self.vnc_decoder is not None:
                joint_commands = self.vnc_decoder.decode()
            
            # Apply motor commands to MuJoCo
            self.apply_motor_commands(joint_commands)
        
        # Step physics
        try:
            import mujoco
            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            print(f"  [WARNING] Physics step error: {e}")
        
        # Update step count
        self.step_count += 1
        
        # Timing
        t_end = time.perf_counter()
        step_duration_ms = (t_end - t_start) * 1000.0
        self._step_start_times.append(step_duration_ms)
        
        # Record metrics
        self.metrics['step_duration_ms'].append(step_duration_ms)
        self.metrics['fired_neurons'].append(total_fired)
        self.metrics['sensory_spikes'].append(total_sensory_spikes)
        
        # Active joints
        active_joints = sum(1 for v in joint_commands.values() if abs(v) > 0.001)
        self.metrics['active_joints'].append(active_joints)
        
        # Track joint activity
        for jname, val in joint_commands.items():
            if jname not in self.metrics:
                self.metrics[f'joint_{jname}'] = []
            self.metrics[f'joint_{jname}'].append(val)
        
        return {
            'step': self.step_count,
            'time_ms': self.sim_time_ms,
            'fired_neurons': total_fired,
            'sensory_spikes': total_sensory_spikes,
            'active_joints': active_joints,
            'step_duration_ms': step_duration_ms,
        }
    
    def run(
        self,
        num_steps: int,
        render_every: int = 1,
        report_every: int = 100,
        render_callback=None,
    ) -> List[Dict[str, Any]]:
        """Run the closed-loop simulation for N steps.
        
        Args:
            num_steps: Number of physics steps to run.
            render_every: Render frame every N steps (1 = always).
            report_every: Report progress every N steps.
            render_callback: Optional callback(report) for each rendered step.
            
        Returns:
            List of step report dicts.
        """
        reports = []
        
        print(f"\n{'='*60}")
        print(f"CLOSED-LOOP SIMULATION")
        print(f"{'='*60}")
        print(f"Brain rate: {self.brain_rate_hz} Hz ({self.brain_steps_per_physics}x sub-steps)")
        print(f"Physics rate: {self.physics_rate_hz} Hz")
        print(f"Total steps: {num_steps} (~{num_steps * self.dt_physics_ms / 1000:.1f}s simulated)")
        print(f"Neurons: {self.engine.array_size:,}")
        print(f"Sensory feedback: {'ON' if self.sensory_feedback_enabled else 'OFF'}")
        print(f"W_syn: {self.w_syn}")
        print(f"{'='*60}\n")
        
        t_start = time.perf_counter()
        
        for i in range(num_steps):
            report = self.step()
            reports.append(report)
            
            # Render frame
            if (i + 1) % render_every == 0 and render_callback is not None:
                render_callback(report)
            
            # Progress report
            if (i + 1) % report_every == 0:
                elapsed = time.perf_counter() - t_start
                real_time_factor = (self.sim_time_ms / 1000.0) / elapsed if elapsed > 0 else 0
                
                avg_fired = sum(self.metrics['fired_neurons'][-report_every:]) / report_every
                avg_sensory = sum(self.metrics['sensory_spikes'][-report_every:]) / report_every
                avg_active = sum(self.metrics['active_joints'][-report_every:]) / report_every
                
                print(f"  Step {i+1}/{num_steps} | "
                      f"sim={self.sim_time_ms/1000:.1f}s | "
                      f"RTF={real_time_factor:.3f}x | "
                      f"fired={avg_fired:.0f} | "
                      f"sensory={avg_sensory:.0f} | "
                      f"active_joints={avg_active:.1f}")
        
        total_elapsed = time.perf_counter() - t_start
        avg_step_ms = sum(self.metrics['step_duration_ms']) / max(1, len(self.metrics['step_duration_ms']))
        
        print(f"\n--- Simulation Complete ---")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Simulated: {self.sim_time_ms/1000:.1f}s")
        print(f"Overall RTF: {self.sim_time_ms/1000 / total_elapsed:.4f}x")
        print(f"Avg step: {avg_step_ms:.1f}ms")
        
        return reports
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation run."""
        steps = len(self.metrics.get('fired_neurons', []))
        if steps == 0:
            return {'error': 'No steps run yet'}
        
        avg_fired = sum(self.metrics['fired_neurons']) / steps
        avg_sensory = sum(self.metrics['sensory_spikes']) / steps
        avg_active = sum(self.metrics['active_joints']) / steps
        avg_step = sum(self.metrics['step_duration_ms']) / steps
        
        return {
            'total_steps': steps,
            'simulated_time_ms': self.sim_time_ms,
            'simulated_time_s': self.sim_time_ms / 1000.0,
            'avg_fired_per_step': avg_fired,
            'avg_sensory_spikes': avg_sensory,
            'avg_active_joints': avg_active,
            'avg_step_duration_ms': avg_step,
            'brain_rate_hz': self.brain_rate_hz,
            'physics_rate_hz': self.physics_rate_hz,
            'neurons': self.engine.array_size,
            'w_syn': self.w_syn,
            'sensory_feedback': self.sensory_feedback_enabled,
        }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get NIRON engine performance statistics."""
        return self.engine.get_stats() if self.engine else {}
    
    def get_injector_stats(self) -> Dict[str, Any]:
        """Get sensory injector statistics."""
        return self.injector.get_stats() if self.injector else {}
    
    def get_vnc_stats(self) -> Dict[str, Any]:
        """Get VNC decoder statistics."""
        return self.vnc_decoder.summary() if self.vnc_decoder else {}
    
    def reset(self) -> None:
        """Reset all state."""
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics.clear()
        self._step_start_times.clear()
        if self.injector:
            self.injector.reset_stats()
        if self.vnc_decoder:
            self.vnc_decoder.reset()
    
    def __repr__(self) -> str:
        return (
            f"ClosedLoop(steps={self.step_count}, "
            f"time={self.sim_time_ms:.1f}ms, "
            f"neurons={self.engine.array_size if self.engine else 0}, "
            f"w_syn={self.w_syn}, "
            f"feedback={'ON' if self.sensory_feedback_enabled else 'OFF'})"
        )