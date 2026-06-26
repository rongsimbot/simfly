#!/usr/bin/env python3
"""
brian2_loop.py — Brian2 Brain → VNC Bridge → MuJoCo Closed Loop

Integrates:
  1. Brian2Runner (138K LIF neurons via Shiu et al. model)
  2. Brian2SensoryFeedback (MuJoCo state → Poisson sensory rates)
  3. DnMnBridge (DN spikes → MANC MN activations)
  4. VNCMotorDecoder (MN → joint actuator torques)

Architecture:
  MuJoCo state (qpos, contacts, vision)
  → Brian2SensoryFeedback.read_and_compute()
  → Brian2Runner.set_sensory_poisson(visual, chemo, mechano)
  → Brian2Runner.step_physics(dt_ms=5.0, n_brain_steps=5)
  → Brian2Runner.get_dn_spikes()
  → DnMnBridge.translate() with compatible loader
  → VNCMotorDecoder.decode()
  → MuJoCo data.ctrl[] = torque

Usage:
  loop = Brian2SimLoop(data_dir="eon-fly-brain/data/", ...)
  loop.initialize()
  loop.start()
  for step in range(N):
      report = loop.step()
      
SCIENTIFIC RIGOR: All spike data from real FlyWire connectome.
No fake or scripted neuron firing patterns.
"""

import json
import math
import os
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import traceback

import numpy as np

# Path setup
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
SENSORY_DIR = os.path.join(CODE_ROOT, "sensory")
BRIAN2_INTEGRATION = os.path.join(CODE_ROOT, "brian2_integration")
EON_DATA = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "eon-fly-brain", "data")

for d in [CODE_ROOT, SENSORY_DIR, BRIAN2_INTEGRATION]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from brian2 import Hz, ms, mV, second
from brian2_runner import Brian2Runner
from brian2_sensory_feedback import Brian2SensoryFeedback
from phase8_integration.dn_mn_bridge import DnMnBridge
from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
from vision import FlyVision
from chemo import ChemoSensorySystem
from mechano import MechanoSensorySystem
import mujoco


class Brian2BridgeLoader:
    """Minimal loader compatible with DnMnBridge.translate().
    
    The bridge needs an object with .idx_to_flywire to convert
    Brian2 neuron indices to FlyWire root IDs.
    """
    def __init__(self, idx_to_flywire):
        self.idx_to_flywire = []
        max_idx = max(idx_to_flywire.keys()) if idx_to_flywire else 0
        self.idx_to_flywire = [0] * (max_idx + 1)
        for b_idx, fw_id in idx_to_flywire.items():
            self.idx_to_flywire[b_idx] = fw_id


class Brian2SimLoop:
    """Full closed-loop: Brian2 brain drives MuJoCo body via VNC bridge."""
    
    def __init__(
        self,
        data_dir=None,
        dn_matches_path=None,
        pathways_path=None,
        vnc_actuator_map_path=None,
        simfly_xml=None,
        food_pos=(1.2, -0.15, 0.20),
        chemo_sigma=4.0,
        arena_bounds=None,
        global_gain=0.002,
        tau_decay=50.0,
        min_pathway_confidence=0.01,
        **kwargs,
    ):
        self.data_dir = data_dir or EON_DATA
        self.dn_matches_path = dn_matches_path or os.path.join(CODE_ROOT, "vnc_bridge", "dn_matches.json")
        self.pathways_path = pathways_path or os.path.join(CODE_ROOT, "vnc_bridge", "dn_mn_pathways.json")
        self.vnc_actuator_map_path = vnc_actuator_map_path or os.path.join(CODE_ROOT, "vnc_bridge", "vnc_actuator_map.json")
        self.simfly_xml = simfly_xml or os.path.join(HOME, "simrobotics-storage", "research", "flywire", "virtual-fly", "simfly_model", "simfly_grounded.xml")
        self.food_pos = food_pos
        self.chemo_sigma = chemo_sigma
        self.arena_bounds = arena_bounds or {'x_min': -10.0, 'x_max': 5.0, 'y_min': -5.0, 'y_max': 5.0, 'z_min': -1.0, 'z_max': 5.0}
        self.global_gain = global_gain
        self.tau_decay = tau_decay
        self.min_pathway_confidence = min_pathway_confidence
        self.synaptic_scale = kwargs.get("synaptic_scale", 5.0)
        
        self.runner = None
        self.sensory_feedback = None
        self.bridge = None
        self.decoder = None
        self.bridge_loader = None
        self.vision = None
        self.chemo = None
        self.mechano = None
        self.model = None
        self.data = None
        self.act_map = {}
        
        self._initialized = False
        self._running = False
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)
        self._firing_history = []
        self._init_error = None
        
        self._total_neurons = 0
        self._total_dns = 0

    def initialize(self):
        try:
            t0 = time.perf_counter()
            print(f"\n{'='*60}")
            print(f"Brian2 Simulation Loop - Initializing")
            print(f"{'='*60}")
            
            print(f"\n[1/5] Building Brian2 network (138K neurons)...", flush=True)
            self.runner = Brian2Runner(data_dir=self.data_dir, dn_matches_path=self.dn_matches_path, synaptic_scale=self.synaptic_scale)
            timings = self.runner.initialize(build_network=True)
            self._total_neurons = self.runner.n_neurons
            self._total_dns = len(self.runner.dn_idxs)
            print(f"  Brian2: {self._total_neurons:,} neurons, {self._total_dns:,} DNs", flush=True)
            
            print(f"\n[2/5] Loading MuJoCo model...", flush=True)
            self.model = mujoco.MjModel.from_xml_path(self.simfly_xml)
            self.data = mujoco.MjData(self.model)
            if self.model.nq >= 7:
                self.data.qpos[:7] = [0.0, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0]
            for _ in range(5000):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                self.act_map[name or "act_%d" % i] = i
            print(f"  MuJoCo: {self.model.nbody} bodies, {self.model.nu} actuators", flush=True)
            
            print(f"\n[3/5] Initializing sensory systems...", flush=True)
            self.sensory_feedback = Brian2SensoryFeedback(food_pos=self.food_pos, chemo_sigma=self.chemo_sigma, arena_bounds=self.arena_bounds)
            self.vision = FlyVision(self.model, self.data, num_rays=20, arena_bounds=self.arena_bounds, food_position=self.food_pos)
            self.chemo = ChemoSensorySystem(sugar_source_pos=self.food_pos, sugar_sigma=self.chemo_sigma)
            self.mechano = MechanoSensorySystem(self.model, self.data)
            print(f"  Sensory systems ready", flush=True)
            
            print(f"\n[4/5] Loading VNC bridge...", flush=True)
            self.decoder = VNCMotorDecoder.load_from_vnc(vnc_actuator_map_path=self.vnc_actuator_map_path, pathways_path=self.pathways_path, tau_decay=self.tau_decay, global_gain=self.global_gain, dt_brain_ms=1.0, dt_physics_ms=5.0)
            self.bridge_loader = Brian2BridgeLoader(self.runner.idx_to_flywire)
            self.bridge = DnMnBridge(dn_matches_path=self.dn_matches_path, pathways_path=self.pathways_path, vnc_decoder=self.decoder, model=self.model, idx_to_flywire=self.bridge_loader.idx_to_flywire, min_pathway_confidence=self.min_pathway_confidence, global_gain=self.global_gain)
            self.bridge.initialize()
            bs = self.bridge.summary()
            print(f"  Bridge: {bs.get('dn_matches_loaded', 0)} DNs to {bs.get('unique_mns_loaded', 0)} MNs", flush=True)
            
            total_elapsed = time.perf_counter() - t0
            print(f"\n{'='*60}")
            print(f"Brian2Loop ready in {total_elapsed:.1f}s")
            print(f"  Neurons: {self._total_neurons:,} | DNs: {self._total_dns}")
            print(f"  Actuators: {self.model.nu}")
            print(f"{'='*60}\n", flush=True)
            
            self._initialized = True
            return True
        except Exception as e:
            self._init_error = "%s\n%s" % (e, traceback.format_exc())
            print(f"\nBRIAN2 INIT FAILED: {self._init_error}", flush=True)
            return False

    def step(self):
        if not self._initialized:
            return {'step': 0, 'error': 'Not initialized'}
        
        dt = 5.0
        vision_data = self.vision.read(dt_ms=dt, sim_time_ms=self.sim_time_ms)
        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=dt)
        mechano_data = self.mechano.read()
        
        sensory_rates = self.sensory_feedback.read_and_compute(
            qpos=self.data.qpos, qvel=getattr(self.data, 'qvel', None),
            contacts=getattr(self.data, 'contact', None),
            sim_time_ms=self.sim_time_ms,
            vision_data=vision_data, chemo_data=chemo_data, mechano_data=mechano_data,
        )
        
        sensory_count = self.runner.set_sensory_poisson(
            visual_rate=sensory_rates['visual'],
            chemo_rate=sensory_rates['chemo'],
            mechano_rate=sensory_rates['mechano'],
            hemisphere_bias=sensory_rates.get('hemisphere_bias', 0.0),
        )
        
        brian2_result = self.runner.step_physics(dt_ms=dt, n_brain_steps=5)
        dn_spikes = self.runner.get_dn_spikes(window_ms=dt)
        fired_dn_indices = set(dn_spikes.keys())
        
        mn_activations = {}
        bridge_report = {"dns": 0, "mns": 0}
        if self.bridge and fired_dn_indices:
            mn_activations = self.bridge.translate(fired_dn_indices, self.bridge_loader)
            bridge_report = {"dns": len(self.bridge.last_fired_dns), "mns": len(self.bridge.last_activated_mns)}
        
        joint_commands = self.decoder.decode() if self.decoder else {}
        self.data.ctrl[:] = 0.0
        nz = 0
        for jname, torque in joint_commands.items():
            if jname in self.act_map:
                self.data.ctrl[self.act_map[jname]] = float(np.clip(torque, -1.0, 1.0))
                if abs(torque) > 0.001:
                    nz += 1
        
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            pass
        
        self.step_count += 1
        self.sim_time_ms += dt
        
        dn_fw_ids = [self.runner.idx_to_flywire.get(idx, 0) for idx in fired_dn_indices]
        mn_ids = list(self.bridge.last_activated_mns) if self.bridge else []
        self._firing_history.append({
            'step': self.step_count, 'fired_dns': dn_fw_ids[:50],
            'mns': mn_ids[:50],
            'joints': {jname: round(float(torque), 6) for jname, torque in list(joint_commands.items())[:36]},
        })
        if len(self._firing_history) > 200:
            self._firing_history.pop(0)
        
        self.metrics['dns'].append(bridge_report['dns'])
        self.metrics['mns'].append(bridge_report['mns'])
        self.metrics['active_joints'].append(nz)
        self.metrics['torque'].append(nz > 0)
        food_dist = float(np.linalg.norm(np.array(self.data.qpos[:2]) - np.array(self.food_pos[:2])))
        self.metrics['food_dist'].append(food_dist)
        
        return {
            'step': self.step_count, 'time_ms': self.sim_time_ms,
            'total_spikes': brian2_result.get('total_spikes', 0),
            'dn_spikes': brian2_result.get('dn_spikes', 0),
            'dn_matches': bridge_report['dns'], 'mns_activated': bridge_report['mns'],
            'active_joints': nz, 'torque_applied': nz > 0,
            'z_height': float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0,
            'on_ground': mechano_data.get('is_on_ground', False),
            'contrast': vision_data.get('contrast', 0.0),
            'food_brightness': vision_data.get('food_brightness', 0.0),
            'has_food_visual': vision_data.get('has_food_visual', False),
            'wall_distance': vision_data.get('wall_distance', 10.0),
            'sugar_conc': chemo_data.get('sugar_concentration', 0.0),
            'food_distance': round(food_dist, 3),
            'looming': vision_data.get('looming_intensity', 0.0),
            'sensory_visual': sensory_rates.get('visual', 0.0),
            'sensory_chemo': sensory_rates.get('chemo', 0.0),
            'sensory_mechano': sensory_rates.get('mechano', 0.0),
            'sensory_neurons_set': sensory_count,
        }

    def stop(self):
        """Stop simulation and clean up Brian2 Network objects (ITEM 3 FIX)."""
        self._running = False
        if self.runner is not None:
            try:
                if self.runner.net is not None:
                    self.runner.net.stop()
                for obj in getattr(self.runner, '_sensory_network_objects', []):
                    try:
                        if self.runner.net is not None:
                            self.runner.net.remove(obj)
                    except Exception:
                        pass
                self.runner._sensory_network_objects.clear()
                self.runner.net = None
                self.runner.neu = None
                self.runner.syn = None
                self.runner.spk_mon = None
                self.runner._sensory_poisson.clear()
                self.runner._sensory_poisson_dict.clear()
                self.runner._initialized = False
                self.runner._sim_time_sec = 0.0
                print("[Brian2Loop] Runner stopped, Network objects cleaned up", flush=True)
            except Exception as e:
                print("[Brian2Loop] Cleanup error (non-fatal): %s" % e, flush=True)
        self._initialized = False
        self.runner = None
        self.sensory_feedback = None
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)
        self._firing_history.clear()

    def reset(self):
        if self.runner is not None:
            self.runner.reset()
            if self.sensory_feedback:
                self.sensory_feedback.reset()
            if self.data is not None and len(self.data.qpos) >= 7:
                self.data.qpos[:7] = [0.0, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0]
            self.step_count = 0
            self.sim_time_ms = 0.0
            self.metrics = defaultdict(list)
            self._firing_history.clear()

    @property
    def is_initialized(self):
        return self._initialized and self.runner is not None

    @property
    def status(self):
        return {
            'initialized': self._initialized, 'running': self._running,
            'neurons': self._total_neurons, 'dns': self._total_dns,
            'step': self.step_count, 'error': self._init_error,
        }
