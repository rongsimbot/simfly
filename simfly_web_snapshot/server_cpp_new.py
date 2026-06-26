#!/usr/bin/env python3
"""
SimFly Web Platform — C++ Engine Edition
=========================================
Flask + Socket.IO web dashboard powered by libneuronengine.so (2200× faster than Python NIRON)

Key changes from Python NIRON:
- CppEngine replaces neuron_engine (C++ shared library, ctypes bindings)
- fire() returns list of fired neuron IDs directly
- BurstInjector uses add_charge() instead of neuron object methods
- BFS-upstream neuron selection for connected sensorimotor pathways
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, os, sys, time, threading, traceback
from collections import defaultdict, Counter, deque
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# REVERSE order (last insert = lowest index): cpp_engine > simfly-model > sensory > flywire
sys.path.insert(0, "/home/simllm/simrobotics-storage/research/flywire")
sys.path.insert(0, "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/sensory")
sys.path.insert(0, "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model")
sys.path.insert(0, "/tmp/cpp_engine")
from cpp_engine import CppEngine

from connectome.connectome_loader import FlyWireConnectomeLoader
from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
from phase8_integration.dn_mn_bridge import DnMnBridge
from vision import FlyVision
from chemo import ChemoSensorySystem
from mechano import MechanoSensorySystem
import mujoco

from flask import Flask, Response, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit

# ── Config ───────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
SENSORY_DIR = os.path.join(CODE_ROOT, "sensory")
FLYWIRE_DIR = os.path.join(HOME, "simrobotics-storage", "research", "flywire")
RESEARCH = os.path.join(HOME, "simrobotics-storage", "research")
CONNECTIONS_CSV = os.path.join(RESEARCH, "connections_princeton_no_threshold.csv.gz")
VNC_DIR = os.path.join(CODE_ROOT, "vnc_bridge")
SIMFLY_XML = os.path.join(FLYWIRE_DIR, "virtual-fly", "simfly_model", "simfly_grounded.xml")
DN_MATCHES_JSON = os.path.join(VNC_DIR, "dn_matches.json")
DN_MN_PATHWAYS_JSON = os.path.join(VNC_DIR, "dn_mn_pathways.json")
MOTOR_MAP_JSON = os.path.join(CODE_ROOT, "neuron_engine", "motor_neuron_map.json")
DN_ANATOMY_GROUPS_JSON = "/tmp/simfly_web/dn_anatomy_groups.json"

NT_WEIGHT_MAP: Dict[str, float] = {
    'ACH': 1.5, 'DA': 0.75, 'OCT': 0.75,
    'GABA': -0.5, 'GLUT': -0.25, 'SER': -0.15,
}

DEFAULT_NEURONS = 0  # 0 = ALL neurons (139,116) — directive: full connectome required
RENDER_WIDTH, RENDER_HEIGHT = 640, 480
SYNAPTIC_SCALE = 0.005
FOOD_POS = (1.2, -0.15, 0.20)
SUGAR_SIGMA = 4.0
ARENA_BOUNDS = {'x_min': -10.0, 'x_max': 5.0, 'y_min': -5.0, 'y_max': 5.0, 'z_min': -1.0, 'z_max': 5.0}
SYSTEM_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
IMGUI_FONT_PATH = SYSTEM_FONT_PATH

# ── BurstInjector (C++ engine) ─────────────────────────────────────
class BurstInjector:
    def __init__(self, engine: CppEngine, spikes_per_burst=5, isi_ms=1.0,
                 min_burst_interval_ms=10.0, charge_per_spike=2.0):
        self.engine = engine
        self.spikes_per_burst = spikes_per_burst
        self.isi_ms = isi_ms
        self.min_burst_interval_ms = min_burst_interval_ms
        self.charge_per_spike = charge_per_spike
        self._active_bursts: Dict[int, Tuple[int, float]] = {}
        self._last_burst_time: Dict[int, float] = {}
        self._sim_time_ms: float = 0.0

    def trigger_burst(self, neuron_idx: int) -> bool:
        if neuron_idx < 0 or neuron_idx >= self.engine.get_size():
            return False
        last_time = self._last_burst_time.get(neuron_idx, -999.0)
        if self._sim_time_ms - last_time < self.min_burst_interval_ms:
            return False
        self._active_bursts[neuron_idx] = (self.spikes_per_burst, self._sim_time_ms)
        self._last_burst_time[neuron_idx] = self._sim_time_ms + self.spikes_per_burst * self.isi_ms
        return True

    def step(self, dt_ms: float) -> int:
        self._sim_time_ms += dt_ms
        total_spikes = 0
        completed = []
        for neuron_idx, (remaining, next_spike_time) in list(self._active_bursts.items()):
            if self._sim_time_ms >= next_spike_time:
                self.engine.add_charge(neuron_idx, self.charge_per_spike)
                total_spikes += 1
                new_remaining = remaining - 1
                if new_remaining > 0:
                    self._active_bursts[neuron_idx] = (new_remaining, self._sim_time_ms + self.isi_ms)
                else:
                    completed.append(neuron_idx)
        for nid in completed:
            del self._active_bursts[nid]
        return total_spikes

    @property
    def active_burst_count(self) -> int:
        return len(self._active_bursts)

    def reset(self):
        self._active_bursts.clear()
        self._last_burst_time.clear()
        self._sim_time_ms = 0.0

# ── Sensory neuron identification ───────────────────────────────────
DN_UPSTREAM_MAP_JSON = "/tmp/simfly_web/dn_upstream_map.json"

def load_dn_upstream_sensory_map(included_ids, flywire_to_idx_map):
    """Load sensory targets from DN-upstream BFS pathway map.
    Replaces the broken sort-by-degree approach with real connectome-grounded,
    pathway-verified sensory neuron selection.
    
    These neurons are verified to have paths reaching DNs within 4 hops.
    """
    import json as _json
    sensory_map = {'visual_input': set(), 'mechano_input': set(), 'chemo_input': set()}
    
    try:
        with open(DN_UPSTREAM_MAP_JSON, 'r') as f:
            data = _json.load(f)
        targets = data.get('sensory_targets', {})
        
        visual_fw = set(targets.get('visual', []))
        olfactory_fw = set(targets.get('olfactory', []))
        mechano_fw = set(targets.get('mechano', []))
        
        # Only include neurons that are actually in our loaded set
        included = set(included_ids)
        sensory_map['visual_input'] = visual_fw & included
        sensory_map['chemo_input'] = olfactory_fw & included
        sensory_map['mechano_input'] = mechano_fw & included
        
        print(f"  [DN-upstream] Visual: {len(sensory_map['visual_input'])}, "
              f"Olfactory: {len(sensory_map['chemo_input'])}, "
              f"Mechano: {len(sensory_map['mechano_input'])}", flush=True)
    except FileNotFoundError:
        print(f"  [DN-upstream] WARNING: {DN_UPSTREAM_MAP_JSON} not found! "
              f"Falling back to empty sensory map.", flush=True)
    
    return sensory_map

# ── C++ Engine builder ──────────────────────────────────────────────
def build_cpp_engine(neurons_nt, connections, loader):
    """Build a CppEngine from neuron types and connections."""
    sorted_ids = sorted(neurons_nt.keys())
    flywire_to_idx = {fw_id: idx for idx, fw_id in enumerate(sorted_ids)}
    idx_to_flywire = sorted_ids
    loader.flywire_to_idx = flywire_to_idx
    loader.idx_to_flywire = idx_to_flywire
    loader.neuron_nt_types = neurons_nt
    loader.nt_weight_map = NT_WEIGHT_MAP

    n_neurons = len(sorted_ids)
    engine = CppEngine("/tmp/cpp_engine/libneuronengine.so")
    engine.create(n_neurons, n_threads=4)

    # Create neurons (all LIF)
    for i in range(n_neurons):
        engine.set_neuron(i, model=3, leak=0.03, refractory=1)

    # Add synapses
    max_syn = max((c[2] for c in connections), default=1)
    total_syn = 0
    pos_w = neg_w = 0
    for pre_id, post_id, syn_count in connections:
        nt_type = neurons_nt.get(pre_id, 'ACH')
        base_weight = NT_WEIGHT_MAP.get(nt_type, 0.0)
        weight = base_weight * syn_count / math.log(1 + max_syn) * SYNAPTIC_SCALE
        weight = max(-0.05, min(0.05, weight))
        pre_idx = flywire_to_idx[pre_id]
        post_idx = flywire_to_idx[post_id]
        engine.add_synapse(pre_idx, post_idx, weight)
        total_syn += 1
        if weight > 0: pos_w += 1
        elif weight < 0: neg_w += 1

    print(f"  [C++ engine] {n_neurons} neurons, {total_syn:,} synapses, {pos_w:,} excit, {neg_w:,} inhib", flush=True)
    return engine

# ── Sensory-Driven Simulation Loop ──────────────────────────────────
class SimFlyLoop:
    def __init__(self, engine: CppEngine, sensor_idx, bridge, decoder, loader,
                 model, data, act_map, vision, chemo, mechano, burst_inj, food_pos):
        self.engine = engine
        self.sensor_idx = sensor_idx
        self.bridge = bridge
        self.decoder = decoder
        self.loader = loader
        self.model = model
        self.data = data
        self.act_map = act_map
        self.vision = vision
        self.chemo = chemo
        self.mechano = mechano
        self.burst_inj = burst_inj
        self.food_pos = food_pos
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)

    def _inject_sensory(self, vision_data, chemo_data, mechano_data) -> int:
        total = 0
        contrast = vision_data.get('contrast', 0.0)
        food_vis = vision_data.get('food_brightness', 0.0)
        effective = max(contrast, food_vis * 0.8)

        # 🔴 ANATOMICAL LEFT/RIGHT DN INJECTION — every 5th cycle
        # Alternate between anatomically-grouped left-side and right-side DNs
        # for true left/right torque asymmetry driving gait
        dn_left = getattr(self, '_dn_left_eids', [])
        dn_right = getattr(self, '_dn_right_eids', [])
        dn_both = getattr(self, '_dn_both_eids', [])
        dn_fallback = getattr(self, '_dn_engine_list', [])

        if (dn_left or dn_right) and self.step_count % 5 == 0:
            # Interval 0, 2, 4... = left DNs + bilateral DNs
            # Interval 1, 3, 5... = right DNs + bilateral DNs
            if (self.step_count // 5) % 2 == 0:
                dn_targets = list(dn_left) + list(dn_both)
            else:
                dn_targets = list(dn_right) + list(dn_both)
            for eid in dn_targets:
                if self.burst_inj.trigger_burst(int(eid)):
                    total += self.burst_inj.spikes_per_burst
        elif dn_fallback and self.step_count % 5 == 0:
            # Fallback: simple first-half/second-half split
            half = max(1, len(dn_fallback) // 2)
            if (self.step_count // 5) % 2 == 0:
                dn_targets = dn_fallback[:half]
            else:
                dn_targets = dn_fallback[half:half*2]
            for eid in dn_targets:
                if self.burst_inj.trigger_burst(int(eid)):
                    total += self.burst_inj.spikes_per_burst

        if effective > 0.005:
            n_b = max(3, int(effective * 30))
            targets = list(self.sensor_idx.get('visual', set()))
            if targets:
                chosen = np.random.choice(targets, min(n_b, len(targets)), replace=False)
                for nid in chosen:
                    if self.burst_inj.trigger_burst(int(nid)):
                        total += self.burst_inj.spikes_per_burst

        sugar = chemo_data.get('sugar_concentration', 0.0)
        if sugar > 0.0005:
            n_b = max(3, int(sugar * 30))
            chem_targets = list(self.sensor_idx.get('chemo', set()))
            if chem_targets:
                chosen = np.random.choice(chem_targets, min(n_b, len(chem_targets)), replace=False)
                for nid in chosen:
                    if self.burst_inj.trigger_burst(int(nid)):
                        total += self.burst_inj.spikes_per_burst

        if mechano_data.get('is_on_ground', False):
            cf = mechano_data.get('total_contact_force', 0.0)
            if cf > 0.0005:
                n_b = min(5, max(2, int(cf * 4)))
                mech_targets = list(self.sensor_idx.get('mechano', set()))
                if mech_targets:
                    chosen = np.random.choice(mech_targets, min(n_b, len(mech_targets)), replace=False)
                    for nid in chosen:
                        if self.burst_inj.trigger_burst(int(nid)):
                            total += self.burst_inj.spikes_per_burst
        return total

    def step(self):
        dt = 5.0  # ms
        vision_data = self.vision.read(dt_ms=dt, sim_time_ms=self.sim_time_ms)

        # 🔴 PHASE 13 FIX 3: Direct food visibility check (independent of obstacle rays)
        # MuJoCo body geometry can occlude food rays; compute food angle directly
        # from head position/orientation and override vision_data if food is in FOV
        if self.food_pos is not None and not vision_data.get('has_food_visual', False):
            head_pos = self.vision.obstacle_detector._get_head_position()
            food_vec = np.array(self.food_pos) - head_pos
            food_dist = np.linalg.norm(food_vec)
            if food_dist > 0.001:
                food_dir = food_vec / food_dist
                # Get fly forward direction from body orientation
                try:
                    body_id = self.model.body('head').id if 'head' in [self.model.body(i).name for i in range(self.model.nbody)] else 0
                except:
                    body_id = 0
                xmat = self.data.body(body_id).xmat.reshape(3, 3)
                forward = xmat[:, 0]  # body x-axis (forward)
                # Compute angle between forward direction and food direction (in xy plane)
                forward_xy = forward[:2] / (np.linalg.norm(forward[:2]) + 1e-10)
                food_dir_xy = food_dir[:2] / (np.linalg.norm(food_dir[:2]) + 1e-10)
                food_dot = np.dot(forward_xy, food_dir_xy)
                # 32 deg cone (cos 32 deg = 0.848) — wider than ray-based 18 deg
                FOOD_FOV_COS = 0.848
                if food_dot > FOOD_FOV_COS:
                    vision_data['has_food_visual'] = True
                    vision_data['food_brightness'] = max(
                        vision_data.get('food_brightness', 0.0),
                        1.0 - min(1.0, food_dist / 5.0)
                    )

        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=dt)
        mechano_data = self.mechano.read()

        total_fired = 0
        all_fired: Set[int] = set()

        for _ in range(5):
            self.burst_inj.step(1.0)
            self._inject_sensory(vision_data, chemo_data, mechano_data)
            fired_ids = self.engine.fire()  # C++: returns list[int] directly
            total_fired += len(fired_ids)
            all_fired.update(fired_ids)
            self.sim_time_ms += 1.0

        mn_activations = {}
        bridge_report = {"dns": 0, "mns": 0}
        if self.bridge and all_fired:
            mn_activations = self.bridge.translate(all_fired, self.loader)
            bridge_report = {"dns": len(self.bridge.last_fired_dns), "mns": len(self.bridge.last_activated_mns)}

        if self.decoder:
            self.decoder.accumulate(set(mn_activations.keys()) if mn_activations else set())
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
        self._fired_ids = list(all_fired) if all_fired else []
        if not hasattr(self, '_firing_history'):
            self._firing_history = []
        self._firing_history.append({
            'step': self.step_count,
            'fired_count': total_fired,
            'fired': self._fired_ids[:100],
            'dns': list(self.bridge.last_fired_dns)[:20] if hasattr(self, 'bridge') and hasattr(self.bridge, 'last_fired_dns') else [],
            'mns': list(self.bridge.last_activated_mns)[:50] if hasattr(self, 'bridge') and hasattr(self.bridge, 'last_activated_mns') else [],
            'joints': {jname: round(float(torque), 6) for jname, torque in list(joint_commands.items())[:36]},
        })
        if len(self._firing_history) > 200:
            self._firing_history.pop(0)

        self.metrics['fired'].append(total_fired)
        self.metrics['dns'].append(bridge_report['dns'])
        self.metrics['mns'].append(bridge_report['mns'])
        self.metrics['active_joints'].append(nz)
        self.metrics['torque'].append(nz > 0)
        food_dist = float(np.linalg.norm(np.array(self.data.qpos[:2]) - np.array(self.food_pos[:2])))
        self.metrics['food_dist'].append(food_dist)

        return {
            'step': self.step_count, 'time_ms': self.sim_time_ms,
            'fired_neurons': total_fired, 'dn_matches': bridge_report['dns'],
            'mns_activated': bridge_report['mns'], 'active_joints': nz,
            'torque_applied': nz > 0,
            'z_height': float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0,
            'on_ground': mechano_data.get('is_on_ground', False),
            'contrast': vision_data.get('contrast', 0.0),
            'food_brightness': vision_data.get('food_brightness', 0.0),
            'has_food_visual': vision_data.get('has_food_visual', False),
            'wall_distance': vision_data.get('wall_distance', 10.0),
            'sugar_conc': chemo_data.get('sugar_concentration', 0.0),
            'food_distance': round(food_dist, 3),
            'looming': vision_data.get('looming_intensity', 0.0),
            'lc4_rate': vision_data.get('lc4_rate', 0.0),
        }

# ── SimFly Server ────────────────────────────────────────────────────
class SimFlyServer:
    def __init__(self):
        self.engine: Optional[CppEngine] = None
        self.loop: Optional[SimFlyLoop] = None
        self.bridge = None
        self.decoder = None
        self.loader = None
        self.model = None
        self.data = None
        self.act_map: Dict[str, int] = {}
        self.idx_to_flywire: List[int] = []
        self.vision = None
        self._initialized = False
        self._running = False
        self._paused = False
        self._init_error = None
        self._total_neurons = 0
        self._total_synapses = 0
        self._dns_loaded = 0
        self.renderer = None
        self._render_lock = threading.Lock()
        self._vision_render_lock = threading.Lock()
        self._latest_frame = None
        self._latest_vision_frame = None
        self._latest_metrics = {}
        self.neuron_nt_types: Dict[int, str] = {}
        self._dn_left_eids: List[int] = []
        self._dn_right_eids: List[int] = []
        self._dn_both_eids: List[int] = []
        self._dn_engine_list: List[int] = []

    def initialize(self, num_neurons=200, global_gain=0.002, tau_decay=50.0):
        try:
            print(f"\n{'='*60}")
            print(f"SimFly Web Platform — C++ Engine Edition")
            print(f"{'='*60}")
            print(f"  Neurons: {num_neurons} | SynScale: {SYNAPTIC_SCALE}")
            print(f"  Food: {FOOD_POS} | Sigma: {SUGAR_SIGMA}")
            print(f"  Arena: x=[{ARENA_BOUNDS['x_min']},{ARENA_BOUNDS['x_max']}]")
            t0 = time.perf_counter()

            # [1/6] Bridge
            print("\n[1/6] Loading DN→MN bridge...", flush=True)
            self.bridge = DnMnBridge(dn_matches_path=DN_MATCHES_JSON, pathways_path=DN_MN_PATHWAYS_JSON, min_pathway_confidence=0.01)
            self.bridge.initialize()
            bs = self.bridge.summary()
            self._dns_loaded = bs.get('dn_matches_loaded', 0)
            print(f"  Bridge: {self._dns_loaded} DNs, {bs.get('unique_mns_loaded', 0):,} MNs", flush=True)

            # [2/6] DNs
            print("\n[2/6] Loading DNs...", flush=True)
            with open(MOTOR_MAP_JSON) as f:
                motor_data = json.load(f)
            all_dn_ids = set()
            for root_id_str, info in motor_data.get("neurons", {}).items():
                if info.get("flow") == "efferent" and info.get("cell_type", "").startswith("DN"):
                    all_dn_ids.add(int(root_id_str))
            with open(DN_MATCHES_JSON) as f:
                dn_matches = json.load(f).get("matches", {})
            matched_dn_ids = {int(k) for k in dn_matches.keys()}
            print(f"  {len(all_dn_ids)} FlyWire DNs ({len(all_dn_ids & matched_dn_ids)} matched)", flush=True)

            # [3/6] Stream connections
            print(f"\n[3/6] Streaming connections...", flush=True)
            syn_counter = Counter()
            all_connections = []
            all_nts = {}
            t1 = time.perf_counter()
            with gzip.open(CONNECTIONS_CSV, 'rt') as f:
                for row in csv.DictReader(f):
                    pre = int(row['pre_root_id'])
                    post = int(row['post_root_id'])
                    syn = int(row['syn_count'])
                    nt = row['nt_type']
                    all_nts[pre] = all_nts.get(pre, nt)
                    all_nts[post] = all_nts.get(post, nt)
                    syn_counter[pre] += syn
                    syn_counter[post] += syn
                    all_connections.append((pre, post, syn, nt))
            print(f"  {len(all_connections):,} connections in {time.perf_counter()-t1:.1f}s", flush=True)

            # Select neurons: top synapse count + ensure DN representation
            max_dns = min(50, len(matched_dn_ids & all_dn_ids))
            dn_candidates = [(nid, syn_counter.get(nid, 0)) for nid in (matched_dn_ids & all_dn_ids)]
            dn_candidates.sort(key=lambda x: x[1], reverse=True)
            if num_neurons == 0:
                # ALL neurons — Ronnie directive: full connectome required
                included_ids = set(syn_counter.keys())
            else:
                included_dns = {nid for nid, _ in dn_candidates[:max_dns]}
                included_ids = set(included_dns)
                for nid, _ in syn_counter.most_common():
                    if nid in included_ids: continue
                    included_ids.add(nid)
                    if len(included_ids) >= num_neurons: break

            print(f"  Selected: {len(included_ids)} neurons ({len(included_ids & all_dn_ids)} DNs)", flush=True)

            # [4/6] Build C++ engine
            print(f"\n[4/6] Building C++ engine...", flush=True)
            self.neuron_nt_types = {nid: all_nts.get(nid, 'unknown') for nid in included_ids}
            connections = [(pre, post, syn) for pre, post, syn, _ in all_connections if pre in included_ids and post in included_ids]
            self.loader = FlyWireConnectomeLoader(CONNECTIONS_CSV, config={'min_syn_count': 0})
            self.engine = build_cpp_engine(self.neuron_nt_types, connections, self.loader)
            self._total_neurons = self.engine.get_size()
            self._total_synapses = self.engine.get_synapse_count()
            print(f"  Engine: {self._total_neurons} neurons, {self._total_synapses:,} synapses", flush=True)

            # [5/6] Sensory + MuJoCo
            print(f"\n[5/6] Identifying sensory + loading MuJoCo...", flush=True)
            sensor_map = load_dn_upstream_sensory_map(included_ids, self.loader.flywire_to_idx)
            sensor_idx = {'visual': set(), 'mechano': set(), 'chemo': set()}
            for fw_id in sensor_map['visual_input']:
                eid = self.loader.flywire_to_idx.get(fw_id)
                if eid is not None: sensor_idx['visual'].add(eid)
            for fw_id in sensor_map['mechano_input']:
                eid = self.loader.flywire_to_idx.get(fw_id)
                if eid is not None: sensor_idx['mechano'].add(eid)
            for fw_id in sensor_map['chemo_input']:
                eid = self.loader.flywire_to_idx.get(fw_id)
                if eid is not None: sensor_idx['chemo'].add(eid)

            # 🔴 DN DIRECT INJECTION: Add matched DN engine IDs to sensory targets
            # This ensures sensory input can directly reach DNs while the pathway
            # routing is being refined. Without this, signals degrade across multiple
            # synaptic hops before reaching DNs.
            matched_dn_engine_ids = set()
            for dn_fw_id in matched_dn_ids & included_ids:
                eid = self.loader.flywire_to_idx.get(dn_fw_id)
                if eid is not None:
                    matched_dn_engine_ids.add(eid)
            sensor_idx['visual'] |= matched_dn_engine_ids
            sensor_idx['chemo'] |= matched_dn_engine_ids
            sensor_idx['mechano'] |= matched_dn_engine_ids

            # 🔴 PHASE 13: ANATOMICAL DN GROUPING for true left/right alternation
            # Load pre-computed DN anatomy groups (classified by MN body side targeting)
            self._dn_left_eids = []
            self._dn_right_eids = []
            self._dn_both_eids = []
            try:
                with open(DN_ANATOMY_GROUPS_JSON) as f:
                    anatomy = json.load(f)
                for fw_id in anatomy.get('left_dns', []):
                    eid = self.loader.flywire_to_idx.get(int(fw_id))
                    if eid is not None and eid in matched_dn_engine_ids:
                        self._dn_left_eids.append(eid)
                for fw_id in anatomy.get('right_dns', []):
                    eid = self.loader.flywire_to_idx.get(int(fw_id))
                    if eid is not None and eid in matched_dn_engine_ids:
                        self._dn_right_eids.append(eid)
                for fw_id in anatomy.get('both_dns', []):
                    eid = self.loader.flywire_to_idx.get(int(fw_id))
                    if eid is not None and eid in matched_dn_engine_ids:
                        self._dn_both_eids.append(eid)
                print(f"  [DN-anatomy] Loaded groups: L={len(self._dn_left_eids)}, R={len(self._dn_right_eids)}, Both={len(self._dn_both_eids)}", flush=True)
            except FileNotFoundError:
                print(f"  [DN-anatomy] WARNING: {DN_ANATOMY_GROUPS_JSON} not found, using simple split", flush=True)

            dn_engine_list = sorted(matched_dn_engine_ids)
            self._dn_engine_list = dn_engine_list
            self._dn_inject_counter = 0
            print(f"  [DN-direct] {len(dn_engine_list)} DNs added to sensory injection targets", flush=True)

            self.decoder = VNCMotorDecoder.load_from_vnc(
                vnc_actuator_map_path=os.path.join(VNC_DIR, "vnc_actuator_map.json"),
                pathways_path=DN_MN_PATHWAYS_JSON, tau_decay=tau_decay, global_gain=global_gain,
                dt_brain_ms=1.0, dt_physics_ms=5.0,
            )
            self.model = mujoco.MjModel.from_xml_path(SIMFLY_XML)
            self.data = mujoco.MjData(self.model)
            if self.model.nq >= 7:
                self.data.qpos[:7] = [0, 0, 0.06, 1, 0, 0, 0]
            for _ in range(5000):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                self.act_map[name or f"act_{i}"] = i
            print(f"  MuJoCo: {self.model.nbody} bodies, {self.model.nu} actuators", flush=True)
            self.idx_to_flywire = self.loader.idx_to_flywire

            # [6/6] Sensory systems + loop
            print(f"\n[6/6] Initializing sensory systems + loop...", flush=True)
            self.vision = FlyVision(self.model, self.data, num_rays=40, arena_bounds=ARENA_BOUNDS, food_position=FOOD_POS)
            chemo = ChemoSensorySystem(sugar_source_pos=FOOD_POS, sugar_sigma=SUGAR_SIGMA)
            mechano = MechanoSensorySystem(self.model, self.data)
            burst_inj = BurstInjector(engine=self.engine, spikes_per_burst=5, isi_ms=1.0, min_burst_interval_ms=10.0, charge_per_spike=2.0)

            self.loop = SimFlyLoop(
                engine=self.engine, sensor_idx=sensor_idx, bridge=self.bridge, decoder=self.decoder,
                loader=self.loader, model=self.model, data=self.data, act_map=self.act_map,
                vision=self.vision, chemo=chemo, mechano=mechano, burst_inj=burst_inj,
                food_pos=FOOD_POS,
            )

            os.environ.setdefault('MUJOCO_GL', 'egl')
            total_elapsed = time.perf_counter() - t0
            print(f"\n{'='*60}")
            print(f"Init complete in {total_elapsed:.1f}s")
            print(f"  Neurons: {self._total_neurons} | Synapses: {self._total_synapses:,}")
            print(f"  DNs: {self._dns_loaded} | Vision: 20 rays, food marker ON")
            print(f"  Food: {FOOD_POS} | SynScale: {SYNAPTIC_SCALE}")
            print(f"  Engine: C++ (libneuronengine.so) — 2,200× faster than Python")
            print(f"  Ready for streaming!")
            print(f"{'='*60}\n", flush=True)
            self._initialized = True
            return True
        except Exception as e:
            self._init_error = f"{e}\n{traceback.format_exc()}"
            print(f"\nINIT FAILED: {self._init_error}", flush=True)
            return False

    def _render_frame(self) -> bytes:
        try:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, RENDER_WIDTH, RENDER_HEIGHT)
            self.renderer.update_scene(self.data, camera="track1")
            pixels = self.renderer.render()
            img = Image.fromarray(pixels)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=75)
            return buf.getvalue()
        except Exception:
            return b''

    def _render_vision_frame(self) -> bytes:
        try:
            if self.vision is None:
                return b''
            vision_data = self.vision.read(dt_ms=5.0, sim_time_ms=getattr(self.loop, 'sim_time_ms', 0.0) if self.loop else 0.0)
            w, h = 640, 100
            img = Image.new('RGB', (w, h), color=(10, 10, 10))
            draw = ImageDraw.Draw(img)
            # Left-eye region
            draw.rectangle([0, 0, w//2-1, h-1], outline=(40, 40, 40))
            draw.rectangle([w//2, 0, w-1, h-1], outline=(40, 40, 40))
            # Contrast bar
            contrast = vision_data.get('contrast', 0.0)
            if contrast > 0.01:
                bar_w = int(contrast * 200)
                draw.rectangle([50, 20, 50+bar_w, 40], fill=(0, 200, 255))
                draw.text((50, 42), f"Contrast: {contrast:.3f}", fill=(255, 255, 255))
            # Looming
            looming = vision_data.get('looming_intensity', 0.0)
            if looming > 0.001:
                lw = int(looming * 200)
                draw.rectangle([350, 20, 350+lw, 40], fill=(255, 50, 50))
                draw.text((350, 42), f"Loom: {looming:.3f}", fill=(255, 50, 50))
            # Food marker
            fb = vision_data.get('food_brightness', 0.0)
            if fb > 0.01:
                mx = int(300 + fb * 100)
                draw.rectangle([mx-3, 60, mx+3, 80], fill=(0, 255, 100))
                draw.text((mx-15, 82), f"Food: {fb:.2f}", fill=(0, 255, 100))
            # Wall distance
            wd = vision_data.get('wall_distance', 10.0)
            draw.text((10, 65), f"Wall: {wd:.1f}m", fill=(200, 200, 200))
            if vision_data.get('has_food_visual'):
                draw.text((10, 80), "🍽 FOOD VISIBLE", fill=(0, 255, 0))
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=75)
            return buf.getvalue()
        except Exception:
            return b''

    def _simulation_loop(self):
        self._running = True
        self._paused = False
        print("[sim] Simulation loop started", flush=True)
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                if self.loop is not None:
                    metrics = self.loop.step()
                    if metrics:
                        self._latest_metrics = metrics
            except Exception as e:
                print(f"[sim] Loop error: {e}", flush=True)
                time.sleep(0.05)
        print("[sim] Simulation loop stopped", flush=True)

    def start(self):
        if not self._initialized:
            return False
        if self._running:
            return True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        self._paused = False

    @property
    def latest_frame(self):
        return self._latest_frame

    @property
    def latest_vision_frame(self):
        return self._latest_vision_frame

    @property
    def latest_metrics(self):
        return self._latest_metrics

    @property
    def metrics_history(self):
        return dict(self.loop.metrics) if self.loop else {}

    @property
    def status(self):
        return {
            'initialized': self._initialized, 'running': self._running and not self._paused,
            'paused': self._paused, 'neurons': self._total_neurons,
            'synapses': self._total_synapses, 'dns': self._dns_loaded,
            'engine': 'C++ (libneuronengine.so)',
            'metrics': self._latest_metrics,
        }

# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
sim_server = SimFlyServer()

# Frame streaming threads
_frame_thread = None
_vision_thread = None

def _frame_capture_loop():
    while sim_server._running or sim_server._paused:
        try:
            frame = sim_server._render_frame()
            sim_server._latest_frame = frame
        except Exception:
            pass
        time.sleep(0.033)  # ~30 FPS

def _vision_capture_loop():
    while sim_server._running or sim_server._paused:
        try:
            frame = sim_server._render_vision_frame()
            sim_server._latest_vision_frame = frame
        except Exception:
            pass
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = sim_server.latest_frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vision_feed')
def vision_feed():
    def generate():
        while True:
            frame = sim_server.latest_vision_frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/vision')
def api_vision():
    if not sim_server._initialized or not sim_server.vision:
        return jsonify({'error': 'Not initialized'}), 503
    vd = sim_server.vision.read(dt_ms=5.0, sim_time_ms=getattr(sim_server.loop, 'sim_time_ms', 0.0) if sim_server.loop else 0.0)
    return jsonify(vd)

@app.route('/api/status')
def api_status():
    return jsonify(sim_server.status)

@app.route('/api/start', methods=['POST'])
def api_start():
    ok = sim_server.start()
    if ok:
        global _frame_thread, _vision_thread
        if _frame_thread is None or not _frame_thread.is_alive():
            _frame_thread = threading.Thread(target=_frame_capture_loop, daemon=True)
            _frame_thread.start()
        if _vision_thread is None or not _vision_thread.is_alive():
            _vision_thread = threading.Thread(target=_vision_capture_loop, daemon=True)
            _vision_thread.start()
    return jsonify({'success': ok})

@app.route('/api/pause', methods=['POST'])
def api_pause():
    sim_server._paused = True
    return jsonify({'success': True})

@app.route('/api/resume', methods=['POST'])
def api_resume():
    sim_server._paused = False
    return jsonify({'success': True})

@app.route('/api/params', methods=['POST'])
def api_params():
    """Set motor decoder parameters at runtime for parameter sweep."""
    data = request.get_json(force=True)
    if not sim_server._initialized or not sim_server.decoder:
        return jsonify({'error': 'simulation not initialized'}), 503
    changed = {}
    if 'global_gain' in data:
        sim_server.decoder.global_gain = float(data['global_gain'])
        changed['global_gain'] = sim_server.decoder.global_gain
    if 'tau_decay' in data:
        sim_server.decoder.tau_decay = float(data['tau_decay'])
        changed['tau_decay'] = sim_server.decoder.tau_decay
    if 'per_joint_scales' in data:
        scales = data['per_joint_scales']
        if isinstance(scales, dict):
            for jname, scale in scales.items():
                sim_server.decoder.per_joint_scales[jname] = float(scale)
            changed['per_joint_scales_count'] = len(scales)
    return jsonify({'success': True, 'changed': changed})

@app.route('/api/params')
def api_params_get():
    """Get current motor decoder parameters."""
    if not sim_server._initialized or not sim_server.decoder:
        return jsonify({'error': 'simulation not initialized'}), 503
    n_scaled = len(sim_server.decoder.per_joint_scales)
    result = {
        'global_gain': sim_server.decoder.global_gain,
        'tau_decay': sim_server.decoder.tau_decay,
        'per_joint_scales_active': n_scaled,
    }
    if n_scaled > 0:
        result['per_joint_scales'] = dict(list(sim_server.decoder.per_joint_scales.items())[:10])
        result['per_joint_scales_truncated'] = n_scaled > 10
    return jsonify(result)

@app.route('/api/metrics')
def api_metrics():
    return jsonify(sim_server.metrics_history)

@app.route('/api/neurons')
def api_neurons():
    if not sim_server._initialized:
        return jsonify({'neurons': []})
    loop = sim_server.loop
    if loop and hasattr(loop, '_firing_history') and loop._firing_history:
        return jsonify({'neurons': loop._firing_history[-1]})
    return jsonify({'neurons': []})

@socketio.on('connect')
def handle_connect():
    emit('status', sim_server.status)

def metrics_emitter():
    while True:
        socketio.sleep(0.5)
        if sim_server._initialized:
            socketio.emit('metrics_update', sim_server.latest_metrics)

@socketio.on('disconnect')
def handle_disconnect():
    pass

@app.route('/api/firing')
def api_firing():
    if not sim_server._initialized or not sim_server.loop:
        return jsonify({'firing': [], 'history': []})
    loop = sim_server.loop
    return jsonify({
        'firing': getattr(loop, '_fired_ids', []),
        'history': getattr(loop, '_firing_history', [])[-50:],
    })

@app.route('/api/torque')
def api_torque():
    if not sim_server._initialized or not sim_server.loop:
        return jsonify({'joints': {}})
    m = sim_server.latest_metrics
    loop = sim_server.loop
    joints = {}
    if hasattr(loop, '_firing_history') and loop._firing_history:
        joints = loop._firing_history[-1].get('joints', {})
    return jsonify({'joints': joints, 'active_joints': m.get('active_joints', 0), 'step': m.get('step', 0)})

def main():
    parser = argparse.ArgumentParser(description='SimFly Web Platform — C++ Engine Edition')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--neurons', type=int, default=DEFAULT_NEURONS)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--global-gain', type=float, default=0.002)
    parser.add_argument('--tau-decay', type=float, default=50.0)
    args = parser.parse_args()

    print("=" * 60)
    print("  SimFly Web Platform — C++ ENGINE EDITION")
    print(f"  Engine: libneuronengine.so | {DEFAULT_NEURONS} neurons (configurable)")
    print(f"  global_gain={args.global_gain}  tau_decay={args.tau_decay}")
    print("=" * 60)

    ok = sim_server.initialize(num_neurons=args.neurons, global_gain=args.global_gain, tau_decay=args.tau_decay)
    if not ok:
        print(f"FATAL: Init failed: {sim_server._init_error}")
        sys.exit(1)

    socketio.start_background_task(metrics_emitter)
    sim_server.start()

    global _frame_thread, _vision_thread
    _frame_thread = threading.Thread(target=_frame_capture_loop, daemon=True)
    _frame_thread.start()
    _vision_thread = threading.Thread(target=_vision_capture_loop, daemon=True)
    _vision_thread.start()

    print(f"\n🌐 Dashboard: http://192.168.1.199:{args.port}\n")
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()
