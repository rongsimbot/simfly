#!/usr/bin/env python3
"""
SimFly Web Platform — Phase 10 Sensory-Driven (v3)
====================================================
Flask + Socket.IO web dashboard for connectome-driven simulation.
Includes FlyVision display — shows what the fly sees.
MJPEG video stream + live metrics via Socket.IO + Chart.js frontend.

FIXES (2026-06-11):
  - Visual food marker in FlyVision
  - Configurable arena bounds
  - Self-geom filtering in obstacle detection
  - Sim-time-based looming detection
  - Closer food (2m), wider chemo gradient (sigma=4.0)
  - Lower synaptic scale (0.005)
  - NEW: Fly vision display endpoint (/api/vision)

Usage:
  DISPLAY=:10 MUJOCO_GL=egl python3 server.py
  → http://192.168.1.199:8080
"""

import argparse, csv, gzip, json, math, os, sys, time, threading, traceback
from collections import defaultdict, Counter, deque
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ───────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
FLYWIRE_DIR = os.path.join(HOME, "simrobotics-storage", "research", "flywire")
SENSORY_DIR = os.path.join(CODE_ROOT, "sensory")

for d in [CODE_ROOT, SENSORY_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from neuron_engine.neurons import NeuronBase, NeuronModel
from neuron_engine.engine import NeuronArrayBase
from neuron_engine.synapses import Synapse, SynapseModel
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
RESEARCH = os.path.join(HOME, "simrobotics-storage", "research")
CONNECTIONS_CSV = os.path.join(RESEARCH, "connections_princeton_no_threshold.csv.gz")
VNC_DIR = os.path.join(CODE_ROOT, "vnc_bridge")
SIMFLY_XML = os.path.join(FLYWIRE_DIR, "virtual-fly", "simfly_model", "simfly_grounded.xml")
DN_MATCHES_JSON = os.path.join(VNC_DIR, "dn_matches.json")
DN_MN_PATHWAYS_JSON = os.path.join(VNC_DIR, "dn_mn_pathways.json")
MOTOR_MAP_JSON = os.path.join(CODE_ROOT, "neuron_engine", "motor_neuron_map.json")

NT_WEIGHT_MAP: Dict[str, float] = {
    'ACH': 1.5, 'DA': 0.75, 'OCT': 0.75,
    'GABA': -0.5, 'GLUT': -0.25, 'SER': -0.15,
}

DEFAULT_NEURONS = 0  # 0 = ALL neurons — directive: full connectome required
RENDER_WIDTH, RENDER_HEIGHT = 640, 480
SYNAPTIC_SCALE = 0.005  # FIX: Lowered for joint differentiation
FOOD_POS = (1.2, -0.15, 0.20)  # FIX: Moved closer
SUGAR_SIGMA = 4.0  # FIX: Widened gradient
ARENA_BOUNDS = {'x_min': -10.0, 'x_max': 5.0, 'y_min': -5.0, 'y_max': 5.0, 'z_min': -1.0, 'z_max': 5.0}

# ── Burst Injector ───────────────────────────────────────────────────
class BurstInjector:
    def __init__(self, engine, spikes_per_burst=5, isi_ms=1.0, min_burst_interval_ms=10.0, charge_per_spike=2.0):
        self.engine = engine
        self.spikes_per_burst = spikes_per_burst
        self.isi_ms = isi_ms
        self.min_burst_interval_ms = min_burst_interval_ms
        self.charge_per_spike = charge_per_spike
        self._active_bursts: Dict[int, Tuple[int, float]] = {}
        self._last_burst_time: Dict[int, float] = {}
        self._sim_time_ms: float = 0.0

    def trigger_burst(self, neuron_idx: int) -> bool:
        if neuron_idx < 0 or neuron_idx >= len(self.engine.neurons):
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
                self.engine.neurons[neuron_idx].add_to_current_value(self.charge_per_spike)
                self.engine.add_neuron_to_fire_list1(neuron_idx)
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


# ── Sensory Neuron ID ────────────────────────────────────────────────
def identify_sensory_neurons(connections_csv, loaded_neuron_ids, max_sensory=500):
    in_degree = Counter()
    out_degree = Counter()
    t0 = time.perf_counter()
    with gzip.open(connections_csv, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = int(row['pre_root_id'])
            post = int(row['post_root_id'])
            syn = int(row['syn_count'])
            if pre in loaded_neuron_ids and post in loaded_neuron_ids:
                out_degree[pre] += syn
                in_degree[post] += syn
    t1 = time.perf_counter()
    print(f"  [sensory] Scanned {t1-t0:.1f}s", flush=True)
    neuron_scores = []
    for nid in loaded_neuron_ids:
        indeg = in_degree.get(nid, 0)
        outdeg = out_degree.get(nid, 0)
        sensory_score = float(outdeg) if indeg == 0 else float(outdeg) / float(indeg)
        neuron_scores.append((nid, sensory_score, outdeg, indeg))
    neuron_scores.sort(key=lambda x: x[1], reverse=True)
    n_visual = min(max_sensory // 2, len(neuron_scores))
    n_mechano = min(max_sensory // 4, max(0, len(neuron_scores) - n_visual))
    n_chemo = min(max_sensory // 4, max(0, len(neuron_scores) - n_visual - n_mechano))
    return {
        'visual_input': [nid for nid, _, _, _ in neuron_scores[:n_visual]],
        'mechano_input': [nid for nid, _, _, _ in neuron_scores[n_visual:n_visual+n_mechano]],
        'chemo_input': [nid for nid, _, _, _ in neuron_scores[n_visual+n_mechano:n_visual+n_mechano+n_chemo]],
    }


# ── Engine Builder ───────────────────────────────────────────────────
def build_engine(neurons_nt, connections, loader):
    sorted_ids = sorted(neurons_nt.keys())
    flywire_to_idx = {fw_id: idx for idx, fw_id in enumerate(sorted_ids)}
    idx_to_flywire = sorted_ids
    loader.flywire_to_idx = flywire_to_idx
    loader.idx_to_flywire = idx_to_flywire
    loader.neuron_nt_types = neurons_nt
    loader.nt_weight_map = NT_WEIGHT_MAP

    neurons = []
    for i, fw_id in enumerate(idx_to_flywire):
        neurons.append(NeuronBase(neuron_id=i, model=NeuronModel.IF, leak_rate=0.03, refractory_delay=1, label=f"fw_{fw_id}"))

    max_syn = max((c[2] for c in connections), default=1)
    synapses_by_pre = defaultdict(list)
    synapses_by_post = defaultdict(list)
    for pre_id, post_id, syn_count in connections:
        nt_type = neurons_nt.get(pre_id, 'ACH')
        base_weight = NT_WEIGHT_MAP.get(nt_type, 0.0)
        weight = base_weight * syn_count / math.log(1 + max_syn) * SYNAPTIC_SCALE
        weight = max(-0.05, min(0.05, weight))  # FIX: Tight clamp
        pre_idx = flywire_to_idx[pre_id]
        post_idx = flywire_to_idx[post_id]
        syn = Synapse(target_neuron_id=post_idx, source_neuron_id=pre_idx, weight=weight, model=SynapseModel.FIXED)
        synapses_by_pre[pre_idx].append(syn)
        synapses_by_post[post_idx].append(syn)

    engine = NeuronArrayBase(neurons=neurons, thread_count=4)
    for pre_idx, syns in synapses_by_pre.items():
        engine.neurons[pre_idx].synapses_out = syns
    for post_idx, syns in synapses_by_post.items():
        engine.neurons[post_idx].synapses_from = syns

    all_weights = [s.weight for n in engine.neurons for s in n.synapses_out]
    pos_w = sum(1 for w in all_weights if w > 0)
    neg_w = sum(1 for w in all_weights if w < 0)
    print(f"  [engine] {len(all_weights):,} synapses, {pos_w:,} excit, {neg_w:,} inhib, mean={np.mean(all_weights):.4f}", flush=True)
    return engine


# ── Sensory-Driven Loop ──────────────────────────────────────────────
class SimFlyLoop:
    def __init__(self, engine, sensor_idx, bridge, decoder, loader, model, data, act_map, vision, chemo, mechano, burst_inj, food_pos):
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
        effective = max(contrast, food_vis * 0.8)  # Visual food is a strong attractant

        if effective > 0.01:
            n_b = max(1, int(effective * 15))
            targets = list(self.sensor_idx.get('visual', set()))
            if targets:
                chosen = np.random.choice(targets, min(n_b, len(targets)), replace=False)
                for nid in chosen:
                    if self.burst_inj.trigger_burst(int(nid)):
                        total += self.burst_inj.spikes_per_burst

        sugar = chemo_data.get('sugar_concentration', 0.0)
        if sugar > 0.001:  # FIX: Lower threshold - food closer means higher concentration
            n_b = max(1, int(sugar * 15))
            chem_targets = list(self.sensor_idx.get('chemo', set()))
            if chem_targets:
                chosen = np.random.choice(chem_targets, min(n_b, len(chem_targets)), replace=False)
                for nid in chosen:
                    if self.burst_inj.trigger_burst(int(nid)):
                        total += self.burst_inj.spikes_per_burst

        if mechano_data.get('is_on_ground', False):
            cf = mechano_data.get('total_contact_force', 0.0)
            if cf > 0.001:
                n_b = min(3, max(1, int(cf * 2)))
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
        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=dt)
        mechano_data = self.mechano.read()

        total_fired = 0
        all_fired: Set[int] = set()

        for _ in range(5):
            self.burst_inj.step(1.0)
            self._inject_sensory(vision_data, chemo_data, mechano_data)
            fc, _ = self.engine.fire()
            total_fired += fc
            for i in range(self.engine.array_size):
                if self.engine.is_in_fire_list2(i):
                    all_fired.add(i)
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
        # Track fired neuron IDs for raster display
        self._fired_ids = list(all_fired) if all_fired else []
        if not hasattr(self, '_firing_history'):
            self._firing_history = []
        self._firing_history.append({
            'step': self.step_count,
            'fired': self._fired_ids[:100],  # Top 100
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
        self.loop = None
        self.renderer = None
        self.engine = None
        self.loader = None
        self.bridge = None
        self.decoder = None
        self.model = None
        self.data = None
        self.act_map = {}
        self.vision = None
        self.neuron_nt_types = {}
        self.idx_to_flywire = []
        self._running = False
        self._paused = False
        self._sim_thread = None
        self._sim_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_vision_jpeg = None
        self._vision_jpeg_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics_history = deque(maxlen=200)
        self._latest_metrics = {}
        self._initialized = False
        self._init_error = None
        self.render_every = 2
        self._total_neurons = 0
        self._total_synapses = 0
        self._dns_loaded = 0

    def initialize(self, num_neurons=500):
        try:
            print(f"\n{'='*60}")
            print(f"SimFly Web Platform — Phase 10 Sensory-Driven")
            print(f"{'='*60}")
            print(f"  Neurons: {num_neurons} | SynScale: {SYNAPTIC_SCALE}")
            print(f"  Food: {FOOD_POS} | Sigma: {SUGAR_SIGMA}")
            print(f"  Arena: x=[{ARENA_BOUNDS['x_min']},{ARENA_BOUNDS['x_max']}]")
            t0 = time.perf_counter()

            # Bridge
            print("\n[1/6] Loading DN→MN bridge...", flush=True)
            self.bridge = DnMnBridge(dn_matches_path=DN_MATCHES_JSON, pathways_path=DN_MN_PATHWAYS_JSON, min_pathway_confidence=0.01)
            self.bridge.initialize()
            bs = self.bridge.summary()
            self._dns_loaded = bs.get('dn_matches_loaded', 0)
            print(f"  Bridge: {self._dns_loaded} DNs, {bs.get('unique_mns_loaded', 0):,} MNs", flush=True)

            # DN IDs
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

            # Stream connections
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

            # Engine
            print(f"\n[4/6] Building NIRON engine...", flush=True)
            self.neuron_nt_types = {nid: all_nts.get(nid, 'unknown') for nid in included_ids}
            connections = [(pre, post, syn) for pre, post, syn, _ in all_connections if pre in included_ids and post in included_ids]
            config = {'min_syn_count': 0, 'leak_rate': 0.03, 'refractory_delay': 1, 'thread_count': 4, 'normalize_weights': True, 'model': NeuronModel.IF}
            self.loader = FlyWireConnectomeLoader(CONNECTIONS_CSV, config=config)
            self.engine = build_engine(self.neuron_nt_types, connections, self.loader)
            self._total_neurons = self.engine.array_size
            self._total_synapses = sum(len(n.synapses_out) for n in self.engine.neurons)
            print(f"  Engine: {self._total_neurons} neurons, {self._total_synapses:,} synapses", flush=True)

            # Sensory
            print(f"\n[5/6] Identifying sensory + loading MuJoCo...", flush=True)
            sensor_map = identify_sensory_neurons(CONNECTIONS_CSV, included_ids, max_sensory=300)
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

            self.decoder = VNCMotorDecoder.load_from_vnc(
                vnc_actuator_map_path=os.path.join(VNC_DIR, "vnc_actuator_map.json"),
                pathways_path=DN_MN_PATHWAYS_JSON, tau_decay=50.0, global_gain=0.0002,  # ultra-low to prevent saturation
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

            # Sensory systems + Loop
            print(f"\n[6/6] Initializing sensory systems + loop...", flush=True)
            self.vision = FlyVision(self.model, self.data, num_rays=20, arena_bounds=ARENA_BOUNDS, food_position=FOOD_POS)
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
            print(f"  Food: {FOOD_POS} | Sigma: {SUGAR_SIGMA} | SynScale: {SYNAPTIC_SCALE}")
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
            img.save(buf, format='JPEG', quality=85)
            return buf.getvalue()
        except Exception as e:
            if "EGL" not in str(e):
                print(f"  [render] {e}", flush=True)
            return self._latest_jpeg or b''

    def _render_vision_frame(self) -> bytes:
        """Render what the fly sees."""
        try:
            if self.vision:
                img = self.vision.get_fly_view_image(width=640, height=240)
                if img:
                    # Add overlay text
                    draw = ImageDraw.Draw(img)
                    vision_data = self.vision.read(dt_ms=5.0, sim_time_ms=self.loop.sim_time_ms if self.loop else 0)
                    draw.text((5, 5), f"FlyVision — Contrast: {vision_data.get('contrast',0):.3f}", fill=(0,255,0))
                    draw.text((5, 22), f"Food: {'VISIBLE' if vision_data.get('has_food_visual') else '---'} | Looming: {vision_data.get('looming_intensity',0):.3f}", fill=(255,200,100))
                    draw.text((5, 39), f"LC4: {vision_data.get('lc4_rate',0):.1f} Hz | Wall: {vision_data.get('wall_distance',10):.1f}m", fill=(150,200,255))
                    buf = BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    return buf.getvalue()
        except Exception as e:
            pass
        # Fallback: dark frame
        img = Image.new('RGB', (640, 240), color=(5, 5, 20))
        draw = ImageDraw.Draw(img)
        draw.text((200, 110), "Vision not initialized", fill=(100, 100, 100))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()

    def _simulation_loop(self):
        print("[SIM] Thread started", flush=True)
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                with self._sim_lock:
                    report = self.loop.step()
            except Exception as e:
                print(f"[SIM] Step error: {e}", flush=True)
                time.sleep(0.1)
                continue

            step = report['step']
            if step % self.render_every == 0:
                try:
                    jpeg = self._render_frame()
                    with self._frame_lock:
                        self._latest_jpeg = jpeg
                except Exception:
                    pass

            # Vision frame every other step
            if step % 3 == 0:
                try:
                    vjpeg = self._render_vision_frame()
                    with self._vision_jpeg_lock:
                        self._latest_vision_jpeg = vjpeg
                except Exception:
                    pass

            with self._metrics_lock:
                self._latest_metrics = report
                self._metrics_history.append(report)

    def start(self):
        if not self._initialized: return False
        if self._running and not self._paused: return True
        if self._paused:
            self._paused = False
            return True
        self._running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()
        return True

    def stop(self):
        self._running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=3.0)

    @property
    def latest_frame(self):
        with self._frame_lock:
            return self._latest_jpeg

    @property
    def latest_vision_frame(self):
        with self._vision_jpeg_lock:
            return self._latest_vision_jpeg

    @property
    def latest_metrics(self):
        with self._metrics_lock:
            return dict(self._latest_metrics) if self._latest_metrics else {}

    @property
    def metrics_history(self):
        with self._metrics_lock:
            return list(self._metrics_history)

    @property
    def status(self):
        return {
            'initialized': self._initialized, 'running': self._running and not self._paused,
            'paused': self._paused, 'neurons': self._total_neurons,
            'synapses': self._total_synapses, 'dns_loaded': self._dns_loaded,
            'step': self._latest_metrics.get('step', 0) if self._latest_metrics else 0,
            'error': self._init_error,
        }


# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'simfly-vision-v3'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')
sim_server = SimFlyServer()


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
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vision_feed')
def vision_feed():
    """Stream what the fly sees."""
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
    """Get current vision data snapshot."""
    if not sim_server._initialized or not sim_server.vision:
        return jsonify({'error': 'Not initialized'}), 503
    vd = sim_server.vision.read(dt_ms=5.0, sim_time_ms=sim_server.loop.sim_time_ms if sim_server.loop else 0)
    return jsonify({
        'contrast': vd.get('contrast', 0),
        'has_food_visual': vd.get('has_food_visual', False),
        'food_brightness': vd.get('food_brightness', 0),
        'has_wall': vd.get('has_wall', False),
        'wall_distance': vd.get('wall_distance', 10),
        'looming_intensity': vd.get('looming_intensity', 0),
        'lc4_rate': vd.get('lc4_rate', 0),
        'left_eye_brightness': vd.get('left_eye_brightness', 0),
        'right_eye_brightness': vd.get('right_eye_brightness', 0),
        'nearest_distance': vd.get('nearest_distance', 10),
        'obstacles_count': sum(1 for o in vd.get('obstacles', []) if o.get('hit')),
    })


@app.route('/api/status')
def api_status():
    return jsonify({**sim_server.status, 'latest_metrics': sim_server.latest_metrics})


@app.route('/api/start', methods=['POST'])
def api_start():
    ok = sim_server.start()
    return jsonify({'success': ok, 'status': sim_server.status})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    sim_server._paused = True
    return jsonify({'success': True, 'status': sim_server.status})


@app.route('/api/resume', methods=['POST'])
def api_resume():
    sim_server._paused = False
    return jsonify({'success': True, 'status': sim_server.status})


@app.route('/api/metrics')
def api_metrics():
    history = sim_server.metrics_history
    return jsonify({'history': history, 'latest': sim_server.latest_metrics})


@app.route('/api/neurons')
def api_neurons():
    if not sim_server._initialized:
        return jsonify({'neurons': []})
    neurons = []
    for idx, fw_id in enumerate(sim_server.idx_to_flywire[:100]):
        nt = sim_server.neuron_nt_types.get(fw_id, 'unknown')
        neurons.append({'index': idx, 'flywire_id': fw_id, 'nt_type': nt})
    return jsonify({'neurons': neurons, 'total': len(sim_server.idx_to_flywire)})


@socketio.on('connect')
def handle_connect():
    emit('status', sim_server.status)


def metrics_emitter():
    last_step = -1
    while sim_server._running:
        m = sim_server.latest_metrics
        if m and m.get('step') != last_step:
            last_step = m['step']
            firing_history = getattr(sim_server.loop, '_firing_history', [])
            last_fired = firing_history[-1] if firing_history else {}
            socketio.emit('firing', {
                'step': last_fired.get('step', 0),
                'fired_count': len(last_fired.get('fired', [])),
                'dn_count': len(last_fired.get('dns', [])),
                'mn_count': len(last_fired.get('mns', [])),
                'fired_ids': last_fired.get('fired', [])[:50],
            })
            socketio.emit('metrics', {
                'step': m.get('step', 0),
                'time_s': m.get('time_ms', 0) / 1000.0,
                'z': round(m.get('z_height', 0), 4),
                'fired': m.get('fired_neurons', 0),
                'dns': m.get('dn_matches', 0),
                'mns': m.get('mns_activated', 0),
                'torque': m.get('torque_applied', False),
                'on_ground': m.get('on_ground', False),
                'contrast': round(m.get('contrast', 0), 4),
                'food_visual': m.get('has_food_visual', False),
                'food_brightness': round(m.get('food_brightness', 0), 4),
                'sugar_conc': round(m.get('sugar_conc', 0), 4),
                'food_distance': round(m.get('food_distance', 0), 3),
                'wall_distance': round(m.get('wall_distance', 0), 1),
                'looming': round(m.get('looming', 0), 4),
                'lc4_rate': round(m.get('lc4_rate', 0), 1),
            })
        socketio.sleep(0.05)


@socketio.on('disconnect')
def handle_disconnect():
    pass

@app.route('/api/firing')
def api_firing():
    """Return recent neuron firing history for raster plot."""
    if not sim_server._initialized or not sim_server.loop:
        return jsonify({'history': []})
    h = getattr(sim_server.loop, '_firing_history', [])
    return jsonify({
        'history': h[-50:],  # Last 50 steps
        'total_neurons': sim_server._total_neurons,
        'step': sim_server.loop.step_count,
    })


@app.route('/api/torque')
def api_torque():
    """Return current joint torque distribution."""
    if not sim_server._initialized or not sim_server.loop:
        return jsonify({'joints': {}})
    h = getattr(sim_server.loop, '_firing_history', [])
    if h:
        return jsonify({'joints': h[-1].get('joints', {}), 'step': h[-1].get('step', 0)})
    return jsonify({'joints': {}, 'step': 0})


def main():
    parser = argparse.ArgumentParser(description='SimFly Web Platform v3')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--neurons', type=int, default=DEFAULT_NEURONS)
    parser.add_argument('--no-init', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')

    if not args.no_init:
        print("Initializing SimFly pipeline...", flush=True)
        ok = sim_server.initialize(num_neurons=args.neurons)
        if not ok:
            print("Pipeline init FAILED. Starting server without simulation.", flush=True)

    print(f"\nSimFly Web Platform v3 on http://{args.host}:{args.port}", flush=True)
    print(f"  Video: http://192.168.1.199:{args.port}/video_feed", flush=True)
    print(f"  Vision: http://192.168.1.199:{args.port}/vision_feed", flush=True)
    print(f"  Status: http://192.168.1.199:{args.port}/api/status", flush=True)
    print(f"  Vision API: http://192.168.1.199:{args.port}/api/vision", flush=True)
    print(f"  Food: {FOOD_POS} | Sigma: {SUGAR_SIGMA} | SynScale: {SYNAPTIC_SCALE}", flush=True)

    if sim_server._initialized:
        sim_server.start()
        print("  Auto-started simulation", flush=True)

    socketio.start_background_task(metrics_emitter)
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
