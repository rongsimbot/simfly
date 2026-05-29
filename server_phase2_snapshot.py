#!/usr/bin/env python3
"""
SimFly Web Platform — Phase 2
==============================
Flask + Socket.IO web dashboard for real-time connectome-driven simulation.
MJPEG video stream + live metrics via Socket.IO + Chart.js frontend.

Phase 2 additions:
  - Hot-reload dev mode (--watch + file watcher)
  - Live reconfigure API (POST /api/configure)
  - Test runner (POST /api/test, GET /api/tests)
  - Connectome stats (GET /api/connectome_stats)
  - Data export (GET /api/export)

Architecture:
  Browser ← HTTP/WebSocket → Flask ← import → Phase 14 Pipeline
                         ↓
                   MuJoCo (EGL headless)

Usage:
  DISPLAY=:10 MUJOCO_GL=egl python3 server.py
  DISPLAY=:10 MUJOCO_GL=egl python3 server.py --watch  # hot-reload mode
  → http://192.168.1.199:8080
"""

import argparse, csv, gzip, json, math, os, sys, time, threading, traceback
import importlib, importlib.util
from collections import defaultdict, Counter, deque
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Path Configuration ───────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
FLYWIRE_DIR = os.path.join(HOME, "simrobotics-storage", "research", "flywire")
SENSORY_DIR = os.path.join(CODE_ROOT, "sensory")

for d in [CODE_ROOT, SENSORY_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# ── Scientific Imports ───────────────────────────────────────────────────
from neuron_engine.engine import NeuronArrayBase
from neuron_engine.neurons import NeuronBase, NeuronModel
from neuron_engine.synapses import Synapse, SynapseModel
from connectome.connectome_loader import FlyWireConnectomeLoader
from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
from phase8_integration.dn_mn_bridge import DnMnBridge
from sensory_injector import SensoryInjector
from vision import FlyVision
from chemo import ChemoSensorySystem
from mechano import MechanoSensorySystem
from arena_visualizer import ArenaVisualizer
from vnc_bridge.dn_subtype_classifier import DNSubtypeClassifier
from vnc_bridge.gait_controller import GaitController
from proprioception import ProprioceptiveFeedback
import mujoco

# ── Web Imports ──────────────────────────────────────────────────────────
from frame_recorder import FrameRecorder
from flask import Flask, Response, render_template, jsonify, request, redirect
from flask_socketio import SocketIO, emit

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
RESEARCH = os.path.join(HOME, "simrobotics-storage", "research")
CONNECTIONS_CSV = os.path.join(RESEARCH, "connections_princeton_no_threshold.csv.gz")
VNC_DIR = os.path.join(CODE_ROOT, "vnc_bridge")
SIMFLY_XML = os.path.join(FLYWIRE_DIR, "virtual-fly", "simfly_model", "simfly_grounded.xml")
MOTOR_MAP_JSON = os.path.join(CODE_ROOT, "neuron_engine", "motor_neuron_map.json")
DN_MATCHES_JSON = os.path.join(VNC_DIR, "dn_matches.json")
DN_MN_PATHWAYS_JSON = os.path.join(VNC_DIR, "dn_mn_pathways.json")
TEST_CONFIGS_PATH = "/tmp/simfly_web/test_configs.json"

NT_WEIGHT_MAP: Dict[str, float] = {
    'ACH': 1.5, 'DA': 0.75, 'OCT': 0.75,
    'GABA': -0.5, 'GLUT': -0.25, 'SER': -0.15,
}

DEFAULT_NEURONS = 2000
DEFAULT_GLOBAL_GAIN = 0.001
RENDER_WIDTH, RENDER_HEIGHT = 640, 480
RENDER_EVERY_DEFAULT = 2
METRICS_HISTORY_SIZE = 200

# ═══════════════════════════════════════════════════════════════════════════
# FILE WATCHER (Phase 2: Hot-Reload Dev Mode)
# ═══════════════════════════════════════════════════════════════════════════
class ModuleFileWatcher:
    """Watches integration module directories for .py changes and reloads affected modules."""

    def __init__(self, server_instance=None):
        self.server = server_instance
        self._observer = None
        self._watched_modules: Dict[str, str] = {}  # module_name -> file_path
        self._running = False

    def start(self):
        """Start the file watcher in a background thread."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            print("[WATCH] ⚠ watchdog not installed. Install: pip install watchdog", flush=True)
            return False

        watch_dirs = []
        # Sensory modules
        if os.path.isdir(SENSORY_DIR):
            watch_dirs.append(SENSORY_DIR)
        # VNC bridge
        vnc_path = os.path.join(CODE_ROOT, "vnc_bridge")
        if os.path.isdir(vnc_path):
            watch_dirs.append(vnc_path)
        # Neuron engine
        engine_path = os.path.join(CODE_ROOT, "neuron_engine")
        if os.path.isdir(engine_path):
            watch_dirs.append(engine_path)
        # Connectome
        conn_path = os.path.join(CODE_ROOT, "connectome")
        if os.path.isdir(conn_path):
            watch_dirs.append(conn_path)

        if not watch_dirs:
            print("[WATCH] ⚠ No integration directories found to monitor", flush=True)
            return False

        class ReloadHandler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    self.watcher._reload_module(event.src_path)

        handler = ReloadHandler(self)
        self._observer = Observer()
        for d in watch_dirs:
            self._observer.schedule(handler, d, recursive=True)
            print(f"[WATCH] Monitoring: {d}", flush=True)

        self._observer.start()
        self._running = True
        print(f"[WATCH] ✅ File watcher active — {len(watch_dirs)} directories", flush=True)
        return True

    def _reload_module(self, filepath: str):
        """Reload a Python module by file path using importlib."""
        try:
            module_name = os.path.splitext(os.path.basename(filepath))[0]
            # Find the module in sys.modules
            for name, mod in list(sys.modules.items()):
                if hasattr(mod, '__file__') and mod.__file__:
                    if os.path.abspath(mod.__file__) == os.path.abspath(filepath):
                        try:
                            importlib.reload(mod)
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[WATCH] 🔄 Reloaded: {name} ({module_name}.py) at {timestamp}", flush=True)
                            return
                        except Exception as e:
                            print(f"[WATCH] ❌ Failed to reload {name}: {e}", flush=True)
                            return
            # If module not found by filepath match, try loading by name
            print(f"[WATCH] 📝 Change detected in {module_name}.py (module not yet tracked)", flush=True)
        except Exception as e:
            print(f"[WATCH] ⚠ Error watching {filepath}: {e}", flush=True)

    def stop(self):
        """Stop the file watcher."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            print("[WATCH] File watcher stopped", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# TEST RUNNER (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════
class TestManager:
    """Manages queued test configurations for automated parameter sweeps."""

    def __init__(self, storage_path=TEST_CONFIGS_PATH):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._test_thread: Optional[threading.Thread] = None
        self._running = False
        self._ensure_storage()

    def _ensure_storage(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([], f)

    def _load(self) -> List[Dict]:
        try:
            with open(self.storage_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self, tests: List[Dict]):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(tests, f, indent=2)

    def queue(self, name: str, config: Dict, duration_seconds: int) -> str:
        """Queue a new test configuration. Returns test_id."""
        test_id = f"test-{int(time.time() * 1000)}"
        with self._lock:
            tests = self._load()
            test = {
                'id': test_id,
                'name': name,
                'config': config,
                'duration_seconds': duration_seconds,
                'status': 'queued',
                'created': datetime.now().isoformat(),
                'started': None,
                'completed': None,
                'results': None,
            }
            tests.append(test)
            self._save(tests)
        return test_id

    def list_tests(self) -> List[Dict]:
        with self._lock:
            return self._load()

    def _run_test_loop(self, sim_server, test: Dict, duration_s: int):
        """Run a single test configuration."""
        config = test.get('config', {})
        print(f"[TEST] Running: {test['name']} ({test['id']})", flush=True)

        # Apply configuration
        if 'gain' in config and sim_server.decoder:
            try:
                sim_server.decoder.global_gain = float(config['gain'])
                print(f"  [TEST] Set gain={config['gain']}", flush=True)
            except Exception as e:
                print(f"  [TEST] Failed to set gain: {e}", flush=True)

        if 'render_every' in config:
            sim_server.render_every = int(config['render_every'])

        # Ensure running
        if not sim_server._running:
            sim_server.start()

        # Run for duration
        start_time = time.perf_counter()
        step_count = 0
        metrics_snapshot = []
        while (time.perf_counter() - start_time) < duration_s:
            time.sleep(0.1)
            step_count = sim_server._latest_metrics.get('step', 0) if sim_server._latest_metrics else 0
            # Capture periodic snapshots
            if sim_server._latest_metrics and step_count % 20 == 0:
                metrics_snapshot.append(dict(sim_server._latest_metrics))

        return {
            'duration_actual_s': round(time.perf_counter() - start_time, 1),
            'final_step': step_count,
            'metrics_snapshots': metrics_snapshot[-10:],  # last 10 snapshots
        }

    def start_runner(self, sim_server):
        """Start background test runner thread if not already running."""
        if self._running:
            return
        self._running = True
        self._test_thread = threading.Thread(
            target=self._test_runner_loop,
            args=(sim_server,),
            daemon=True,
            name="test-runner"
        )
        self._test_thread.start()

    def _test_runner_loop(self, sim_server):
        """Background loop: process queued tests sequentially."""
        print("[TEST] Test runner started", flush=True)
        while self._running:
            with self._lock:
                tests = self._load()
                queued = [t for t in tests if t['status'] == 'queued']

            if not queued:
                time.sleep(1.0)
                continue

            test = queued[0]
            # Mark as running
            with self._lock:
                tests = self._load()
                for t in tests:
                    if t['id'] == test['id']:
                        t['status'] = 'running'
                        t['started'] = datetime.now().isoformat()
                self._save(tests)

            try:
                results = self._run_test_loop(
                    sim_server, test, test.get('duration_seconds', 30)
                )
                with self._lock:
                    tests = self._load()
                    for t in tests:
                        if t['id'] == test['id']:
                            t['status'] = 'completed'
                            t['completed'] = datetime.now().isoformat()
                            t['results'] = results
                    self._save(tests)
                print(f"[TEST] ✅ Completed: {test['name']}", flush=True)
            except Exception as e:
                print(f"[TEST] ❌ Failed: {test['name']} — {e}", flush=True)
                with self._lock:
                    tests = self._load()
                    for t in tests:
                        if t['id'] == test['id']:
                            t['status'] = 'failed'
                            t['completed'] = datetime.now().isoformat()
                            t['results'] = {'error': str(e)}
                    self._save(tests)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════
# BURST INJECTOR (from Phase 14)
# ═══════════════════════════════════════════════════════════════════════════
class BurstInjector:
    def __init__(self, engine, spikes_per_burst=5, isi_ms=1.0,
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
                neuron = self.engine.neurons[neuron_idx]
                neuron.add_to_current_value(self.charge_per_spike)
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


# ═══════════════════════════════════════════════════════════════════════════
# DIRECT SENSORY INJECTOR (from Phase 14)
# ═══════════════════════════════════════════════════════════════════════════
class DirectSensoryInjector:
    def __init__(self, engine, burst_injector, sensor_engine_indices, rate_per_step=10):
        self.engine = engine
        self.burst_injector = burst_injector
        self.sensor_indices = sorted(sensor_engine_indices) if sensor_engine_indices else []
        self.rate_per_step = rate_per_step
        self.step_count = 0

    def inject(self) -> int:
        if not self.sensor_indices:
            return 0
        self.step_count += 1
        total = 0
        n = min(self.rate_per_step, len(self.sensor_indices))
        offset = (self.step_count * n) % len(self.sensor_indices)
        for i in range(n):
            idx = self.sensor_indices[(offset + i) % len(self.sensor_indices)]
            if self.burst_injector.trigger_burst(idx):
                total += self.burst_injector.spikes_per_burst
        return total


# ═══════════════════════════════════════════════════════════════════════════
# SENSORY NEURON IDENTIFIER (from Phase 14)
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 14 LOOP (adapted for continuous streaming)
# ═══════════════════════════════════════════════════════════════════════════
class Phase14Loop:
    def __init__(self, engine, sensor_engine_idx, bridge, vnc_decoder, loader,
                 model, data, actuator_map, vision, chemo, mechano, burst_injector,
                 direct_injector=None, brain_rate_hz=1000, physics_rate_hz=200, food_pos=(0.0,0.0,0.0),
                 dn_classifier=None, gait_controller=None, proprio=None):
        self.engine = engine
        self.sensor_engine_idx = sensor_engine_idx
        self.bridge = bridge
        self.decoder = vnc_decoder
        self.loader = loader
        self.model = model
        self.data = data
        self.actuator_map = actuator_map
        self.vision = vision
        self.chemo = chemo
        self.mechano = mechano
        self.burst_injector = burst_injector
        self.direct_injector = direct_injector
        self.dn_classifier = dn_classifier
        self.gait_controller = gait_controller
        self.proprio = proprio
        self.current_gait_mode = 'stance'
        self._food_pos = food_pos
        self.brain_rate_hz = brain_rate_hz
        self.physics_rate_hz = physics_rate_hz
        self.dt_brain_ms = 1000.0 / brain_rate_hz
        self.dt_physics_ms = 1000.0 / physics_rate_hz
        self.brain_steps_per_physics = max(1, brain_rate_hz // physics_rate_hz)
        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)
        self.sensory_input_history = []
        self.all_z_heights = []
        self.all_ground_contacts = []

    def _inject_sensory_bursts(self, vision_data, chemo_data, mechano_data) -> int:
        total = 0
        contrast = vision_data.get('contrast', 0.0)
        wall_hit = vision_data.get('has_wall', False)
        wall_dist = vision_data.get('wall_distance', 10.0)
        # Visual sensory fires on ANY contrast (landmarks, patterns, walls)
        if contrast > 0.01:
            n_bursts = max(1, int(contrast * 15))
            vis_targets = list(self.sensor_engine_idx.get('visual', set()))
            if vis_targets:
                chosen = np.random.choice(vis_targets, min(n_bursts, len(vis_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst
        # Wall-specific looming response (LC4 neurons)
        if wall_hit and wall_dist < 2.0:
            lc4_targets = list(self.sensor_engine_idx.get('lc4', set()))
            if lc4_targets:
                n_lc4 = max(1, int((2.0 - wall_dist) * 5))
                chosen = np.random.choice(lc4_targets, min(n_lc4, len(lc4_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst
        if chemo_data.get('sugar_concentration', 0) > 0.01:
            n_bursts = max(1, int(chemo_data['sugar_concentration'] * 10))
            chemo_targets = list(self.sensor_engine_idx.get('chemo', set()))
            if chemo_targets:
                chosen = np.random.choice(chemo_targets, min(n_bursts, len(chemo_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst
        if mechano_data.get('is_on_ground', False):
            contact_force = mechano_data.get('total_contact_force', 0)
            if contact_force > 0.001:
                n_bursts = min(3, max(1, int(contact_force * 2)))
                mech_targets = list(self.sensor_engine_idx.get('mechano', set()))
                if mech_targets:
                    chosen = np.random.choice(mech_targets, min(n_bursts, len(mech_targets)), replace=False)
                    for nid in chosen:
                        if self.burst_injector.trigger_burst(int(nid)):
                            total += self.burst_injector.spikes_per_burst
        return total

    def step(self) -> Dict[str, Any]:
        vision_data = self.vision.read(dt_ms=self.dt_physics_ms)
        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=self.dt_physics_ms)
        mechano_data = self.mechano.read()

        self.sensory_input_history.append({
            'contrast': vision_data.get('contrast', 0.0),
            'wall_distance': vision_data.get('wall_distance', 10.0),
            'has_wall': vision_data.get('has_wall', False),
            'on_ground': mechano_data.get('is_on_ground', False),
            'contact_force': mechano_data.get('total_contact_force', 0.0),
            'z_height': float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0,
        })

        # M3: Determine gait mode from chemotaxis gradient
        chemo_grad_x = chemo_data.get('gradient_x', 0.0)
        chemo_grad_y = chemo_data.get('gradient_y', 0.0)
        chemo_conc = chemo_data.get('sugar_concentration', 0.0)
        gait_mode = 'stance'
        if self.dn_classifier is not None:
            gait_mode = self.dn_classifier.determine_gait_mode(
                (chemo_grad_x, chemo_grad_y), chemo_conc
            )
        self.current_gait_mode = gait_mode

        if len(self.data.qpos) > 2:
            self.all_z_heights.append(float(self.data.qpos[2]))
        self.all_ground_contacts.append(1 if mechano_data.get('is_on_ground', False) else 0)

        total_fired = 0
        total_burst_spikes = 0
        all_fired_engine_indices: Set[int] = set()

        for _ in range(self.brain_steps_per_physics):
            burst_spikes = self.burst_injector.step(self.dt_brain_ms)
            total_burst_spikes += burst_spikes
            env_bursts = self._inject_sensory_bursts(vision_data, chemo_data, mechano_data)

            if self.direct_injector is not None and env_bursts == 0:
                self.direct_injector.inject()

            fired_count, cycle = self.engine.fire()
            # TODO: Cascade fix needed for DN matching
            total_fired += fired_count

            for word_idx, word_val in enumerate(self.engine._fire_list_2):
                if word_val == 0:
                    continue
                base_id = word_idx * 64
                remaining = word_val
                while remaining:
                    bit_pos = (remaining & -remaining).bit_length() - 1
                    neuron_id = base_id + bit_pos
                    if neuron_id < len(self.engine.neurons):
                        all_fired_engine_indices.add(neuron_id)
                    remaining &= remaining - 1
            self.sim_time_ms += self.dt_brain_ms

        mn_activations = {}
        bridge_report = {"dn_matches_found": 0, "mns_activated": 0, "dn_types": []}
        if self.bridge is not None and all_fired_engine_indices:
            mn_activations = self.bridge.translate(
                all_fired_engine_indices, self.loader,
                dn_classifier=self.dn_classifier, gait_mode=gait_mode
            )
            bridge_report = {
                "dn_matches_found": len(self.bridge.last_fired_dns),
                "mns_activated": len(self.bridge.last_activated_mns),
                "dn_types": list(self.bridge.last_fired_dns)[:10],
            }
            # M3: DN subtype classification
            walking_dn_count = 0
            turning_dn_count = 0
            if self.dn_classifier is not None and self.bridge.last_fired_dns:
                dn_sub_counts = self.dn_classifier.classify_dns(list(self.bridge.last_fired_dns))
                walking_dn_count = dn_sub_counts.get('walking', 0)
                turning_dn_count = dn_sub_counts.get('turn_left', 0) + dn_sub_counts.get('turn_right', 0)
                bridge_report['walking_dns'] = walking_dn_count
                bridge_report['turning_dns'] = turning_dn_count
                bridge_report['stop_dns'] = dn_sub_counts.get('stop', 0)

        if self.decoder is not None:
            self.decoder.accumulate(set(mn_activations.keys()) if mn_activations else set())
        joint_commands = self.decoder.decode() if self.decoder is not None else {}

        # M3: Apply gait controller to connectome-driven torques
        if self.gait_controller is not None:
            joint_commands = self.gait_controller.apply(
                gait_mode, joint_commands, dt_ms=self.dt_physics_ms
            )

        # M3: Chemotaxis baseline — ensure walk/turn produces visible movement
        # Applies realistic torque to swing+stance legs based on tripod phase
        if gait_mode != 'stance' and self.gait_controller is not None:
            gait_info = self.gait_controller.get_phase_info()
            swing_legs = set(gait_info.get('swing_legs', []))
            stance_legs = set(gait_info.get('stance_legs', []))
            chemo_strength = min(1.0, chemo_conc * 5.0)
            for jname in list(joint_commands.keys()):
                leg = self.gait_controller.joint_to_leg.get(jname)
                if leg is None:
                    continue
                if leg in swing_legs:
                    if 'femur' in jname and 'twist' not in jname:
                        joint_commands[jname] = 0.4 * chemo_strength
                    elif 'coxa' in jname and 'twist' not in jname and 'abduct' not in jname:
                        joint_commands[jname] = 0.2 * chemo_strength * (-1 if leg.endswith('_right') else 1)
                    elif 'tibia' in jname:
                        joint_commands[jname] = 0.1 * chemo_strength
                elif leg in stance_legs:
                    if 'femur' in jname and 'twist' not in jname:
                        joint_commands[jname] = -0.3 * chemo_strength
                    elif 'coxa' in jname and 'twist' not in jname and 'abduct' not in jname:
                        joint_commands[jname] = -0.1 * chemo_strength
                    elif 'tibia' in jname:
                        joint_commands[jname] = -0.05 * chemo_strength
        
        # Reset ctrl to prevent stale values from non-VNC actuators
        self.data.ctrl[:] = 0.0

        for jname, torque in joint_commands.items():
            if jname in self.actuator_map:
                idx = self.actuator_map[jname]
                self.data.ctrl[idx] = float(np.clip(torque, -1.0, 1.0))

        # M3: Z-height stabilizer — prevent fly from floating too high
        TARGET_Z = 0.06
        Z_DEADBAND = 0.012  # Allow ±2.5cm oscillation for walking
        Z_KP = 12.0
        Z_KD = 2.0
        if len(self.data.qpos) > 2:
            z_current = float(self.data.qpos[2])
            z_error = z_current - TARGET_Z
            # Only correct when fly is significantly above target (allow walk bounce)
            if z_error > Z_DEADBAND:
                z_vel = float(self.data.qvel[2]) if len(self.data.qvel) > 2 else 0.0
                z_correction = -(Z_KP * (z_error - Z_DEADBAND) + Z_KD * max(0, z_vel))
                z_clamped = float(np.clip(z_correction, -0.6, 0.0))  # Only downward correction
                # Apply to femur+coxa joints
                for jname in self.actuator_map:
                    if any(jt in jname for jt in ['femur', 'coxa', 'tibia']) and 'twist' not in jname and 'abduct' not in jname:
                        idx = self.actuator_map[jname]
                        self.data.ctrl[idx] += z_clamped

        # M4: Proprioceptive load compensation — balance forces across legs
        proprio_data = {}
        if self.proprio is not None:
            proprio_data = self.proprio.read()
            load_comp = proprio_data.get('compensation_torques', {})
            for jname, torque_adj in load_comp.items():
                if jname in self.actuator_map:
                    idx = self.actuator_map[jname]
                    self.data.ctrl[idx] += float(np.clip(torque_adj, -0.5, 0.5))

        try:
            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            pass
        
        self.step_count += 1
        active_joints = sum(1 for v in joint_commands.values() if abs(v) > 0.001)
        torque_applied = active_joints > 0

        self.metrics['fired_neurons'].append(total_fired)
        self.metrics['burst_spikes'].append(total_burst_spikes)
        self.metrics['dn_matches'].append(bridge_report.get("dn_matches_found", 0))
        self.metrics['mns_activated'].append(bridge_report.get("mns_activated", 0))
        self.metrics['active_joints'].append(active_joints)
        self.metrics['torque_applied'].append(torque_applied)
        self.metrics['walking_dns'].append(bridge_report.get('walking_dns', 0))
        self.metrics['turning_dns'].append(bridge_report.get('turning_dns', 0))
        self.metrics['gait_phase'].append(gait_mode)

        return {
            'step': self.step_count,
            'time_ms': self.sim_time_ms,
            'fired_neurons': total_fired,
            'burst_spikes': total_burst_spikes,
            'dn_matches': bridge_report.get("dn_matches_found", 0),
            'mns_activated': bridge_report.get("mns_activated", 0),
            'active_joints': active_joints,
            'torque_applied': torque_applied,
            'dn_types': bridge_report.get("dn_types", []),
            'gait_phase': gait_mode,
            'walking_dns_count': bridge_report.get('walking_dns', 0),
            'turning_dns_count': bridge_report.get('turning_dns', 0),
            'z_height': float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0,
            'on_ground': mechano_data.get('is_on_ground', False),
            'food_distance': float(np.linalg.norm(np.array(self.data.qpos[0:2]) - np.array(self._food_pos[0:2]))) if (len(self.data.qpos) > 1 and hasattr(self, '_food_pos')) else 0,
            'odor_concentration': chemo_conc,
            'wall_distance': vision_data.get('wall_distance', 10.0),
            'has_wall': vision_data.get('has_wall', False),
            'contrast': vision_data.get('contrast', 0.0),
            # M4: Proprioceptive metrics
            'stance_legs': proprio_data.get('stance_legs', []),
            'stance_leg_count': proprio_data.get('stance_leg_count', 0),
            'load_balance': round(proprio_data.get('load_variance', 0.0), 6),
            'proprioceptive_firing': proprio_data.get('proprioceptive_firing', 0),
            'total_leg_load': round(proprio_data.get('total_leg_load', 0.0), 4),
            'mean_leg_load': round(proprio_data.get('mean_leg_load', 0.0), 4),
            'has_slip': bool(proprio_data.get('slips', {})),
            'recovery': proprio_data.get('recovery', {}),
        }

    def get_summary(self) -> Dict:
        return {
            'total_steps': self.step_count,
            'simulated_time_s': self.sim_time_ms / 1000.0,
            'avg_fired_per_step': np.mean(self.metrics['fired_neurons']) if self.metrics['fired_neurons'] else 0,
            'avg_burst_spikes': np.mean(self.metrics['burst_spikes']) if self.metrics['burst_spikes'] else 0,
            'avg_dn_matches': np.mean(self.metrics['dn_matches']) if self.metrics['dn_matches'] else 0,
            'avg_mns_activated': np.mean(self.metrics['mns_activated']) if self.metrics['mns_activated'] else 0,
            'any_torque': any(self.metrics['torque_applied']),
            'torque_steps': sum(1 for t in self.metrics['torque_applied'] if t),
            'torque_step_pct': sum(1 for t in self.metrics['torque_applied'] if t) / max(1, len(self.metrics['torque_applied'])) * 100,
            'active_burst_count': self.burst_injector.active_burst_count,
            'max_z_height': max(self.all_z_heights) if self.all_z_heights else 0,
            'mean_z_height': np.mean(self.all_z_heights) if self.all_z_heights else 0,
            'ground_contact_pct': np.mean(self.all_ground_contacts) * 100 if self.all_ground_contacts else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE BUILDER
# ═══════════════════════════════════════════════════════════════════════════
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
        neurons.append(NeuronBase(
            neuron_id=i, model=NeuronModel.IF,
            leak_rate=0.03, refractory_delay=1,
            label=f"fw_{fw_id}",
        ))

    max_syn = max((c[2] for c in connections), default=1)
    synapses_by_pre = defaultdict(list)
    synapses_by_post = defaultdict(list)

    for pre_id, post_id, syn_count in connections:
        nt_type = neurons_nt.get(pre_id, 'ACH')
        base_weight = NT_WEIGHT_MAP.get(nt_type, 0.0)
        weight = base_weight * syn_count
        weight = weight / math.log(1 + max_syn)
        weight = max(-10.0, min(10.0, weight))
        pre_idx = flywire_to_idx[pre_id]
        post_idx = flywire_to_idx[post_id]
        synapse = Synapse(target_neuron_id=post_idx, source_neuron_id=pre_idx,
                          weight=weight, model=SynapseModel.FIXED)
        synapses_by_pre[pre_idx].append(synapse)
        synapses_by_post[post_idx].append(synapse)

    engine = NeuronArrayBase(neurons=neurons, thread_count=4)
    for pre_idx, syns in synapses_by_pre.items():
        engine.neurons[pre_idx].synapses_out = syns
    for post_idx, syns in synapses_by_post.items():
        engine.neurons[post_idx].synapses_from = syns
    # Resolve synapse object references (CRITICAL for propagation!)
    for pre_idx, syns in synapses_by_pre.items():
        for syn in syns:
            syn.target_neuron = engine.neurons[syn.target_neuron_id]
            syn.source_neuron = engine.neurons[syn.source_neuron_id]

    all_weights = [s.weight for n in engine.neurons for s in n.synapses_out]
    pos_w = sum(1 for w in all_weights if w > 0)
    neg_w = sum(1 for w in all_weights if w < 0)
    print(f"  [engine] {len(all_weights):,} synapses, {pos_w:,} excit, {neg_w:,} inhib, "
          f"mean={np.mean(all_weights):.3f}", flush=True)
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# SIMFLY SERVER — Web Platform (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════
class SimFlyServer:
    """Manages simulation state, frame streaming, and metrics collection."""

    def __init__(self):
        self.loop: Optional[Phase14Loop] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.engine = None
        self.loader = None
        self.decoder = None
        self.bridge = None
        self.model = None
        self.data = None
        self.actuator_map = {}
        self.dn_classifier = None
        self.gait_controller = None
        self.proprio = None
        self.neuron_nt_types: Dict[int, str] = {}
        self.idx_to_flywire: List[int] = []

        # Simulation control
        self._running = False
        self._paused = False
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_lock = threading.Lock()
        self.render_every = RENDER_EVERY_DEFAULT

        # Frame buffer for MJPEG
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._frame_ready = threading.Event()

        # Metrics history for dashboard
        self._metrics_lock = threading.Lock()
        self._metrics_history: deque = deque(maxlen=METRICS_HISTORY_SIZE)
        self._latest_metrics: Dict[str, Any] = {}

        # Neuron firing history for raster
        self._fired_history: deque = deque(maxlen=100)

        # Pipeline initialized flag
        self._initialized = False
        self._init_error: Optional[str] = None

        # Status frame for visible restart
        self._status_jpeg: Optional[bytes] = None
        self._status_message: str = "Initializing..."
        self._status_progress: int = 0

        # O5: Frame recorder for auto-record
        self.recorder = FrameRecorder(max_frames=1800)
        self._recording_enabled = True
        self._status_total: int = 6

        # Stats
        self._total_neurons = 0
        self._total_synapses = 0
        self._dns_loaded = 0

        # Store initial config for reset (preserve --neurons)
        self._initial_neurons = None
        self._initial_gain = None
        self._initial_direct_inject = False

        # Phase 2: Track components separately for hot-reload
        self._burst_injector = None
        self._sensor_engine_idx = {}
        self._connections_cache = None  # for reconfigure without reloading

    # ── Status Frame Generation ────────────────────────────────────────
    def _generate_status_frame(self, message: str = None, progress: int = None,
                                 total: int = None) -> bytes:
        """Generate a JPEG placeholder frame showing pipeline status."""
        if message is not None:
            self._status_message = message
        if progress is not None:
            self._status_progress = progress
        if total is not None:
            self._status_total = total

        img = Image.new('RGB', (640, 480), color=(10, 10, 30))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_body = font_small = font_title

        draw.text((320, 60), "SimFly — Pipeline Restart", fill=(100, 180, 255), font=font_title, anchor="mt")

        bar_x, bar_y, bar_w, bar_h = 120, 200, 400, 30
        draw.rectangle([bar_x-2, bar_y-2, bar_x+bar_w+2, bar_y+bar_h+2], outline=(60, 60, 120), width=2)
        progress = self._status_progress / max(1, self._status_total)
        fill_w = int(bar_w * min(1.0, progress))
        if fill_w > 0:
            draw.rectangle([bar_x, bar_y, bar_x+fill_w, bar_y+bar_h], fill=(0, 150, 80))
        pct_text = f"{int(progress*100)}%"
        draw.text((bar_x + bar_w//2, bar_y + bar_h//2), pct_text, fill=(255, 255, 255), font=font_body, anchor="mm")

        draw.text((320, 260), self._status_message, fill=(200, 200, 200), font=font_body, anchor="mt")

        for i in range(self._status_total):
            x = 200 + i * 48
            y = 310
            color = (0, 180, 80) if i < self._status_progress else (60, 60, 100)
            draw.ellipse([x-8, y-8, x+8, y+8], fill=color)
            step_labels = ["Bridge", "DNs", "Connect", "NIRON", "MuJoCo", "Render"]
            if i < len(step_labels):
                draw.text((x, y+20), step_labels[i], fill=(150, 150, 150), font=font_small, anchor="mt")

        draw.text((320, 420), f"Neurons: {self._total_neurons or '—'}  |  Synapses: {self._total_synapses or '—'}  |  DNs: {self._dns_loaded or '—'}",
                  fill=(120, 120, 160), font=font_small, anchor="mt")

        buf = BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return buf.getvalue()

    # ── Pipeline Initialization ──────────────────────────────────────────
    def initialize(self, num_neurons=DEFAULT_NEURONS, global_gain=DEFAULT_GLOBAL_GAIN,
                   direct_inject=False) -> bool:
        """Load the full pipeline: connectome, engine, bridge, MuJoCo, sensory."""
        try:
            # Store initial args for future resets
            self._initial_neurons = num_neurons
            self._initial_gain = global_gain
            self._initial_direct_inject = direct_inject
            print(f"\n{'='*60}")
            print(f"SimFly Web Platform — Initializing Pipeline")
            print(f"{'='*60}")
            print(f"  Neurons: {num_neurons} | Gain: {global_gain}")
            print(f"  Model: GROUNDED (simfly_grounded.xml)")
            t_total = time.perf_counter()

            # 0. Bridge
            self._status_jpeg = self._generate_status_frame("Loading DN→MN bridge...", 0, 6)
            print("\n[0/6] Loading DN→MN bridge...", flush=True)
            self.bridge = DnMnBridge(
                dn_matches_path=DN_MATCHES_JSON,
                pathways_path='/tmp/simfly_web/dn_mn_pathways_filtered.json',
                min_pathway_confidence=0.0,
            )
            self.bridge.initialize()
            bs = self.bridge.summary()
            self._dns_loaded = bs.get('dn_matches_loaded', 0)
            print(f"  ✅ Bridge: {self._dns_loaded} DNs, {bs.get('unique_mns_loaded', 0):,} MNs", flush=True)
            self._status_jpeg = self._generate_status_frame("Loading DN root IDs...", 1, 6)

            # 1. DN IDs
            print("\n[1/6] Loading DN root IDs...", flush=True)
            with open(MOTOR_MAP_JSON) as f:
                motor_data = json.load(f)
            neurons_info = motor_data.get("neurons", {})
            all_dn_ids: Set[int] = set()
            for root_id_str, info in neurons_info.items():
                if info.get("flow") == "efferent" and info.get("cell_type", "").startswith("DN"):
                    all_dn_ids.add(int(root_id_str))
            with open(DN_MATCHES_JSON) as f:
                dn_matches = json.load(f).get("matches", {})
            matched_dn_ids = {int(k) for k in dn_matches.keys()}
            print(f"  Found {len(all_dn_ids)} FlyWire DNs ({len(all_dn_ids & matched_dn_ids)} matched)", flush=True)
            self._status_jpeg = self._generate_status_frame("Streaming connectome connections...", 2, 6)

            # 2. Stream connections
            print(f"\n[2/6] Streaming connections for {num_neurons}-neuron selection...", flush=True)
            syn_counter = Counter()
            all_connections = []
            all_nts: Dict[int, str] = {}
            t0 = time.perf_counter()
            with gzip.open(CONNECTIONS_CSV, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pre = int(row['pre_root_id'])
                    post = int(row['post_root_id'])
                    syn = int(row['syn_count'])
                    nt = row['nt_type']
                    all_nts[pre] = all_nts.get(pre, nt)
                    all_nts[post] = all_nts.get(post, nt)
                    syn_counter[pre] += syn
                    syn_counter[post] += syn
                    all_connections.append((pre, post, syn, nt))
            t1 = time.perf_counter()
            print(f"  Scanned {len(all_connections):,} connections in {t1-t0:.1f}s", flush=True)

            if num_neurons <= 0:
                # Use ALL neurons
                included_ids = set(syn_counter.keys())
                print(f"  Using all {len(included_ids)} neurons (no limit)", flush=True)
            else:
                max_dns = min(500, len(matched_dn_ids & all_dn_ids))
                dn_candidates = [(nid, syn_counter.get(nid, 0)) for nid in (matched_dn_ids & all_dn_ids)]
                dn_candidates.sort(key=lambda x: x[1], reverse=True)
                included_dns = {nid for nid, _ in dn_candidates[:max_dns]}
                included_ids = set(included_dns)
                for nid, count in syn_counter.most_common():
                    if nid in included_ids:
                        continue
                    included_ids.add(nid)
                    if len(included_ids) >= num_neurons:
                        break

            actual_neurons = len(included_ids)
            print(f"  Selected: {actual_neurons} neurons", flush=True)

            # 3. Engine
            self._status_jpeg = self._generate_status_frame("Building NIRON engine...", 3, 6)
            print(f"\n[3/6] Building NIRON engine...", flush=True)
            self.neuron_nt_types = {nid: all_nts.get(nid, 'unknown') for nid in included_ids}
            connections = [(pre, post, syn) for pre, post, syn, _ in all_connections
                           if pre in included_ids and post in included_ids]
            # Cache for reconfigure
            self._connections_cache = all_connections
            self._included_ids_cache = included_ids
            config = {
                'min_syn_count': 0, 'leak_rate': 0.03, 'refractory_delay': 1,
                'thread_count': 4, 'normalize_weights': True, 'model': NeuronModel.IF,
            }
            self.loader = FlyWireConnectomeLoader(CONNECTIONS_CSV, config=config)
            self.engine = build_engine(self.neuron_nt_types, connections, self.loader)
            self._total_neurons = self.engine.array_size
            self._total_synapses = sum(len(n.synapses_out) for n in self.engine.neurons)
            t2 = time.perf_counter()
            print(f"  ✅ Engine: {self._total_neurons} neurons, {self._total_synapses:,} synapses ({t2-t1:.1f}s)", flush=True)

            # 4. Sensory
            self._status_jpeg = self._generate_status_frame("Identifying sensory neurons...", 4, 6)
            print(f"\n[4/6] Identifying sensory neurons...", flush=True)
            sensory_map = identify_sensory_neurons(CONNECTIONS_CSV, included_ids, max_sensory=500)
            sensor_engine_idx = {'visual': set(), 'lc4': set(), 'mechano': set(), 'chemo': set()}
            for fw_id in sensory_map['visual_input']:
                eng_idx = self.loader.flywire_to_idx.get(fw_id)
                if eng_idx is not None:
                    sensor_engine_idx['visual'].add(eng_idx)
            vids = sensory_map['visual_input']
            lc4_start = len(vids) // 2
            for fw_id in vids[lc4_start:]:
                eng_idx = self.loader.flywire_to_idx.get(fw_id)
                if eng_idx is not None:
                    sensor_engine_idx['lc4'].add(eng_idx)
            for fw_id in sensory_map['mechano_input']:
                eng_idx = self.loader.flywire_to_idx.get(fw_id)
                if eng_idx is not None:
                    sensor_engine_idx['mechano'].add(eng_idx)
            for fw_id in sensory_map['chemo_input']:
                eng_idx = self.loader.flywire_to_idx.get(fw_id)
                if eng_idx is not None:
                    sensor_engine_idx['chemo'].add(eng_idx)
            self._sensor_engine_idx = sensor_engine_idx
            print(f"  ✅ Sensors ready", flush=True)

            # 5. VNC Decoder + MuJoCo
            self._status_jpeg = self._generate_status_frame("Loading MuJoCo physics model...", 5, 6)
            print("\n[5/6] Loading VNC decoder + MuJoCo model...", flush=True)
            self.decoder = VNCMotorDecoder.load_from_vnc(
                vnc_actuator_map_path=os.path.join(VNC_DIR, "vnc_actuator_map.json"),
                pathways_path='/tmp/simfly_web/dn_mn_pathways_filtered.json',
                tau_decay=50.0,
                global_gain=global_gain,
                dt_brain_ms=1.0,
                dt_physics_ms=5.0,
            )
            print(f"  ✅ Decoder: {self.decoder.summary()['total_joints']} joints, gain={global_gain}", flush=True)

            # M3: DN subtype classifier
            self.dn_classifier = DNSubtypeClassifier(DN_MATCHES_JSON)

            self.model = mujoco.MjModel.from_xml_path(SIMFLY_XML)
            self.data = mujoco.MjData(self.model)

            self._food_pos = (0.07, 0.07, 0.005)
            ARENA_HALF = 0.1
            self._arena_viz = ArenaVisualizer(food_pos=self._food_pos[:2], sigma=0.03, arena_half=ARENA_HALF)
            self._arena_loaded = True
            print(f"  M1: Arena overlay loaded - food marker + gradient", flush=True)
            if self.model.nq >= 7:
                self.data.qpos[0] = 0.0
                self.data.qpos[1] = 0.0
                self.data.qpos[2] = 0.06
                self.data.qpos[3] = 1.0
                self.data.qpos[4] = 0.0
                self.data.qpos[5] = 0.0
                self.data.qpos[6] = 0.0

            print(f"  Settling fly (5000 steps)...", flush=True)
            
            # ── O2: Mass calibration to biological 1mg ──
            fly_body_names = set()
            for i in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and not any(x in (name or '') for x in ['wall', 'corner', 'floor', 'food']):
                    fly_body_names.add(i)
            
            total_fly_mass = sum(self.model.body_mass[i] for i in fly_body_names)
            MASS_SCALE = 1e-6 / total_fly_mass if total_fly_mass > 0 else 1.0
            for i in fly_body_names:
                self.model.body_mass[i] *= MASS_SCALE
                self.model.body_inertia[i][:] *= MASS_SCALE
            new_mass_mg = sum(self.model.body_mass[i] for i in fly_body_names) * 1e6
            print(f"  O2: Mass calibrated {total_fly_mass*1e6:.1f}mg → {new_mass_mg:.2f}mg (scale={MASS_SCALE:.6f})", flush=True)
            
            # Compensate global_gain for lighter body
            gain_compensation = MASS_SCALE ** 0.5
            self.decoder.global_gain *= gain_compensation
            print(f"  O2: Gain compensated: {global_gain} → {self.decoder.global_gain:.8f}", flush=True)
            for i in range(10000):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

            self.actuator_map = {}
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                self.actuator_map[name or f"act_{i}"] = i

            z_settled = float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0
            print(f"  ✅ MuJoCo: {self.model.nbody} bodies, {self.model.nu} actuators, z={z_settled:.3f}m", flush=True)
            self.idx_to_flywire = self.loader.idx_to_flywire

            # M3: Gait controller (tripod gait state machine)
            self.gait_controller = GaitController(self.actuator_map, gait_freq_hz=5.0)

            # M4: Proprioceptive feedback
            self.proprio = ProprioceptiveFeedback(
                self.model, self.data, self.actuator_map,
                contact_threshold=0.0005,
                load_balance_gain=0.3,
                slip_compensate_gain=0.4,
            )
            print(f"  M4: Proprioceptive feedback — {self.proprio.summary()['num_leg_joints']} leg joints, "
                  f"{self.proprio.summary()['num_legs']} legs", flush=True)

            # 6. Sensory systems + Loop
            print("\n[6/6] Initializing sensory + loop...", flush=True)
            vision = FlyVision(self.model, self.data, num_rays=20)
            chemo = ChemoSensorySystem(sugar_source_pos=(0.07, 0.07, 0.0), sugar_sigma=0.03)
            mechano = MechanoSensorySystem(self.model, self.data)

            burst_injector = BurstInjector(
                engine=self.engine, spikes_per_burst=5, isi_ms=1.0,
                min_burst_interval_ms=10.0, charge_per_spike=2.0,
            )
            self._burst_injector = burst_injector

            all_sensor_indices = set()
            for k in ['visual', 'lc4', 'mechano', 'chemo']:
                all_sensor_indices |= sensor_engine_idx.get(k, set())
            direct_inj = DirectSensoryInjector(
                engine=self.engine, burst_injector=burst_injector,
                sensor_engine_indices=all_sensor_indices, rate_per_step=15,
            ) if direct_inject else None

            self.loop = Phase14Loop(
                engine=self.engine, sensor_engine_idx=sensor_engine_idx,
                bridge=self.bridge, vnc_decoder=self.decoder, loader=self.loader,
                model=self.model, data=self.data, actuator_map=self.actuator_map,
                vision=vision, chemo=chemo, mechano=mechano,
                burst_injector=burst_injector, direct_injector=direct_inj,
                brain_rate_hz=1000, physics_rate_hz=200,
                food_pos=self._food_pos,
                dn_classifier=self.dn_classifier,
                gait_controller=self.gait_controller,
                proprio=self.proprio,
            )

            os.environ.setdefault('MUJOCO_GL', 'egl')
            print(f"  ✅ Renderer: {RENDER_WIDTH}x{RENDER_HEIGHT} (EGL headless, deferred)", flush=True)

            total_elapsed = time.perf_counter() - t_total
            self._status_jpeg = self._generate_status_frame("✅ Pipeline ready!", 6, 6)
            print(f"\n{'='*60}")
            print(f"✅ Pipeline initialized in {total_elapsed:.1f}s")
            print(f"   Neurons: {self._total_neurons} | Synapses: {self._total_synapses:,}")
            print(f"   DNs loaded: {self._dns_loaded}")
            print(f"   Ready for streaming!")
            print(f"{'='*60}\n", flush=True)

            self._initialized = True
            return True

        except Exception as e:
            self._init_error = f"{e}\n{traceback.format_exc()}"
            print(f"\n❌ INITIALIZATION FAILED: {self._init_error}", flush=True)
            return False

    # ── Phase 2: Live Reconfigure ──────────────────────────────────────
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration changes live without full pipeline restart.

        Supported keys:
          - gain (float): Update decoder global gain
          - render_every (int): Update render interval
          - burst_rate (int): Update direct injector rate
          - reload_bridge (bool): Reinitialize DN→MN bridge
          - reload_sensory (bool): Reload sensory modules
          - reload_engine (bool): Reinitialize engine only
        """
        result = {'success': True, 'applied': [], 'errors': []}

        # ── Numeric Parameters ─────────────────────────────────────────
        if 'gain' in config:
            try:
                new_gain = float(config['gain'])
                if self.decoder is not None:
                    self.decoder.global_gain = new_gain
                result['applied'].append(f'gain={new_gain}')
                print(f"[CONFIG] Gain set to {new_gain}", flush=True)
            except Exception as e:
                result['errors'].append(f'gain: {e}')

        if 'render_every' in config:
            try:
                self.render_every = max(1, min(50, int(config['render_every'])))
                result['applied'].append(f'render_every={self.render_every}')
                print(f"[CONFIG] Render every {self.render_every} steps", flush=True)
            except Exception as e:
                result['errors'].append(f'render_every: {e}')

        if 'burst_rate' in config:
            try:
                new_rate = int(config['burst_rate'])
                if self.loop is not None and self.loop.direct_injector is not None:
                    self.loop.direct_injector.rate_per_step = new_rate
                result['applied'].append(f'burst_rate={new_rate}')
                print(f"[CONFIG] Burst rate set to {new_rate}", flush=True)
            except Exception as e:
                result['errors'].append(f'burst_rate: {e}')

        # ── Component Reloads ──────────────────────────────────────────
        if config.get('reload_bridge'):
            try:
                print("[CONFIG] Reloading DN→MN bridge...", flush=True)
                from phase8_integration.dn_mn_bridge import DnMnBridge as BridgeClass
                importlib.reload(sys.modules.get('phase8_integration.dn_mn_bridge'))
                self.bridge = BridgeClass(
                    dn_matches_path=DN_MATCHES_JSON,
                    pathways_path='/tmp/simfly_web/dn_mn_pathways_filtered.json',
                    min_pathway_confidence=0.0,
                )
                self.bridge.initialize()
                if self.loop is not None:
                    self.loop.bridge = self.bridge
                bs = self.bridge.summary()
                self._dns_loaded = bs.get('dn_matches_loaded', 0)
                result['applied'].append('reload_bridge')
                print(f"[CONFIG] ✅ Bridge reloaded: {self._dns_loaded} DNs", flush=True)
            except Exception as e:
                result['errors'].append(f'reload_bridge: {e}')
                print(f"[CONFIG] ❌ Bridge reload failed: {e}", flush=True)

        if config.get('reload_sensory'):
            try:
                print("[CONFIG] Reloading sensory modules...", flush=True)
                for mod_name in ['sensory_injector', 'vision', 'chemo', 'mechano']:
                    if mod_name in sys.modules:
                        importlib.reload(sys.modules[mod_name])
                result['applied'].append('reload_sensory')
                print("[CONFIG] ✅ Sensory modules reloaded", flush=True)
            except Exception as e:
                result['errors'].append(f'reload_sensory: {e}')
                print(f"[CONFIG] ❌ Sensory reload failed: {e}", flush=True)

        if config.get('reload_engine'):
            try:
                print("[CONFIG] Reloading engine module...", flush=True)
                from neuron_engine.engine import NeuronArrayBase as EngineClass
                importlib.reload(sys.modules.get('neuron_engine.engine'))
                result['applied'].append('reload_engine')
                print("[CONFIG] ✅ Engine module reloaded (full rebuild may be needed)", flush=True)
            except Exception as e:
                result['errors'].append(f'reload_engine: {e}')
                print(f"[CONFIG] ❌ Engine reload failed: {e}", flush=True)

        result['current_state'] = {
            'gain': self.decoder.global_gain if self.decoder else None,
            'render_every': self.render_every,
            'initialized': self._initialized,
            'neurons': self._total_neurons,
            'dns_loaded': self._dns_loaded,
        }
        return result

    # ── Phase 2: Connectome Stats ───────────────────────────────────────
    def get_connectome_stats(self) -> Dict[str, Any]:
        """Compute comprehensive connectome statistics."""
        stats = {
            'total_neurons': self._total_neurons,
            'total_synapses': self._total_synapses,
            'dns_loaded': self._dns_loaded,
            'connectome_source': 'FlyWire v783',
        }

        # NT breakdown from neuron_nt_types
        nt_breakdown = Counter()
        for fw_id, nt in self.neuron_nt_types.items():
            nt_breakdown[nt] += 1
        stats['nt_breakdown'] = dict(nt_breakdown)

        # Synapse stats from engine
        excitatory = 0
        inhibitory = 0
        total_syn_count = 0
        if self.engine is not None:
            for neuron in self.engine.neurons:
                for syn in neuron.synapses_out:
                    total_syn_count += 1
                    if syn.weight > 0:
                        excitatory += 1
                    elif syn.weight < 0:
                        inhibitory += 1

        stats['excitatory_synapses'] = excitatory
        stats['inhibitory_synapses'] = inhibitory
        stats['synapses_per_neuron_avg'] = round(
            total_syn_count / max(1, self._total_neurons), 1
        )

        # Unique MNs from bridge
        if self.bridge is not None:
            try:
                bs = self.bridge.summary()
                stats['unique_mns'] = bs.get('unique_mns_loaded', 0)
            except Exception:
                stats['unique_mns'] = 0
        else:
            stats['unique_mns'] = 0

        # O4: DN subtype classification from pathways
        if self.dn_classifier is not None:
            try:
                dn_stats = self.dn_classifier.summary()
                subtypes = dn_stats.get('subtypes', {})
                stats['dn_subtype_counts'] = subtypes
                total_cls = sum(subtypes.values())
                stats['dn_classified_total'] = total_cls
                unknown_n = subtypes.get('unknown', 0)
                cov = round((total_cls - unknown_n) / max(1, total_cls) * 100, 1)
                stats['dn_coverage_pct'] = cov
                stats['dn_classification_accuracy'] = f'{cov}% prefix coverage'
            except Exception:
                stats['dn_subtype_counts'] = {}
                stats['dn_classified_total'] = 0
                stats['dn_classification_accuracy'] = 'unavailable'
        
        # O2: Body mass info
        try:
            fly_mass = 0.0
            if self.model is not None:
                for i in range(self.model.nbody):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if name and not any(x in (name or '') for x in ['wall', 'corner', 'floor', 'food']):
                        fly_mass += self.model.body_mass[i]
                stats['body_mass_mg'] = round(fly_mass * 1e6, 3)
                stats['body_mass_biological'] = 1.0
        except Exception:
            stats['body_mass_mg'] = 'unknown'

        # Pathways total from bridge pathways file
        try:
            with open('/tmp/simfly_web/dn_mn_pathways_filtered.json') as f:
                pathways_data = json.load(f)
            all_pathways = pathways_data.get('pathways', {})
            total_paths = sum(len(v) for v in all_pathways.values()) if isinstance(all_pathways, dict) else 0
            stats['pathways_total'] = total_paths
        except Exception:
            stats['pathways_total'] = 0

        return stats

    # ── Simulation Thread ────────────────────────────────────────────────
    def _render_frame(self) -> bytes:
        """Render current MuJoCo scene to JPEG bytes."""
        try:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, RENDER_WIDTH, RENDER_HEIGHT)
                print("[SIM] EGL renderer created in sim thread", flush=True)
            self.renderer.update_scene(self.data, camera="track1")
            pixels = self.renderer.render()
            # M1: Arena overlay (odor gradient + food marker + walls)
            try:
                if hasattr(self, '_arena_viz'):
                    fly_pos = (float(self.data.qpos[0]), float(self.data.qpos[1]))
                    pixels = self._arena_viz.render_overlay(pixels, fly_pos)
            except Exception:
                pass
            img = Image.fromarray(pixels)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=85)
            return buf.getvalue()
        except Exception as e:
            if "EGL" not in str(e):
                print(f"  [render error] {e}", flush=True)
            return self._latest_jpeg or b''

    def _simulation_loop(self):
        """Main simulation thread — runs continuously, pushes frames + metrics."""
        print("[SIM] Simulation thread started", flush=True)
        t_start = time.perf_counter()

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            try:
                with self._sim_lock:
                    report = self.loop.step()
            except Exception as e:
                print(f"[SIM] Step error: {e}", flush=True)
                traceback.print_exc()
                time.sleep(0.1)
                continue

            step = report['step']

            # Render frame
            if step % self.render_every == 0:
                try:
                    jpeg = self._render_frame()
                    with self._frame_lock:
                        self._latest_jpeg = jpeg
                    self._frame_ready.set()
                    # O5: Record frame to circular buffer
                    if self._recording_enabled:
                        self.recorder.add_frame(jpeg)
                except Exception as e:
                    if step <= 5:
                        print(f"[SIM] Frame {step} error: {e}", flush=True)

            # O5: Auto-record check (when fly approaches food)
            if self._recording_enabled and self.recorder._auto_threshold is not None:
                food_dist = report.get('food_distance', 1.0)
                triggered_file = self.recorder.check_auto_trigger(food_dist)
                if triggered_file:
                    print(f"[RECORD] Auto-saved: {triggered_file}", flush=True)
            
            # Store metrics
            with self._metrics_lock:
                self._latest_metrics = report
                self._metrics_history.append(report)

            rtf = (self.loop.sim_time_ms / 1000.0) / max(0.001, time.perf_counter() - t_start)
            time.sleep(0.001)

        print("[SIM] Simulation thread stopped", flush=True)

    # ── Control Methods ──────────────────────────────────────────────────
    def start(self) -> bool:
        if not self._initialized:
            return False
        if self._running and not self._paused:
            return True
        if self._paused:
            self._paused = False
            return True
        self._running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()
        return True

    def pause(self) -> bool:
        self._paused = True
        return True

    def resume(self) -> bool:
        self._paused = False
        return True

    def stop(self):
        self._running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=3.0)

    def reset(self) -> bool:
        """Reset simulation to initial state (requires re-initialization)."""
        was_running = self._running
        self.stop()
        time.sleep(0.5)
        # Clear live frame, show status
        with self._frame_lock:
            self._latest_jpeg = None
        self._initialized = False
        self._status_jpeg = self._generate_status_frame("Restarting pipeline...", 0, 6)
        # Reinitialize with original args
        neurons = self._initial_neurons if self._initial_neurons is not None else DEFAULT_NEURONS
        gain = self._initial_gain if self._initial_gain is not None else DEFAULT_GLOBAL_GAIN
        di = self._initial_direct_inject if self._initial_direct_inject is not None else False
        print(f'  [reset] Using stored config: neurons={neurons}, gain={gain}')
        success = self.initialize(num_neurons=neurons, global_gain=gain, direct_inject=di)
        if success:
            # Clear status frame - live frames take over
            self._status_jpeg = None
            if was_running:
                self.start()
        return success

    def set_speed(self, render_every: int):
        self.render_every = max(1, min(50, render_every))

    @property
    def status(self) -> Dict:
        return {
            'initialized': self._initialized,
            'running': self._running and not self._paused,
            'paused': self._paused,
            'neurons': self._total_neurons,
            'synapses': self._total_synapses,
            'dns_loaded': self._dns_loaded,
            'step': self._latest_metrics.get('step', 0) if self._latest_metrics else 0,
            'error': self._init_error,
        }

    @property
    def latest_metrics(self) -> Dict:
        with self._metrics_lock:
            return dict(self._latest_metrics) if self._latest_metrics else {}

    @property
    def metrics_history(self) -> List[Dict]:
        with self._metrics_lock:
            return list(self._metrics_history)

    @property
    def latest_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg


# ═══════════════════════════════════════════════════════════════════════════
# FLASK + SOCKET.IO APP
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config['SECRET_KEY'] = 'simfly-web-phase2'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# Global instances
sim_server = SimFlyServer()
test_manager = TestManager()


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint - shows live frames or pipeline status during reset."""
    def generate():
        while True:
            frame = sim_server.latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            elif sim_server._status_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + sim_server._status_jpeg + b'\r\n')
                time.sleep(0.5)
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    status = sim_server.status
    metrics = sim_server.latest_metrics
    return jsonify({**status, 'latest_metrics': metrics})


@app.route('/api/start', methods=['POST'])
def api_start():
    ok = sim_server.start()
    return jsonify({'success': ok, 'status': sim_server.status})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    ok = sim_server.pause()
    return jsonify({'success': ok, 'status': sim_server.status})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    ok = sim_server.reset()
    return jsonify({'success': ok, 'status': sim_server.status})


@app.route('/api/reset_progress')
def api_reset_progress():
    """Get current reset/initialization progress."""
    return jsonify({
        'initialized': sim_server._initialized,
        'in_progress': sim_server._status_jpeg is not None and not sim_server._initialized,
        'message': sim_server._status_message,
        'progress': sim_server._status_progress,
        'total': sim_server._status_total,
    })


@app.route('/api/minimap/toggle', methods=['POST'])
def api_minimap_toggle():
    if sim_server._initialized and hasattr(sim_server, '_arena_viz'):
        sim_server._arena_viz.show_minimap = not sim_server._arena_viz.show_minimap
        return jsonify({'success': True, 'minimap_visible': sim_server._arena_viz.show_minimap})
    return jsonify({'success': False, 'error': 'Not initialized'}), 503

@app.route('/api/speed', methods=['POST'])
def api_speed():
    data = request.get_json() or {}
    render_every = data.get('render_every', RENDER_EVERY_DEFAULT)
    sim_server.set_speed(render_every)
    return jsonify({'success': True, 'render_every': sim_server.render_every})


# ── M4: Proprioceptive Feedback Endpoints ─────────────────────────────
@app.route('/api/perturb', methods=['POST'])
def api_perturb():
    """Apply a perturbation to a random leg joint for stumble recovery testing.
    
    Request body: {"joint": "random", "magnitude": 0.01}
    Returns perturbation details + recovery metrics.
    """
    if not sim_server._initialized:
        return jsonify({'success': False, 'error': 'Pipeline not initialized'}), 503
    if not sim_server.proprio:
        return jsonify({'success': False, 'error': 'Proprioception not initialized'}), 503
    
    data = request.get_json() or {}
    joint = data.get('joint', 'random')
    magnitude = float(data.get('magnitude', 0.01))
    
    # Apply perturbation
    result = sim_server.proprio.apply_perturbation(joint=joint, magnitude=magnitude)
    
    # Include current recovery state
    recovery = sim_server.proprio.get_recovery_metrics()
    result['recovery'] = recovery
    
    # Add current load balance for context
    metrics = sim_server.latest_metrics
    if metrics:
        result['current_load_balance'] = metrics.get('load_balance', 0)
        result['stance_legs'] = metrics.get('stance_legs', [])
    
    return jsonify(result)


@app.route('/api/proprio', methods=['GET'])
def api_proprio():
    """Get detailed proprioceptive state."""
    if not sim_server._initialized:
        return jsonify({'success': False, 'error': 'Pipeline not initialized'}), 503
    if not sim_server.proprio:
        return jsonify({'success': False, 'error': 'Proprioception not initialized'}), 503
    
    data = sim_server.proprio.read()
    # Simplify for API — remove large dicts
    data.pop('compensation_torques', None)
    data.pop('firing_rates', None)
    return jsonify({'success': True, 'proprioceptive_state': data})


# ── Phase 2: Live Reconfigure ──────────────────────────────────────────
@app.route('/api/configure', methods=['POST'])
def api_configure():
    """Apply configuration changes live without full pipeline restart."""
    if not sim_server._initialized:
        return jsonify({'success': False, 'error': 'Pipeline not initialized'}), 503
    data = request.get_json() or {}
    result = sim_server.configure(data)
    return jsonify(result)


# ── Phase 2: Connectome Stats ───────────────────────────────────────────
@app.route('/api/connectome_stats')
def api_connectome_stats():
    """Return comprehensive connectome statistics."""
    if not sim_server._initialized:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    stats = sim_server.get_connectome_stats()
    # Add live metrics to stats
    metrics = sim_server.latest_metrics
    if metrics:
        stats['fired_per_step'] = metrics.get('fired_neurons', 0)
        stats['dns_matched_now'] = metrics.get('dn_matches', 0)
        stats['mns_active_now'] = metrics.get('mns_activated', 0)
    # Add model info
    if sim_server.model is not None:
        stats['joint_count'] = sim_server.model.nu
        stats['body_count'] = sim_server.model.nbody
    stats['is_grounded'] = True
    stats['has_arena_walls'] = False
    return jsonify(stats)


# ── Phase 2: Test Runner ────────────────────────────────────────────────
@app.route('/api/test', methods=['POST'])
def api_queue_test():
    """Queue a test configuration."""
    data = request.get_json() or {}
    name = data.get('name', f'auto-{int(time.time())}')
    config = data.get('config', {})
    duration = int(data.get('duration_seconds', 30))
    test_id = test_manager.queue(name, config, duration)
    # Start test runner if not running
    if sim_server._initialized:
        test_manager.start_runner(sim_server)
    return jsonify({
        'success': True,
        'test_id': test_id,
        'message': f'Test "{name}" queued for {duration}s',
    })


@app.route('/api/tests')
def api_list_tests():
    """List all queued/running/completed tests."""
    tests = test_manager.list_tests()
    return jsonify({
        'tests': tests,
        'count': len(tests),
        'queued': sum(1 for t in tests if t['status'] == 'queued'),
        'running': sum(1 for t in tests if t['status'] == 'running'),
        'completed': sum(1 for t in tests if t['status'] == 'completed'),
        'failed': sum(1 for t in tests if t['status'] == 'failed'),
    })


# ── Phase 2: Data Export ────────────────────────────────────────────────
@app.route('/api/export')
def api_export():
    """Export metrics history as CSV or JSON for offline analysis."""
    fmt = request.args.get('format', 'json').lower()
    history = sim_server.metrics_history

    if not history:
        return jsonify({'error': 'No metrics data available'}), 404

    if fmt == 'csv':
        # Build CSV
        output = StringIO()
        fieldnames = ['step', 'time_ms', 'fired_neurons', 'burst_spikes',
                      'dn_matches', 'mns_activated', 'active_joints',
                      'torque_applied', 'z_height', 'on_ground',
                      'wall_distance', 'has_wall', 'contrast']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in history:
            writer.writerow(row)
        csv_data = output.getvalue()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=simfly_metrics_{timestamp}.csv'}
        )
    else:
        # JSON export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return Response(
            json.dumps({
                'export_time': datetime.now().isoformat(),
                'metrics_count': len(history),
                'metrics': history,
            }, default=str),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename=simfly_metrics_{timestamp}.json'}
        )


# ── O5: Recording Endpoints ───────────────────────────────────────────
@app.route('/api/record', methods=['GET', 'POST'])
def api_record():
    """Save last N seconds of video to MP4.
    GET /api/record?duration=30 — save last 30s
    POST /api/record?duration=60 — save last 60s
    """
    if not sim_server._initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 503
    
    if request.method == 'POST':
        data = request.get_json() or {}
        duration = int(data.get('duration', 30))
        label = data.get('label', None)
    else:
        duration = int(request.args.get('duration', 30))
        label = request.args.get('label', None)
    
    duration = max(5, min(300, duration))
    result = sim_server.recorder.save_mp4(duration, label=label)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/record/auto', methods=['POST'])
def api_record_auto():
    """Configure auto-recording trigger.
    POST /api/record/auto?food_threshold=0.05&duration=30
    """
    if not sim_server._initialized:
        return jsonify({'success': False, 'error': 'Not initialized'}), 503
    
    data = request.get_json() or {}
    food_threshold = data.get('food_threshold', None)
    duration = data.get('duration', None)
    
    sim_server.recorder.configure_auto(food_threshold=food_threshold, duration_s=duration)
    
    return jsonify({
        'success': True,
        'auto_threshold': sim_server.recorder._auto_threshold,
        'auto_duration': sim_server.recorder._auto_duration,
        'buffer_frames': sim_server.recorder.frame_count,
        'buffer_duration_s': round(sim_server.recorder.buffer_duration_s, 1),
    })


@app.route('/api/recordings')
def api_recordings():
    """List saved recordings."""
    files = sim_server.recorder.list_recordings()
    return jsonify({
        'recordings': files,
        'count': len(files),
        'record_dir': sim_server.recorder.record_dir,
        'buffer_frames': sim_server.recorder.frame_count,
        'buffer_duration_s': round(sim_server.recorder.buffer_duration_s, 1),
    })


@app.route('/api/record/toggle', methods=['POST'])
def api_record_toggle():
    """Toggle recording on/off."""
    sim_server._recording_enabled = not sim_server._recording_enabled
    return jsonify({
        'success': True,
        'recording_enabled': sim_server._recording_enabled,
    })


@app.route('/api/neurons')
def api_neurons():
    if not sim_server._initialized:
        return jsonify({'neurons': []})
    neurons = []
    for idx, fw_id in enumerate(sim_server.idx_to_flywire[:100]):
        nt = sim_server.neuron_nt_types.get(fw_id, 'unknown')
        neurons.append({'index': idx, 'flywire_id': fw_id, 'nt_type': nt})
    return jsonify({'neurons': neurons, 'total': len(sim_server.idx_to_flywire)})


@app.route('/api/neuron/<int:neuron_id>')
def api_neuron(neuron_id):
    if not sim_server._initialized or not sim_server.engine:
        return jsonify({'error': 'Not initialized'}), 503
    if neuron_id >= len(sim_server.engine.neurons):
        return jsonify({'error': 'Neuron not found'}), 404
    neuron = sim_server.engine.neurons[neuron_id]
    fw_id = sim_server.idx_to_flywire[neuron_id] if neuron_id < len(sim_server.idx_to_flywire) else None
    return jsonify({
        'index': neuron_id,
        'flywire_id': fw_id,
        'nt_type': sim_server.neuron_nt_types.get(fw_id, 'unknown'),
        'model': str(neuron.model),
        'synapses_out': len(neuron.synapses_out),
        'synapses_in': len(neuron.synapses_from),
    })


# ── Socket.IO Events ──────────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    """Send initial status + connectome stats on connect."""
    emit('status', sim_server.status)
    if sim_server._initialized:
        emit('connectome', sim_server.get_connectome_stats())
    if sim_server._running:
        socketio.start_background_task(metrics_emitter)


def metrics_emitter():
    """Background task that emits metrics to connected clients."""
    last_step = -1
    while sim_server._running:
        metrics = sim_server.latest_metrics
        if metrics and metrics.get('step', 0) != last_step:
            last_step = metrics['step']
            payload = {
                'step': metrics.get('step', 0),
                'time_s': metrics.get('time_ms', 0) / 1000.0,
                'z': round(metrics.get('z_height', 0), 4),
                'fired': metrics.get('fired_neurons', 0),
                'dns': metrics.get('dn_matches', 0),
                'mns': metrics.get('mns_activated', 0),
                'torque': metrics.get('torque_applied', False),
                'wall': metrics.get('has_wall', False),
                'on_ground': metrics.get('on_ground', False),
                'food_distance': round(metrics.get('food_distance', 0), 3),
                'odor_concentration': round(metrics.get('odor_concentration', 0), 4),
                'gait_phase': metrics.get('gait_phase', 'stance'),
                'walking_dns': metrics.get('walking_dns_count', 0),
                'turning_dns': metrics.get('turning_dns_count', 0),
                # M4: Proprioceptive metrics
                'stance_legs': metrics.get('stance_leg_count', 0),
                'load_balance': metrics.get('load_balance', 0),
                'proprio_firing': metrics.get('proprioceptive_firing', 0),
            }
            socketio.emit('metrics', payload)
        socketio.sleep(0.05)


@socketio.on('disconnect')
def handle_disconnect():
    pass


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='SimFly Web Platform — Phase 2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--neurons', type=int, default=DEFAULT_NEURONS)
    parser.add_argument('--gain', type=float, default=DEFAULT_GLOBAL_GAIN)
    parser.add_argument('--direct-inject', action='store_true', default=False)
    parser.add_argument('--no-init', action='store_true', help='Skip pipeline init (for testing UI)')
    parser.add_argument('--watch', action='store_true', help='Enable hot-reload file watcher mode')
    args = parser.parse_args()

    os.environ.setdefault('MUJOCO_GL', 'egl')

    # Phase 2: Start file watcher if requested
    watcher = None
    if args.watch:
        watcher = ModuleFileWatcher(server_instance=sim_server)
        watcher.start()

    if not args.no_init:
        print("Initializing SimFly pipeline (this may take 30-60s)...", flush=True)
        ok = sim_server.initialize(
            num_neurons=args.neurons,
            global_gain=args.gain,
            direct_inject=args.direct_inject,
        )
        if not ok:
            print("❌ Pipeline initialization FAILED. Starting server without simulation.", flush=True)

    print(f"\n🚀 SimFly Web Platform (Phase 2) starting on http://{args.host}:{args.port}", flush=True)
    print(f"   Video: http://192.168.1.199:{args.port}/video_feed", flush=True)
    print(f"   Status: http://192.168.1.199:{args.port}/api/status", flush=True)
    print(f"   Connectome: http://192.168.1.199:{args.port}/api/connectome_stats", flush=True)
    if args.watch:
        print(f"   🔄 Hot-reload: ENABLED", flush=True)

    # Auto-start simulation if initialized
    if sim_server._initialized:
        sim_server.start()
        print("   ✅ Simulation auto-started", flush=True)

    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    finally:
        # Cleanup
        if watcher:
            watcher.stop()
        test_manager.stop()


if __name__ == '__main__':
    main()
