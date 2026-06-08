#!/usr/bin/env python3
"""
rl_simfly_pipeline.py — Real Connectome Pipeline for RL Torque Optimization.

Implements the protocol expected by rl_bridge.py's SimFlyRLEnv:
  reset(), get_observation(), get_connectome_torques(),
  apply_torques(), step_physics(), get_state()

This wraps the FULL biological pipeline:
  Sensory → Burst Injector → NIRON engine → DN→MN Bridge → VNC Decoder → Torques
RL modulates ONLY the final torque mapping: modulated[j] = connectome_torque[j] * gain[j] + bias[j]

SCIENTIFIC RIGOR: The connectome still drives ALL movement. RL only calibrates.
"""
from __future__ import annotations
import csv, gzip, json, math, os, sys, time, traceback
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

HOME = os.path.expanduser("~")
CODE_ROOT = os.path.join(HOME, "simrobotics-storage", "research", "flywire", "simfly-robotic-model")
FLYWIRE_DIR = os.path.join(HOME, "simrobotics-storage", "research", "flywire")
SENSORY_DIR = os.path.join(CODE_ROOT, "sensory")

for d in [CODE_ROOT, SENSORY_DIR]:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from neuron_engine.engine import NeuronArrayBase
from neuron_engine.neurons import NeuronBase, NeuronModel
from neuron_engine.synapses import Synapse, SynapseModel
from connectome.connectome_loader import FlyWireConnectomeLoader
from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
from phase8_integration.dn_mn_bridge import DnMnBridge
from vision import FlyVision
from chemo import ChemoSensorySystem
from mechano import MechanoSensorySystem
import mujoco

# Import the RL bridge (same directory)
sys.path.insert(0, CODE_ROOT)
from rl_bridge import RLConfig, apply_modulation

# ── Paths ────────────────────────────────────────────────────────────────
RESEARCH = os.path.join(HOME, "simrobotics-storage", "research")
CONNECTIONS_CSV = os.path.join(RESEARCH, "connections_princeton_no_threshold.csv.gz")
VNC_DIR = os.path.join(CODE_ROOT, "vnc_bridge")
SIMFLY_XML = os.path.join(FLYWIRE_DIR, "virtual-fly", "simfly_model", "simfly_grounded.xml")
DN_MATCHES_JSON = os.path.join(VNC_DIR, "dn_matches.json")
DN_MN_PATHWAYS_JSON = os.path.join(VNC_DIR, "dn_mn_pathways.json")
NT_WEIGHT_MAP: Dict[str, float] = {
    'ACH': 1.5, 'DA': 0.75, 'OCT': 0.75,
    'GABA': -0.5, 'GLUT': -0.25, 'SER': -0.15,
}

# ── Active Joints (36 for RL observation) ─────────────────────────────────
# We track only the 36 leg joints with actuators (skip adhesion, head, wings, antennae)
ACTIVE_LEG_JOINTS = []
for seg in ["T1", "T2", "T3"]:
    for side in ["left", "right"]:
        for joint in ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia"]:
            ACTIVE_LEG_JOINTS.append(f"{joint}_{seg}_{side}")


# ── Burst Injector (same as server) ──────────────────────────────────────
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

    def reset(self):
        self._active_bursts.clear()
        self._last_burst_time.clear()
        self._sim_time_ms = 0.0


# ── SimFlyRLPipeline — implements protocol for SimFlyRLEnv ───────────────
class SimFlyRLPipeline:
    """Wraps the FULL biological simulation pipeline for RL torque calibration.

    Protocol (must match what SimFlyRLEnv expects):
      - cfg: RLConfig
      - joint_names: List[str] (36 active joints)
      - reset() -> None
      - get_observation() -> np.ndarray (obs_dim,)
      - get_connectome_torques() -> Dict[str, float]
      - apply_torques(Dict[str, float]) -> None
      - step_physics() -> None
      - get_state() -> Dict with x_velocity, upright, z_height, food_distance, fell
    """

    def __init__(self, config: RLConfig, max_neurons: int = 0,  # 0 = ALL neurons in connectome
                 food_pos: Tuple[float, float, float] = (8.0, 0.0, 0.0),
                 init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.06)):
        self.cfg = config
        self.joint_names = list(ACTIVE_LEG_JOINTS[:config.n_joints])
        self.max_neurons = max_neurons
        self.food_pos = food_pos
        self.init_pos_base = init_pos

        # Will be populated by initialize()
        self.engine: Optional[NeuronArrayBase] = None
        self.loader: Optional[FlyWireConnectomeLoader] = None
        self.bridge: Optional[DnMnBridge] = None
        self.decoder: Optional[VNCMotorDecoder] = None
        self.model: Optional[Any] = None
        self.data: Optional[Any] = None
        self.actuator_map: Dict[str, int] = {}
        self.vision: Optional[FlyVision] = None
        self.chemo: Optional[ChemoSensorySystem] = None
        self.mechano: Optional[Any] = None
        self.burst_injector: Optional[BurstInjector] = None
        self.sensor_engine_idx: Dict[str, Set[int]] = {}
        self.dn_matches: Dict[str, Any] = {}

        # Runtime state
        self._steps = 0
        self._sim_time_ms = 0.0
        self._initialized = False
        self._last_observation: Optional[np.ndarray] = None
        self._last_connectome_torques: Dict[str, float] = {}
        self._last_applied_torques: Dict[str, float] = {}
        self._prev_pos: Optional[np.ndarray] = None  # for velocity computation

        # Metrics
        self.metrics: Dict[str, List] = defaultdict(list)

    def initialize(self, verbose: bool = True) -> SimFlyRLPipeline:
        """Full initialization: load connectome, build engine, load bridge + decoder."""
        t0 = time.perf_counter()
        if verbose:
            print(f"[RL-PIPELINE] Initializing with {self.max_neurons} neurons, {self.cfg.n_joints} joints...")

        # ── 1. Load DN→MN Bridge ──────────────────────────────────
        self.bridge = DnMnBridge(
            dn_matches_path=DN_MATCHES_JSON,
            pathways_path=DN_MN_PATHWAYS_JSON,
            min_pathway_confidence=0.01,
        )
        self.bridge.initialize(verbose=verbose)
        bs = self.bridge.summary()

        with open(DN_MATCHES_JSON) as f:
            self.dn_matches = json.load(f).get("matches", {})
        matched_dn_ids = {int(k) for k in self.dn_matches.keys()}

        # ── 2. Load Connections + Select Neurons ──────────────────
        if verbose:
            print(f"  [RL-PIPELINE] Streaming connections for neuron selection...")
        syn_counter = Counter()
        all_connections: List[Tuple[int, int, int, str]] = []
        all_nts: Dict[int, str] = {}

        with gzip.open(CONNECTIONS_CSV, 'rt') as f:
            reader = csv.DictReader(f)
            row_count = 0
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
                row_count += 1
                if row_count % 3000000 == 0 and verbose:
                    print(f"    ... {row_count/1e6:.0f}M rows")

        # Include ALL matched DNs from bridge (953 DNs -> 737 MNs)
        # NOTE: dn_matches.json has no 'flow' field; all entries are descending neurons
        all_dn_ids = set(int(k) for k in self.dn_matches.keys())
        included_dns = {nid for nid in (matched_dn_ids & all_dn_ids)}
        print(f"  [RL-PIPELINE] Bridge DNs: {len(all_dn_ids)}, in connectome: {len(included_dns)}",
              flush=True)
        dn_sorted = sorted(included_dns, key=lambda x: syn_counter.get(x, 0), reverse=True)
        # Include ALL DNs (no 500 cap) as starting set
        included_ids = set(dn_sorted)

        # When max_neurons=0: include ALL neurons from connectome
        remaining = self.max_neurons - len(included_ids) if self.max_neurons > 0 else None
        for nid, _ in syn_counter.most_common():
            if nid in included_ids:
                continue
            included_ids.add(nid)
            if remaining is not None and len(included_ids) >= self.max_neurons:
                break

        actual_neurons = len(included_ids)
        if verbose:
            print(f"  [RL-PIPELINE] Selected {actual_neurons} neurons "
                  f"({len(included_ids & all_dn_ids)} DNs)")

        # ── 3. Build NIRON Engine ─────────────────────────────────
        neurons_nt = {nid: all_nts.get(nid, 'unknown') for nid in included_ids}
        connections = [(pre, post, syn) for pre, post, syn, _ in all_connections
                       if pre in included_ids and post in included_ids]
        self.neuron_nt_types = neurons_nt

        config_dict = {'min_syn_count': 0, 'leak_rate': 0.03, 'refractory_delay': 1,
                       'thread_count': 4, 'normalize_weights': True, 'model': NeuronModel.IF}

        self.loader = FlyWireConnectomeLoader(CONNECTIONS_CSV, config=config_dict)
        sorted_ids = sorted(neurons_nt.keys())
        flywire_to_idx = {fw_id: idx for idx, fw_id in enumerate(sorted_ids)}
        idx_to_flywire = sorted_ids
        self.loader.flywire_to_idx = flywire_to_idx
        self.loader.idx_to_flywire = idx_to_flywire
        self.loader.neuron_nt_types = neurons_nt
        self.loader.nt_weight_map = NT_WEIGHT_MAP

        neurons = []
        for i, fw_id in enumerate(idx_to_flywire):
            neurons.append(NeuronBase(
                neuron_id=i, model=NeuronModel.IF, leak_rate=0.03,
                refractory_delay=1, label=f"fw_{fw_id}"))

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

        self.engine = NeuronArrayBase(neurons=neurons, thread_count=4)
        for pre_idx, syns in synapses_by_pre.items():
            self.engine.neurons[pre_idx].synapses_out = syns
        for post_idx, syns in synapses_by_post.items():
            self.engine.neurons[post_idx].synapses_from = syns
        for pre_idx, syns in synapses_by_pre.items():
            for syn in syns:
                syn.target_neuron = self.engine.neurons[syn.target_neuron_id]
                syn.source_neuron = self.engine.neurons[syn.source_neuron_id]

        n_syn = sum(len(n.synapses_out) for n in self.engine.neurons)
        if verbose:
            print(f"  [RL-PIPELINE] Engine: {self.engine.array_size} neurons, {n_syn:,} synapses")

        # ── 4. Identify Sensory Neurons ───────────────────────────
        sensory_map = self._identify_sensory(included_ids, verbose=verbose)
        self.sensor_engine_idx = {'visual': set(), 'lc4': set(), 'mechano': set(), 'chemo': set()}
        for fw_id in sensory_map['visual_input']:
            eng_idx = flywire_to_idx.get(fw_id)
            if eng_idx is not None:
                self.sensor_engine_idx['visual'].add(eng_idx)
        vids = sensory_map['visual_input']
        lc4_start = len(vids) // 2
        for fw_id in vids[lc4_start:]:
            eng_idx = flywire_to_idx.get(fw_id)
            if eng_idx is not None:
                self.sensor_engine_idx['lc4'].add(eng_idx)
        for fw_id in sensory_map['mechano_input']:
            eng_idx = flywire_to_idx.get(fw_id)
            if eng_idx is not None:
                self.sensor_engine_idx['mechano'].add(eng_idx)
        for fw_id in sensory_map['chemo_input']:
            eng_idx = flywire_to_idx.get(fw_id)
            if eng_idx is not None:
                self.sensor_engine_idx['chemo'].add(eng_idx)

        # ── 5. Load VNC Decoder ──────────────────────────────────
        self.decoder = VNCMotorDecoder.load_from_vnc(
            vnc_actuator_map_path=os.path.join(VNC_DIR, "vnc_actuator_map.json"),
            pathways_path=DN_MN_PATHWAYS_JSON,
            tau_decay=50.0, global_gain=0.5, dt_brain_ms=1.0, dt_physics_ms=5.0)

        # ── 6. Load MuJoCo ───────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(SIMFLY_XML)
        self.data = mujoco.MjData(self.model)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.actuator_map[name or f"act_{i}"] = i

        # ── 7. Initialize Sensory Systems + Burst Injector ────────
        self.vision = FlyVision(self.model, self.data, num_rays=20)
        self.chemo = ChemoSensorySystem(
            sugar_source_pos=self.food_pos, sugar_sigma=2.0)
        self.mechano = MechanoSensorySystem(self.model, self.data)
        self.burst_injector = BurstInjector(
            engine=self.engine, spikes_per_burst=5, isi_ms=1.0,
            min_burst_interval_ms=10.0, charge_per_spike=2.0)

        self._initialized = True
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  [RL-PIPELINE] ✅ Initialized in {elapsed:.1f}s")
            print(f"  [RL-PIPELINE] Bridge: {bs['dn_matches_loaded']} DNs, {bs['unique_mns_loaded']:,} MNs")
            print(f"  [RL-PIPELINE] Decoder: {self.decoder.summary()['total_joints']} joints")
            print(f"  [RL-PIPELINE] SimFLy: {self.model.nbody} bodies, {self.model.nu} actuators")
        return self

    def _identify_sensory(self, loaded_neuron_ids, max_sensory=500, verbose=True):
        """Identify input-layer neurons from connectome topology."""
        if verbose:
            print(f"  [RL-PIPELINE] Identifying sensory neurons...")
        in_degree = Counter()
        out_degree = Counter()
        t0 = time.perf_counter()
        with gzip.open(CONNECTIONS_CSV, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pre = int(row['pre_root_id'])
                post = int(row['post_root_id'])
                syn = int(row['syn_count'])
                if pre in loaded_neuron_ids and post in loaded_neuron_ids:
                    out_degree[pre] += syn
                    in_degree[post] += syn
        if verbose:
            print(f"    Scanned in {time.perf_counter() - t0:.1f}s")
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

    # ── Protocol Methods ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Reset MuJoCo
        self.data.qpos[:] = 0
        if len(self.data.qpos) >= 7:
            self.data.qpos[0] = self.init_pos_base[0]
            self.data.qpos[1] = self.init_pos_base[1]
            self.data.qpos[2] = self.init_pos_base[2]
            self.data.qpos[3] = 1.0  # quat w
        self.data.qvel[:] = 0
        mujoco.mj_step(self.model, self.data)

        # Reset engine state (reset all neuron potentials)
        for neuron in self.engine.neurons:
            neuron.reset()
        self.burst_injector.reset()

        # Reset decoder
        self.decoder.reset()

        # Reset runtime state
        self._steps = 0
        self._sim_time_ms = 0.0
        self._last_connectome_torques = {}
        self._last_applied_torques = {}
        self._prev_pos = np.array([self.init_pos_base[0], self.init_pos_base[1], self.init_pos_base[2]])
        self._last_observation = None

        # Initialize observation
        self._last_observation = self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Build observation vector: joint angles + ground contact + COM velocity.

        Returns: np.ndarray of shape (2 * n_joints + 3,) = 75 for n_joints=36
        """
        n = self.cfg.n_joints

        # Joint angles from MuJoCo state
        joint_angles = np.zeros(n)
        for i, jname in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0 and jid < self.model.njnt:
                qpos_addr = self.model.jnt_qposadr[jid]
                if qpos_addr < len(self.data.qpos):
                    joint_angles[i] = float(self.data.qpos[qpos_addr])

        # Ground contact (binary per joint: is foot touching ground?)
        contact = np.zeros(n)
        if hasattr(self.mechano, 'contact_forces'):
            cf = self.mechano.contact_forces
            for i, jname in enumerate(self.joint_names):
                # Derive leg group from joint name
                leg = jname.split('_')[-2] + '_' + jname.split('_')[-1]
                contact[i] = 1.0 if cf.get(leg, 0.0) > 0.001 else 0.0

        # COM velocity
        if len(self.data.qvel) >= 3:
            com_vel = np.array([float(self.data.qvel[0]), float(self.data.qvel[1]), float(self.data.qvel[2])])
        else:
            com_vel = np.zeros(3)

        # Exteroceptive: chemotaxis gradient + wall distance + upright
        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=5.0)
        food_vec = np.array(self.food_pos[:2]) - np.array(self.data.qpos[:2])
        food_dist = float(np.linalg.norm(food_vec))
        food_dir = food_vec / (food_dist + 1e-8) if food_dist > 1e-6 else np.zeros(2)

        # Upright (dot product of z-axis with world up)
        upright = 1.0 if len(self.data.qpos) <= 3 else float(np.clip(
            self.data.qpos[2] / 0.1, 0.0, 1.0))

        extero = np.array([food_dist / 10.0, upright, float(chemo_data.get('sugar_concentration', 0.0))])

        obs = np.concatenate([joint_angles, contact, extero]).astype(np.float32)
        return obs

    def get_connectome_torques(self) -> Dict[str, float]:
        """Run one connectome cycle and return raw torques.

        Flow: Sensory input → burst injector → NIRON fire → DN→MN bridge →
              VNC decoder → raw torque per joint
        """
        if not self._initialized:
            return {}

        vision_data = self.vision.read(dt_ms=5.0)
        chemo_data = self.chemo.read(self.data.qpos[0:3], dt_ms=5.0)
        mechano_data = self.mechano.read()

        # Inject sensory bursts
        self._inject_sensory(vision_data, chemo_data, mechano_data)

        # Run brain sub-steps (5 brain steps per physics step at 1000Hz→200Hz)
        total_fired = 0
        all_fired_engine_indices: Set[int] = set()

        for _ in range(5):
            self.burst_injector.step(1.0)
            fired_count, _ = self.engine.fire()
            total_fired += fired_count

            # Collect fired engine indices from fire_list_2
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
            self._sim_time_ms += 1.0

        # DN→MN bridge
        mn_activations = {}
        if self.bridge is not None and all_fired_engine_indices:
            mn_activations = self.bridge.translate(all_fired_engine_indices, self.loader)

        # VNC Decoder
        if self.decoder is not None:
            self.decoder.accumulate(set(mn_activations.keys()) if mn_activations else set())
            joint_commands = self.decoder.decode()
        else:
            joint_commands = {}

        # Extract torques for active joints only
        torques = {}
        for jname in self.joint_names:
            torques[jname] = joint_commands.get(jname, 0.0)

        self._last_connectome_torques = torques
        self.metrics['fired_neurons'].append(total_fired)
        self.metrics['dn_matches'].append(len(self.bridge.last_fired_dns) if self.bridge else 0)
        self.metrics['mns_activated'].append(len(mn_activations))

        return torques

    def apply_torques(self, torques: Dict[str, float]) -> None:
        """Apply (potentially RL-modulated) torques to MuJoCo actuators."""
        # Reset all ctrl values to prevent stale commands
        self.data.ctrl[:] = 0.0

        for jname, torque in torques.items():
            if jname in self.actuator_map:
                idx = self.actuator_map[jname]
                self.data.ctrl[idx] = float(np.clip(torque, -1.0, 1.0))

        self._last_applied_torques = dict(torques)

    def step_physics(self) -> None:
        """Step MuJoCo physics forward one step."""
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            pass
        self._steps += 1

        # Track metrics
        nz = sum(1 for v in self._last_applied_torques.values() if abs(float(v)) > 0.001)
        self.metrics['active_joints'].append(nz)

    def get_state(self) -> Dict[str, float]:
        """Return simulation state dictionary.

        Keys: x_velocity, upright, z_height, food_distance, wall_min_distance, fell
        """
        if not self._initialized:
            return {'x_velocity': 0.0, 'upright': 0.0, 'z_height': 0.0,
                    'food_distance': 0.0, 'wall_min_distance': 1.0, 'fell': False}

        # Velocity
        if len(self.data.qvel) >= 3 and self._prev_pos is not None:
            curr_pos = np.array([float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])])
            vel = (curr_pos - self._prev_pos) / 0.005  # 5ms physics step
            x_vel = float(vel[0])
            self._prev_pos = curr_pos.copy()
        else:
            x_vel = 0.0

        # Upright (z height normalized)
        z_height = float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0.06
        upright = float(np.clip(z_height / 0.06, 0.0, 2.0)) if z_height > 0 else 0.0

        # Food distance
        food_vec = np.array([self.food_pos[0] - float(self.data.qpos[0]),
                             self.food_pos[1] - float(self.data.qpos[1]),
                             self.food_pos[2] - float(self.data.qpos[2])])
        food_dist = float(np.linalg.norm(food_vec))

        # Check if fell (z below ground significantly)
        fell = z_height < 0.005

        # Wall distance (from vision, if available)
        try:
            vis_data = self.vision.read(dt_ms=5.0)
            wall_dist = float(vis_data.get('wall_distance', 10.0))
        except Exception:
            wall_dist = 10.0

        return {
            'x_velocity': x_vel,
            'upright': upright,
            'z_height': z_height,
            'food_distance': food_dist,
            'wall_min_distance': wall_dist,
            'fell': fell,
        }

    # ── Sensory Injection ────────────────────────────────────────────────

    def _inject_sensory(self, vision_data, chemo_data, mechano_data) -> int:
        """Inject sensory-triggered burst trains into connectome input neurons."""
        total = 0
        contrast = vision_data.get('contrast', 0.0)

        # Visual input
        if contrast > 0.01:
            n_bursts = max(1, int(contrast * 15))
            vis_targets = list(self.sensor_engine_idx.get('visual', set()))
            if vis_targets:
                chosen = np.random.choice(vis_targets, min(n_bursts, len(vis_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst

        # Chemo input
        sugar_conc = chemo_data.get('sugar_concentration', 0.0)
        if sugar_conc > 0.01:
            n_bursts = max(1, int(sugar_conc * 10))
            chemo_targets = list(self.sensor_engine_idx.get('chemo', set()))
            if chemo_targets:
                chosen = np.random.choice(chemo_targets, min(n_bursts, len(chemo_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst

        # Mechano (ground contact)
        if mechano_data.get('is_on_ground', False):
            contact_force = mechano_data.get('total_contact_force', 0.0)
            if contact_force > 0.001:
                n_bursts = min(3, max(1, int(contact_force * 2)))
                mech_targets = list(self.sensor_engine_idx.get('mechano', set()))
                if mech_targets:
                    chosen = np.random.choice(mech_targets, min(n_bursts, len(mech_targets)), replace=False)
                    for nid in chosen:
                        if self.burst_injector.trigger_burst(int(nid)):
                            total += self.burst_injector.spikes_per_burst

        return total

    def close(self) -> None:
        """Clean up resources."""
        pass
