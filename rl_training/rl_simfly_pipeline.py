#!/usr/bin/env python3
"""
rl_simfly_pipeline.py — Real Connectome Pipeline for RL Torque Optimization.
v2: BFS-Upstream neuron selection (sensorimotor pathway tracing).

Implements the protocol expected by rl_bridge.py's SimFlyRLEnv:
  reset(), get_observation(), get_connectome_torques(),
  apply_torques(), step_physics(), get_state()

This wraps the FULL biological pipeline:
  Sensory → Burst Injector → C++ Engine → DN→MN Bridge → VNC Decoder → Torques
RL modulates ONLY the final torque mapping: modulated[j] = connectome_torque[j] * gain[j] + bias[j]

SCIENTIFIC RIGOR: The connectome still drives ALL movement. RL only calibrates.

NEURON SELECTION (v2 — BFS-Upstream):
  When bfs_hops > 0, neurons are selected by tracing upstream from DNs through the
  connectome graph. This captures the complete sensorimotor pathway while excluding
  brain regions (mushroom bodies, central complex) that don't feed motor output.
  
  Selection = DNs ∪ BFS(N hops upstream from DNs) ∪ sensory neurons ∪ motor neurons

  When bfs_hops = 0, falls back to original max_neurons-based selection.
"""
from __future__ import annotations
import csv, gzip, json, math, os, sys, time, traceback
from collections import defaultdict, Counter, deque
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

from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder
from phase8_integration.dn_mn_bridge import DnMnBridge
from vision import FlyVision
from chemo import ChemoSensorySystem
from mechano import MechanoSensorySystem
import mujoco
import cpp_engine  # Linux C++ engine binding

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
ACTIVE_LEG_JOINTS = []
for seg in ["T1", "T2", "T3"]:
    for side in ["left", "right"]:
        for joint in ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia"]:
            ACTIVE_LEG_JOINTS.append(f"{joint}_{seg}_{side}")


# ── Burst Injector (same as server) ──────────────────────────────────────
class BurstInjector:
    def __init__(self, cpp_eng, spikes_per_burst=5, isi_ms=1.0,
                 min_burst_interval_ms=10.0, charge_per_spike=2.0):
        self._cpp = cpp_eng

        self.spikes_per_burst = spikes_per_burst
        self.isi_ms = isi_ms
        self.min_burst_interval_ms = min_burst_interval_ms
        self.charge_per_spike = charge_per_spike
        self._active_bursts: Dict[int, Tuple[int, float]] = {}
        self._last_burst_time: Dict[int, float] = {}
        self._sim_time_ms: float = 0.0

    def trigger_burst(self, neuron_idx: int) -> bool:
        if neuron_idx < 0 or self._cpp is None:
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
                self._cpp.add_charge(neuron_idx, self.charge_per_spike)
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

    Neuron Selection Modes:
      - bfs_hops=N (N>0): BFS upstream from DNs for N hops + sensory + MNs
      - bfs_hops=0 + max_neurons=N: Original most-connected selection (backward compat)
      - bfs_hops=0 + max_neurons=0: ALL neurons (backward compat)
    """

    def __init__(self, config: RLConfig,
                 max_neurons: int = 0,       # 0 = ALL (fallback mode, not recommended)
                 bfs_hops: int = 3,           # 0 = use max_neurons fallback; N>0 = BFS upstream from DNs
                 syn_threshold: int = 5,       # minimum synapse count to include a connection (filters noise)
                 brain_steps: int = 2,        # brain substeps per physics step (was 5, reduced for speed)
                 synaptic_scale: float = 0.005,   # global weight scale to prevent torque saturation
                 food_pos: Tuple[float, float, float] = (2.0, 0.0, 0.0),  # FIX: Moved closer
                 init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.06)):
        self.cfg = config
        self.joint_names = list(ACTIVE_LEG_JOINTS[:config.n_joints])
        self.max_neurons = max_neurons
        self.bfs_hops = bfs_hops
        self.syn_threshold = syn_threshold
        self.brain_steps = brain_steps
        self.synaptic_scale = synaptic_scale
        self.food_pos = food_pos
        self.init_pos_base = init_pos

        # Will be populated by initialize()
        self.cpp_eng: Optional[Any] = None  # CppEngine instance (C++ .so)
        self.loader: Optional[Any] = None
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
        self._prev_pos: Optional[np.ndarray] = None

        # Metrics
        self.metrics: Dict[str, List] = defaultdict(list)

    def initialize(self, verbose: bool = True) -> SimFlyRLPipeline:
        """Full initialization: load connectome, build engine, load bridge + decoder."""
        t0 = time.perf_counter()
        mode_str = f"BFS-{self.bfs_hops}, syn>= {self.syn_threshold}, brain_steps={self.brain_steps}" if self.bfs_hops > 0 else f"max_neurons={self.max_neurons}"
        if verbose:
            print(f"[RL-PIPELINE] Initializing with {mode_str}, {self.cfg.n_joints} joints...")

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

        # Extract MN IDs from the bridge (737 MNs)
        mn_ids: Set[int] = set()
        try:
            raw_matches = json.load(open(DN_MATCHES_JSON)).get("matches", {})
            for dn_str, mn_list in raw_matches.items():
                if isinstance(mn_list, list):
                    for mn in mn_list:
                        if isinstance(mn, (int, float)):
                            mn_ids.add(int(mn))
        except Exception:
            pass
        if verbose:
            print(f"  [RL-PIPELINE] Extracted {len(mn_ids)} MN IDs from bridge")

        # ── 2. Load Connections + Build Graph + Select Neurons ─────
        if verbose:
            print(f"  [RL-PIPELINE] Streaming connections for neuron selection...")
        syn_counter = Counter()
        all_connections: List[Tuple[int, int, int, str]] = []
        all_nts: Dict[int, str] = {}

        # BFS: build reverse adjacency (post → set of pre neurons)
        post_to_pre: Dict[int, Set[int]] = defaultdict(set)

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

                if self.bfs_hops > 0:
                    post_to_pre[post].add(pre)

                row_count += 1
                if row_count % 3000000 == 0 and verbose:
                    if self.bfs_hops > 0:
                        print(f"    ... {row_count/1e6:.0f}M rows | post_to_pre size: {len(post_to_pre):,}")
                    else:
                        print(f"    ... {row_count/1e6:.0f}M rows")

        # ── NEURON SELECTION ──────────────────────────────────────
        all_dn_ids = set(int(k) for k in self.dn_matches.keys())
        included_dns = {nid for nid in (matched_dn_ids & all_dn_ids)}
        print(f"  [RL-PIPELINE] Bridge DNs: {len(all_dn_ids)}, in connectome: {len(included_dns)}",
              flush=True)

        if self.bfs_hops > 0:
            # ── MODE: BFS-Upstream Sensorimotor Pathway ──────────
            included_ids = self._bfs_upstream_from_dns(
                dn_ids=included_dns,
                mn_ids=mn_ids,
                post_to_pre=post_to_pre,
                all_nts=all_nts,
                verbose=verbose,
            )
        else:
            # ── MODE: Original max_neurons-based selection ───────
            dn_sorted = sorted(included_dns, key=lambda x: syn_counter.get(x, 0), reverse=True)
            included_ids = set(dn_sorted)
            remaining = self.max_neurons - len(included_ids) if self.max_neurons > 0 else None
            for nid, _ in syn_counter.most_common():
                if nid in included_ids:
                    continue
                included_ids.add(nid)
                if remaining is not None and len(included_ids) >= self.max_neurons:
                    break

        actual_neurons = len(included_ids)
        dn_count = len(included_ids & all_dn_ids)
        mn_count = len(included_ids & mn_ids)
        if verbose:
            parts = [f"{actual_neurons} neurons"]
            if dn_count > 0:
                parts.append(f"{dn_count} DNs")
            if mn_count > 0:
                parts.append(f"{mn_count} MNs")
            print(f"  [RL-PIPELINE] Selected {', '.join(parts)}")

        # ── 3. Build C++ Engine (libneuronengine.so) ────────────────
        neurons_nt = {nid: all_nts.get(nid, 'unknown') for nid in included_ids}
        connections = [(pre, post, syn) for pre, post, syn, _ in all_connections
                       if pre in included_ids and post in included_ids and syn >= self.syn_threshold]
        self.neuron_nt_types = neurons_nt

        sorted_ids = sorted(neurons_nt.keys())
        flywire_to_idx = {fw_id: idx for idx, fw_id in enumerate(sorted_ids)}
        idx_to_flywire = sorted_ids
        self._flywire_to_idx = flywire_to_idx
        self._idx_to_flywire = idx_to_flywire

        max_syn = max((c[2] for c in connections), default=1)

        # ── C++ Engine (libneuronengine.so) ──
        actual_neurons_count = len(included_ids)
        self.cpp_eng = cpp_engine.CppEngine()
        self.cpp_eng.create(n_neurons=actual_neurons_count, n_threads=4)

        for eng_idx, fw_id in enumerate(idx_to_flywire):
            self.cpp_eng.set_neuron(eng_idx, model=cpp_engine.CppEngine.MODEL_LIF, leak=0.03)

        for pre_id, post_id, syn_count in connections:
            nt_type = neurons_nt.get(pre_id, 'ACH')
            base_weight = NT_WEIGHT_MAP.get(nt_type, 0.0)
            weight = base_weight * syn_count
            weight = weight / math.log(1 + max_syn)
            weight = weight * self.synaptic_scale
            weight = max(-10.0, min(10.0, weight))
            pre_idx = flywire_to_idx[pre_id]
            post_idx = flywire_to_idx[post_id]
            self.cpp_eng.add_synapse(pre_idx, post_idx, weight)

        n_syn = self.cpp_eng.synapse_count()
        if verbose:
            print(f"  [RL-PIPELINE] C++ Engine: {actual_neurons_count} neurons, {n_syn:,} synapses, scale={self.synaptic_scale}")

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
            tau_decay=50.0, global_gain=0.01, dt_brain_ms=1.0, dt_physics_ms=5.0)

        # ── 6. Load MuJoCo ───────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(SIMFLY_XML)
        self.data = mujoco.MjData(self.model)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.actuator_map[name or f"act_{i}"] = i

        # ── 7. Initialize Sensory Systems + Burst Injector ────────
        self.vision = FlyVision(self.model, self.data, num_rays=20, arena_bounds={'x_min': -10.0, 'x_max': 5.0, 'y_min': -5.0, 'y_max': 5.0, 'z_min': -1.0, 'z_max': 5.0}, food_position=self.food_pos)  # FIX
        self.chemo = ChemoSensorySystem(
            sugar_source_pos=self.food_pos, sugar_sigma=4.0)  # FIX: Widened
        self.mechano = MechanoSensorySystem(self.model, self.data)
        self.burst_injector = BurstInjector(
            cpp_eng=self.cpp_eng, spikes_per_burst=5, isi_ms=1.0,
            min_burst_interval_ms=10.0, charge_per_spike=2.0)

        self._initialized = True
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  [RL-PIPELINE] ✅ Initialized in {elapsed:.1f}s")
            print(f"  [RL-PIPELINE] Bridge: {bs['dn_matches_loaded']} DNs, {bs['unique_mns_loaded']:,} MNs")
            print(f"  [RL-PIPELINE] Decoder: {self.decoder.summary()['total_joints']} joints")
            print(f"  [RL-PIPELINE] SimFLy: {self.model.nbody} bodies, {self.model.nu} actuators")
        return self

    def _bfs_upstream_from_dns(
        self,
        dn_ids: Set[int],
        mn_ids: Set[int],
        post_to_pre: Dict[int, Set[int]],
        all_nts: Dict[int, str],
        verbose: bool = True,
    ) -> Set[int]:
        """Trace upstream from DNs through the connectome via BFS.

        Strategy:
          1. Start with all DNs as seeds (the motor output bottleneck)
          2. BFS backwards (post→pre edges) for self.bfs_hops hops
          3. Also include all MNs (motor output)
          4. Also include sensory neurons (indegree=0 in post_to_pre data)
          5. Union = complete sensorimotor pathway

        Args:
            dn_ids: Set of DN FlyWire IDs (already intersected with connectome)
            mn_ids: Set of MN IDs from the DN→MN bridge
            post_to_pre: Reverse adjacency map (post neuron → set of pre neurons)
            all_nts: Neuron ID → neurotransmitter type mapping
            verbose: Print progress

        Returns:
            Set of FlyWire neuron IDs comprising the sensorimotor pathway
        """
        if verbose:
            print(f"  [BFS] Tracing upstream from {len(dn_ids)} DNs for {self.bfs_hops} hops...")

        # Seed: all DNs
        visited: Set[int] = set(dn_ids)
        frontier: Set[int] = set(dn_ids)

        hop_sizes = [len(visited)]
        for hop in range(self.bfs_hops):
            next_frontier: Set[int] = set()
            for neuron_id in frontier:
                pre_set = post_to_pre.get(neuron_id)
                if pre_set is None:
                    continue
                for pre_id in pre_set:
                    if pre_id not in visited:
                        visited.add(pre_id)
                        next_frontier.add(pre_id)
            frontier = next_frontier
            hop_sizes.append(len(visited))
            if verbose:
                print(f"    Hop {hop+1}: +{len(next_frontier):,} new → {len(visited):,} total")

        # Include MNs
        mn_added = 0
        for mn_id in mn_ids:
            if mn_id not in visited:
                visited.add(mn_id)
                mn_added += 1
        if verbose:
            print(f"    MNs added: {mn_added}")

        # Include sensory neurons (indegree=0 in connectome = pure input neurons)
        # These are neurons in visited that have no pre-synaptic connections
        # But ALSO neurons NOT in visited that have indegree=0 (sensory by topology)
        sensory_added = 0
        # We identify sensory from the post_to_pre structure:
        # A neuron with indegree=0 has NO entries in post_to_pre keys AND
        # is not listed as post in any connection
        all_posts = set(post_to_pre.keys())
        all_pres = set()
        for pre_set in post_to_pre.values():
            all_pres.update(pre_set)

        # Pure sensory: neurons that appear as pre but never as post
        pure_sensory = all_pres - all_posts

        # Also include any neurons with indegree=0 from the full neuron set
        # (neurons that appear in all_nts but never as a post target)
        all_known = set(all_nts.keys())
        known_sensory = all_known - all_posts  # neurons with no incoming edges

        # Combine and add to visited
        all_sensory_candidates = pure_sensory | known_sensory
        for sid in all_sensory_candidates:
            if sid not in visited:
                visited.add(sid)
                sensory_added += 1
        if verbose:
            print(f"    Sensory neurons added: {sensory_added} (of {len(all_sensory_candidates):,} candidates)")

        # Final stats
        n_dns = len(visited & dn_ids)
        n_mns = len(visited & mn_ids)
        if verbose:
            print(f"  [BFS] Final selection: {len(visited):,} neurons "
                  f"({n_dns} DNs, {n_mns} MNs, {len(visited)-n_dns-n_mns} interneurons)")
            print(f"  [BFS] Hop sizes: {' → '.join(str(s) for s in hop_sizes)}")

        return visited

    def _get_minimal_loader(self):
        """Return a minimal loader object for bridge.translate() when using C++ engine."""
        # The bridge.translate() needs loader.flywire_to_idx and loader.idx_to_flywire
        # We build them from the sorted neuron IDs
        class MinimalLoader:
            pass
        loader = MinimalLoader()
        loader.flywire_to_idx = self._flywire_to_idx
        loader.idx_to_flywire = self._idx_to_flywire
        return loader

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
        self.cpp_eng.reset()
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

        # Run brain sub-steps (configurable, default 2)
        total_fired = 0
        all_fired_engine_indices: Set[int] = set()

        for _ in range(self.brain_steps):
            self.burst_injector.step(1.0)
            fired = self.cpp_eng.fire()
            total_fired += len(fired)
            all_fired_engine_indices.update(fired)
            self._sim_time_ms += 1.0

        # DN→MN bridge
        mn_activations = {}
        if self.bridge is not None and all_fired_engine_indices:
            mn_activations = self.bridge.translate(all_fired_engine_indices, self._get_minimal_loader())

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
        nz = sum(1 for v in self._last_applied_torques.values() if abs(float(v)) > 0.001)
        self.metrics['active_joints'].append(nz)

    def get_state(self) -> Dict[str, float]:
        """Return simulation state dictionary."""
        if not self._initialized:
            return {'x_velocity': 0.0, 'upright': 0.0, 'z_height': 0.0,
                    'food_distance': 0.0, 'wall_min_distance': 1.0, 'fell': False}

        if len(self.data.qvel) >= 3 and self._prev_pos is not None:
            curr_pos = np.array([float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])])
            vel = (curr_pos - self._prev_pos) / 0.005
            x_vel = float(vel[0])
            self._prev_pos = curr_pos.copy()
        else:
            x_vel = 0.0

        z_height = float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0.06
        upright = float(np.clip(z_height / 0.06, 0.0, 2.0)) if z_height > 0 else 0.0

        food_vec = np.array([self.food_pos[0] - float(self.data.qpos[0]),
                             self.food_pos[1] - float(self.data.qpos[1]),
                             self.food_pos[2] - float(self.data.qpos[2])])
        food_dist = float(np.linalg.norm(food_vec))
        fell = z_height < 0.005

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

        if contrast > 0.01:
            n_bursts = max(1, int(contrast * 15))
            vis_targets = list(self.sensor_engine_idx.get('visual', set()))
            if vis_targets:
                chosen = np.random.choice(vis_targets, min(n_bursts, len(vis_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst

        sugar_conc = chemo_data.get('sugar_concentration', 0.0)
        if sugar_conc > 0.01:
            n_bursts = max(1, int(sugar_conc * 10))
            chemo_targets = list(self.sensor_engine_idx.get('chemo', set()))
            if chemo_targets:
                chosen = np.random.choice(chemo_targets, min(n_bursts, len(chemo_targets)), replace=False)
                for nid in chosen:
                    if self.burst_injector.trigger_burst(int(nid)):
                        total += self.burst_injector.spikes_per_burst

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
