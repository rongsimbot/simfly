#!/usr/bin/env python3
"""
Phase 9: CONNECTOME-DRIVEN SIMULATION — DN→MN Bridge Integration
FlyWire Brain → [DN→MN Bridge] → MANC VNC → SimFLy Body → Sensory Feedback

Phase 9 adds the DN→MN translation bridge that was missing in Phase 8.
Every motor command MUST trace back to real FlyWire connectome DNs firing
→ real MANC DN→IN→MN pathways → real MANC MN torque commands.

Key addition: NIRON engine neurons are now filtered to include descending
neurons (DNs). When DNs fire, the bridge translates to MANC MN activations
via 953 matched DN→MN pathways.

SCIENTIFIC RIGOR: NO scripted motor patterns. If no DNs fire, that's a valid
scientific result — documented, not faked.
"""

import argparse, json, math, os, subprocess, sys, time, traceback
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

# ── Path Configuration ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLYWIRE_DIR = os.path.dirname(SCRIPT_DIR)  # ~/.../flywire/
CODE_ROOT = os.path.join(FLYWIRE_DIR, "simfly-robotic-model")
SENSORY_DIR = os.path.join(FLYWIRE_DIR, "sensory")

# Add code roots (including flywire_dir itself for phase8_integration imports)
for d in [FLYWIRE_DIR, SENSORY_DIR, CODE_ROOT]:  # reverse insert so CODE_ROOT is sys.path[0]
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

print(f"[INIT] CODE_ROOT={CODE_ROOT}", flush=True)
print(f"[INIT] SENSORY_DIR={SENSORY_DIR}", flush=True)

# ── Imports ──────────────────────────────────────────────────────────────
print("[INIT] Importing NIRON engine...", flush=True)
from neuron_engine.neurons import NeuronBase, NeuronModel
from neuron_engine.synapses import Synapse, SynapseModel
from neuron_engine.engine import NeuronArrayBase
print("[INIT] NIRON OK", flush=True)

print("[INIT] Importing connectome loader...", flush=True)
from connectome.connectome_loader import FlyWireConnectomeLoader, DEFAULT_NT_WEIGHT_MAP
print("[INIT] Connectome OK", flush=True)

print("[INIT] Importing VNC decoder...", flush=True)
from vnc_bridge.vnc_motor_decoder import VNCMotorDecoder, MN_TYPE_GAIN
print("[INIT] VNC OK", flush=True)

print("[INIT] Importing DN→MN Bridge...", flush=True)
from phase8_integration.dn_mn_bridge import DnMnBridge
print("[INIT] Bridge OK", flush=True)

print("[INIT] Importing sensory modules...", flush=True)
from sensory_injector import SensoryInjector
from closed_loop import ClosedLoop
print("[INIT] Sensory OK", flush=True)

print("[INIT] Importing MuJoCo (EGL headless)...", flush=True)
import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
print(f"[INIT] MuJoCo {mujoco.__version__} OK", flush=True)

# ── Paths ────────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
RESEARCH = os.path.join(HOME, "simrobotics-storage", "research")
CONNECTIONS_CSV = os.path.join(RESEARCH, "connections_princeton.csv.gz")
VNC_DIR = os.path.join(CODE_ROOT, "vnc_bridge")
SIMFLY_XML = os.path.join(FLYWIRE_DIR, "virtual-fly", "simfly_model", "simfly.xml")
MOTOR_MAP_JSON = os.path.join(CODE_ROOT, "neuron_engine", "motor_neuron_map.json")
DN_MATCHES_JSON = os.path.join(VNC_DIR, "dn_matches.json")
DN_MN_PATHWAYS_JSON = os.path.join(VNC_DIR, "dn_mn_pathways.json")
OUTPUT_DIR = "/tmp/connectome_video"

print(f"[INIT] Connectome CSV: {CONNECTIONS_CSV} ({os.path.getsize(CONNECTIONS_CSV)/1e6:.0f}MB)", flush=True)
print(f"[INIT] VNC data dir: {VNC_DIR}", flush=True)
print(f"[INIT] SimFLy XML: {SIMFLY_XML}", flush=True)
print(f"[INIT] Motor map: {MOTOR_MAP_JSON} ({os.path.getsize(MOTOR_MAP_JSON)/1e6:.1f}MB)", flush=True)

# ── DN loading ───────────────────────────────────────────────────────────
def load_dn_root_ids(motor_map_path: str) -> Set[int]:
    """Load FlyWire root IDs for all descending neurons.
    
    Returns set of integer FlyWire root IDs that are known DNs.
    """
    with open(motor_map_path) as f:
        data = json.load(f)
    
    neurons = data.get("neurons", {})
    dn_ids: Set[int] = set()
    
    for root_id_str, info in neurons.items():
        flow = info.get("flow", "")
        cell_type = info.get("cell_type", "")
        
        if flow == "efferent" and cell_type.startswith("DN"):
            dn_ids.add(int(root_id_str))
    
    return dn_ids


def filter_connections_for_dns(
    csv_path: str,
    dn_ids: Set[int],
    extra_ids: Optional[Set[int]] = None,
    max_total_neurons: int = 500,
) -> Tuple[Dict[int, str], List[Tuple[int, int, int]]]:
    """Load connectome connections ensuring DNs are included.
    
    Streams the connections CSV and collects:
    - All connections where pre OR post is a DN
    - Additional neurons to fill up to max_total_neurons (top by connection count)
    
    Args:
        csv_path: Path to connections CSV.
        dn_ids: Set of FlyWire root IDs that are DNs (must be included).
        extra_ids: Optional set of additional neuron IDs to include.
        max_total_neurons: Maximum total neurons to include.
    
    Returns:
        Tuple of (neurons_nt dict, connections list).
    """
    import gzip, csv
    from collections import Counter
    
    print(f"  [load] Filtering connections for {len(dn_ids)} DNs...")
    
    # Collect all connection counts in one pass (if we need top-N)
    syn_count_per_neuron: Counter = Counter()
    all_connections: List[Tuple[int, int, int, str]] = []
    all_neuron_nts: Dict[int, str] = {}
    
    t0 = time.perf_counter()
    with gzip.open(csv_path, 'rt') as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            pre_id = int(row['pre_root_id'])
            post_id = int(row['post_root_id'])
            syn_count = int(row['syn_count'])
            nt_type = row['nt_type']
            
            all_neuron_nts[pre_id] = nt_type
            all_neuron_nts[post_id] = nt_type
            
            syn_count_per_neuron[pre_id] += syn_count
            syn_count_per_neuron[post_id] += syn_count
            
            all_connections.append((pre_id, post_id, syn_count, nt_type))
            row_count += 1
            
            if row_count % 2000000 == 0:
                print(f"    ... scanned {row_count/1e6:.0f}M rows", flush=True)
    
    t1 = time.perf_counter()
    print(f"  [load] Scanned {row_count:,} rows in {t1-t0:.1f}s", flush=True)
    print(f"  [load] {len(all_neuron_nts):,} unique neurons, {len(all_connections):,} connections", flush=True)
    
    # Determine which neurons to include
    included_ids: Set[int] = set(dn_ids)
    if extra_ids:
        included_ids.update(extra_ids)
    
    # Fill remaining slots with top-connected neurons (excluding DNs already included)
    remaining_slots = max_total_neurons - len(included_ids)
    if remaining_slots > 0:
        for nid, _ in syn_count_per_neuron.most_common():
            if nid in included_ids:
                continue
            included_ids.add(nid)
            if len(included_ids) >= max_total_neurons:
                break
    
    print(f"  [load] Including {len(included_ids)} neurons "
          f"({len(included_ids & dn_ids)} DNs + {len(included_ids - dn_ids)} others)", flush=True)
    
    # Filter connections
    neurons_nt: Dict[int, str] = {}
    connections: List[Tuple[int, int, int]] = []
    
    for pre_id, post_id, syn_count, nt_type in all_connections:
        if pre_id in included_ids and post_id in included_ids:
            neurons_nt[pre_id] = all_neuron_nts.get(pre_id, nt_type)
            neurons_nt[post_id] = all_neuron_nts.get(post_id, nt_type)
            connections.append((pre_id, post_id, syn_count))
    
    dn_connections = sum(1 for pre, post, _ in connections if pre in dn_ids or post in dn_ids)
    print(f"  [load] Filtered: {len(neurons_nt):,} neurons, {len(connections):,} connections "
          f"({dn_connections:,} involve DNs)", flush=True)
    
    return neurons_nt, connections


# ── Modified ClosedLoop with Bridge ──────────────────────────────────────
class BridgeAwareClosedLoop(ClosedLoop):
    """Extended ClosedLoop that routes NIRON output through the DN→MN bridge.
    
    Instead of directly passing fired engine indices to the VNC decoder,
    we translate through the DN→MN bridge:
    
    1. Read fired engine indices from NIRON fire_list_2
    2. Pass through DN→MN bridge → get MANC MN body IDs
    3. Feed MANC MN IDs to VNC decoder
    
    Also supports DN stimulation (realistic: optogenetic protocol) and
    spontaneous activity (realistic: biological background firing).
    """
    
    def __init__(
        self, bridge: DnMnBridge, loader,
        dn_engine_indices: Optional[Set[int]] = None,
        stimulate_dns: int = 0,
        spontaneous_rate: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bridge = bridge
        self.loader = loader
        self.dn_engine_indices = dn_engine_indices or set()
        self.stimulate_dns = stimulate_dns
        self.spontaneous_rate = spontaneous_rate
        self._bridge_report: Dict[str, Any] = {
            "dn_matches_found": 0,
            "mns_activated": 0,
            "dn_neurons_fired": [],
        }
        self._stim_count: int = 0
        self._spont_count: int = 0
    
    def _stimulate(self):
        """Inject stimulation into DN neurons.
        
        Realistic protocol: targeted optogenetic stimulation of DNs.
        Adds charge directly to push neurons over threshold, then marks
        them for Fire1 evaluation.
        """
        import random
        if self.stimulate_dns <= 0 or not self.dn_engine_indices:
            return 0
        
        dn_list = list(self.dn_engine_indices)
        n_stim = min(self.stimulate_dns, len(dn_list))
        targets = random.sample(dn_list, n_stim)
        
        for engine_idx in targets:
            if engine_idx < len(self.engine.neurons):
                # Add charge to push over threshold (1.0+)
                self.engine.neurons[engine_idx].add_to_current_value(1.5)
                # Mark for Fire1 evaluation
                self.engine.add_neuron_to_fire_list1(engine_idx)
        
        self._stim_count += n_stim
        return n_stim
    
    def _spontaneous_activity(self):
        """Add spontaneous background firing.
        
        Realistic: biological neurons show spontaneous activity.
        Each neuron has a small probability of receiving enough
        charge to fire each cycle.
        """
        import random
        if self.spontaneous_rate <= 0:
            return 0
        
        n_spont = 0
        for i in range(len(self.engine.neurons)):
            if random.random() < self.spontaneous_rate:
                neuron = self.engine.neurons[i]
                # Add charge proportional to leak rate (temporary depolarization)
                neuron.add_to_current_value(0.5 + random.random() * 1.0)
                self.engine.add_neuron_to_fire_list1(i)
                n_spont += 1
        
        self._spont_count += n_spont
        return n_spont
    
    def step(self) -> Dict[str, Any]:
        """Execute one full closed-loop step with bridge translation."""
        t_start = time.perf_counter()
        
        # Read sensors (once per physics step)
        sensors = self.read_sensors()
        
        # Brain sub-steps
        total_fired = 0
        total_sensory_spikes = 0
        total_stim_spikes = 0
        total_spont_spikes = 0
        all_fired_engine_indices: Set[int] = set()
        
        for _ in range(self.brain_steps_per_physics):
            # ── STIMULATION (before sensory) ──────────────────────────
            # Realistic: optogenetic DN stimulation + spontaneous activity
            total_stim_spikes += self._stimulate()
            total_spont_spikes += self._spontaneous_activity()
            
            # Update sensory models and inject spikes
            sensory_spikes = self.update_sensory(sensors)
            total_sensory_spikes += sensory_spikes
            
            # Run NIRON fire cycle
            fired_count, cycle = self.engine.fire()
            total_fired += fired_count
            
            # Read motor output from fired neurons (engine indices)
            fired_motor_neurons = self.read_motor_output()
            all_fired_engine_indices.update(fired_motor_neurons)
            
            self.sim_time_ms += self.dt_brain_ms
        
        # ── BRIDGE TRANSLATION ─────────────────────────────────────────
        # Translate engine indices → MANC MN IDs via DN→MN bridge
        mn_activations = {}
        bridge_report = {"dn_matches_found": 0, "mns_activated": 0, "dn_neurons_fired": []}
        
        if self.bridge is not None and all_fired_engine_indices:
            mn_activations, bridge_report = self.bridge.translate_batch(
                all_fired_engine_indices, self.loader
            )
            self._bridge_report = bridge_report
        
        # Accumulate MANC MN IDs in VNC decoder
        if self.vnc_decoder is not None:
            if mn_activations:
                # Feed MANC MN IDs (not engine indices!) to the decoder
                manc_mn_ids = set(mn_activations.keys())
                self.vnc_decoder.accumulate(manc_mn_ids)
            else:
                # No DNs fired — no MN activation (scientifically valid)
                self.vnc_decoder.accumulate(set())
        
        # Decode accumulated motor commands
        joint_commands: Dict[str, float] = {}
        if self.vnc_decoder is not None:
            joint_commands = self.vnc_decoder.decode()
        
        # Apply motor commands to MuJoCo
        self.apply_motor_commands(joint_commands)
        
        # Step physics
        try:
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
        self.metrics['dn_matches'].append(bridge_report.get("dn_matches_found", 0))
        self.metrics['mns_activated'].append(bridge_report.get("mns_activated", 0))
        
        # Active joints
        active_joints = sum(1 for v in joint_commands.values() if abs(v) > 0.001)
        self.metrics['active_joints'].append(active_joints)
        
        return {
            'step': self.step_count,
            'time_ms': self.sim_time_ms,
            'fired_neurons': total_fired,
            'sensory_spikes': total_sensory_spikes,
            'stim_spikes': total_stim_spikes,
            'spont_spikes': total_spont_spikes,
            'active_joints': active_joints,
            'step_duration_ms': step_duration_ms,
            'dn_matches': bridge_report.get("dn_matches_found", 0),
            'mns_activated': bridge_report.get("mns_activated", 0),
            'torque_applied': active_joints > 0,
            'dn_types': bridge_report.get("dn_types", []),
        }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 9: Connectome-Driven with DN→MN Bridge")
    parser.add_argument('--steps', type=int, default=50, help='Physics steps')
    parser.add_argument('--neurons', type=int, default=500, help='Max total neurons (including DNs)')
    parser.add_argument('--render', action='store_true', help='Render frames')
    parser.add_argument('--video', action='store_true', help='Compile MP4')
    parser.add_argument('--render-every', type=int, default=1, help='Render every N steps')
    parser.add_argument('--w-syn', type=float, default=1.0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--bridge-only', action='store_true', help='Only init bridge, exit')
    parser.add_argument('--stimulate-dns', type=int, default=0, help='Stimulate N random DNs per brain cycle (0=off). Realistic: optogenetic stimulation protocol.')
    parser.add_argument('--spontaneous-rate', type=float, default=0.0, help='Spontaneous firing probability per neuron per cycle (0=off)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PHASE 9: CONNECTOME-DRIVEN SIMULATION (DN→MN Bridge)")
    print(f"{'='*60}")
    print(f"Neurons: ≤{args.neurons} | Steps: {args.steps} | W_syn: {args.w_syn}")
    print(f"Render: {args.render} | Video: {args.video}")
    print(f"DN Matches: {DN_MATCHES_JSON}")
    print(f"Pathways: {DN_MN_PATHWAYS_JSON}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n", flush=True)

    # ── STEP 0: Initialize DN→MN Bridge ────────────────────────────────
    print("[0/6] Loading DN→MN bridge...", flush=True)
    t0 = time.perf_counter()
    
    bridge = DnMnBridge(
        dn_matches_path=DN_MATCHES_JSON,
        pathways_path=DN_MN_PATHWAYS_JSON,
        min_pathway_confidence=0.01,
    )
    bridge.initialize()
    t1 = time.perf_counter()
    bridge_summary = bridge.summary()
    print(f"  ✅ Bridge: {bridge_summary['dn_matches_loaded']} DN matches, "
          f"{bridge_summary['unique_mns_loaded']:,} MNs, "
          f"{bridge_summary['dns_with_pathways']} DNs with pathways "
          f"({t1-t0:.1f}s)", flush=True)
    
    if args.bridge_only:
        print(f"\n  Bridge summary: {json.dumps(bridge_summary, indent=2)}")
        return 0

    # ── STEP 1: Load Connectome with DNs ───────────────────────────────
    print("\n[1/6] Loading FlyWire connectome with DN priority...", flush=True)
    t0 = time.perf_counter()
    
    # Load DN root IDs
    dn_ids = load_dn_root_ids(MOTOR_MAP_JSON)
    print(f"  Found {len(dn_ids)} FlyWire DNs in motor map", flush=True)
    
    # Load connections filtered to include DNs
    neurons_nt, connections = filter_connections_for_dns(
        CONNECTIONS_CSV,
        dn_ids=dn_ids,
        max_total_neurons=args.neurons,
    )
    
    # Build engine using the loader's build_engine (but with our filtered data)
    config = {
        'min_syn_count': 0,
        'leak_rate': 0.05,
        'refractory_delay': 1,
        'thread_count': 4,
        'normalize_weights': True,
    }
    
    loader = FlyWireConnectomeLoader(CONNECTIONS_CSV, config=config)
    # Manually set up the ID mapping from our filtered neurons
    sorted_ids = sorted(neurons_nt.keys())
    loader.flywire_to_idx = {fw_id: idx for idx, fw_id in enumerate(sorted_ids)}
    loader.idx_to_flywire = sorted_ids
    loader.neuron_nt_types = neurons_nt
    
    engine = loader.build_engine(neurons_nt, connections)
    n_syn = sum(len(n.synapses_out) for n in engine.neurons)
    n_dns_loaded = sum(1 for fw_id in sorted_ids if fw_id in dn_ids)
    
    t1 = time.perf_counter()
    print(f"  ✅ Engine: {engine.array_size:,} neurons ({n_dns_loaded} DNs), "
          f"{n_syn:,} synapses ({t1-t0:.1f}s)", flush=True)

    # ── STEP 2: Load VNC Decoder ───────────────────────────────────────
    print("\n[2/6] Loading VNC motor decoder...", flush=True)
    t0 = time.perf_counter()
    
    vnc_map = os.path.join(VNC_DIR, "vnc_actuator_map.json")
    pathways = DN_MN_PATHWAYS_JSON
    
    decoder = VNCMotorDecoder.load_from_vnc(
        vnc_actuator_map_path=vnc_map,
        pathways_path=pathways,
        tau_decay=50.0, global_gain=0.1,
        dt_brain_ms=1.0, dt_physics_ms=5.0,
    )
    t1 = time.perf_counter()
    s = decoder.summary()
    print(f"  ✅ Decoder: {s['total_joints']} joints, {s['mns_with_types']} typed MNs ({t1-t0:.1f}s)", flush=True)

    # ── STEP 3: Load SimFLy Body ───────────────────────────────────────
    print("\n[3/6] Loading SimFLy MuJoCo body...", flush=True)
    model = mujoco.MjModel.from_xml_path(SIMFLY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print(f"  ✅ SimFLy: {model.nbody} bodies, {model.nu} actuators", flush=True)

    # Build actuator index
    act_idx = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_idx[name or f"act_{i}"] = i
    print(f"  Mapped {len(act_idx)} actuators", flush=True)
    
    # ── Pre-compute DN engine indices for stimulation ──────────────────
    dn_engine_indices: Set[int] = set()
    if args.stimulate_dns > 0 or args.spontaneous_rate > 0:
        for fw_id in dn_ids:
            if fw_id in loader.flywire_to_idx:
                dn_engine_indices.add(loader.flywire_to_idx[fw_id])
        print(f"  DN engine indices: {len(dn_engine_indices)} DNs mapped "
              f"({len(dn_engine_indices & dn_engine_indices)} of {len(dn_ids)} FlyWire DNs)", flush=True)

    # ── STEP 4: Initialize Sensory ─────────────────────────────────────
    print("\n[4/6] Initializing sensory modules...", flush=True)
    
    injector = SensoryInjector(
        engine=engine,
        num_ommatidia=100, num_touch_sensors=50,
        num_proprio_sensors=50, num_orn_types=10, num_grn_types=5,
    )
    
    # Map sensory neurons to engine indices
    for i in range(min(500, engine.array_size)):
        if i < 200:
            injector.sensory_to_brain_index[('photoreceptor_R1_R6', i % 100)] = i
        elif i < 350:
            injector.sensory_to_brain_index[('proprioception', i - 200)] = i
        elif i < 500:
            injector.sensory_to_brain_index[('touch', i - 350)] = i
    
    print(f"  ✅ Injector: {len(injector.sensory_to_brain_index)} sensory→brain mappings", flush=True)

    # ── STEP 5: Initialize Bridge-Aware Closed Loop ────────────────────
    print("\n[5/6] Initializing bridge-aware closed loop...", flush=True)
    
    loop = BridgeAwareClosedLoop(
        bridge=bridge,
        loader=loader,
        dn_engine_indices=dn_engine_indices,
        stimulate_dns=args.stimulate_dns,
        spontaneous_rate=args.spontaneous_rate,
        engine=engine,
        injector=injector,
        vnc_decoder=decoder,
        model=model,
        data=data,
        actuator_map=act_idx,
        brain_rate_hz=1000,
        physics_rate_hz=200,
        w_syn=args.w_syn,
        sensory_feedback_enabled=True,
        enable_vision=True,
        enable_proprioception=True,
        enable_touch=True,
    )
    loop.initialize()
    stim_info = f"DN stim={args.stimulate_dns}/cycle" if args.stimulate_dns > 0 else "no DN stim"
    spont_info = f"spont rate={args.spontaneous_rate}" if args.spontaneous_rate > 0 else ""
    extra_info = " | ".join(filter(None, [stim_info, spont_info]))
    print(f"  ✅ BridgeAwareClosedLoop: {engine.array_size} neurons → "
          f"bridge ({bridge.num_dn_matches} DNs) → "
          f"decoder ({decoder.summary()['total_joints']} joints) → "
          f"{model.nu} actuators"
          f"{' | ' + extra_info if extra_info else ''}", flush=True)

    # ── STEP 6: Run Simulation ─────────────────────────────────────────
    print(f"\n[6/6] Running simulation ({args.steps} steps)...", flush=True)
    print(f"{'='*60}", flush=True)
    
    renderer = None
    frames = []
    if args.render:
        try:
            renderer = mujoco.Renderer(model, 1280, 720)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"  Renderer: 1280x720", flush=True)
        except Exception as e:
            print(f"  [WARN] Renderer failed: {e}", flush=True)
    
    t_start = time.perf_counter()
    step_reports = []
    any_torque = False
    any_dn_fired = False
    
    for step_i in range(args.steps):
        report = loop.step()
        step_reports.append(report)
        
        if report.get('torque_applied'):
            any_torque = True
        if report.get('dn_matches', 0) > 0:
            any_dn_fired = True
        
        # Render
        if renderer and (step_i + 1) % args.render_every == 0:
            try:
                renderer.update_scene(data, camera=-1)
                pixels = renderer.render()
                from PIL import Image, ImageDraw, ImageFont
                img = Image.fromarray(pixels)
                draw = ImageDraw.Draw(img)
                
                # HUD: Connectome-Driven label
                y = 10
                draw.text((10, y), "CONNECTOME-DRIVEN (Phase 9)", fill=(0, 255, 0))
                y += 22
                
                # Brain cycle info
                brain_cycle = loop.engine.cycle if hasattr(loop.engine, 'cycle') else step_i
                draw.text((10, y), 
                    f"Brain Cycle: {brain_cycle} | Step: {step_i+1}/{args.steps}", 
                    fill=(255, 255, 255))
                y += 22
                
                # Fired neurons
                draw.text((10, y), 
                    f"Fired Neurons: {report['fired_neurons']}", 
                    fill=(255, 200, 100))
                y += 22
                
                # DN matches
                dn_count = report.get('dn_matches', 0)
                if dn_count > 0:
                    color = (0, 255, 100)  # Green for DN match
                else:
                    color = (255, 100, 100)  # Red for no DN match
                draw.text((10, y), 
                    f"DN Matches: {dn_count} | MNs Activated: {report.get('mns_activated', 0)}", 
                    fill=color)
                y += 22
                
                # Torque status
                if report.get('torque_applied'):
                    draw.text((10, y), "Torque: APPLIED", fill=(0, 255, 0))
                else:
                    draw.text((10, y), "Torque: NONE", fill=(255, 100, 100))
                y += 22
                
                # DN types if any
                dn_types = report.get('dn_types', [])
                if dn_types:
                    draw.text((10, y), f"DN Types: {', '.join(dn_types[:3])}", fill=(200, 200, 255))
                
                frame_path = os.path.join(OUTPUT_DIR, f"frame_{step_i+1:06d}.png")
                img.save(frame_path)
                frames.append(frame_path)
            except Exception as e:
                if step_i == 0:
                    print(f"  [WARN] Frame save failed: {e}", flush=True)
        
        # Progress
        if (step_i + 1) % max(1, args.steps // 10) == 0:
            elapsed = time.perf_counter() - t_start
            rtf = (loop.sim_time_ms / 1000.0) / elapsed if elapsed > 0 else 0
            print(f"  Step {step_i+1}/{args.steps} | sim={loop.sim_time_ms/1000:.1f}s | "
                  f"RTF={rtf:.4f}x | fired={report['fired_neurons']} | "
                  f"DNs={report.get('dn_matches',0)} | "
                  f"MNs={report.get('mns_activated',0)} | "
                  f"torque={'YES' if report.get('torque_applied') else 'no'}",
                  flush=True)
    
    total_elapsed = time.perf_counter() - t_start
    
    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 9 SIMULATION COMPLETE")
    print(f"{'='*60}")
    
    sim_summary = loop.get_summary()
    bridge_stats = bridge.summary()
    engine_stats = engine.get_stats()
    
    # Add bridge-specific metrics to summary
    avg_dn = sum(loop.metrics.get('dn_matches', [0])) / max(1, len(loop.metrics.get('dn_matches', [])))
    avg_mn = sum(loop.metrics.get('mns_activated', [0])) / max(1, len(loop.metrics.get('mns_activated', [])))
    
    print(f"  Steps: {sim_summary['total_steps']}")
    print(f"  Simulated: {sim_summary['simulated_time_s']:.1f}s")
    print(f"  Wall time: {total_elapsed:.1f}s")
    print(f"  Overall RTF: {sim_summary['simulated_time_s'] / total_elapsed:.6f}x")
    print(f"  Avg fired/step: {sim_summary['avg_fired_per_step']:.0f}")
    print(f"  Avg DN matches/step: {avg_dn:.1f}")
    print(f"  Avg MNs activated/step: {avg_mn:.1f}")
    print(f"  Any DN ever fired: {any_dn_fired}")
    print(f"  Any torque applied: {any_torque}")
    print(f"  DNs loaded: {n_dns_loaded}")
    print(f"  Bridge stats: {bridge_stats['dn_matches_loaded']} matched DNs, "
          f"{bridge_stats['unique_mns_loaded']:,} MNs")
    
    # Engine stats
    es = engine_stats
    print(f"  NIRON cycles: {es.get('cycles', 0)}")
    
    # ── Scientific Report ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SCIENTIFIC RESULT:")
    if any_torque:
        print(f"  ✅ CONNECTOME-DRIVEN MOVEMENT ACHIEVED")
        print(f"  Torque commands generated from real FlyWire→MANC pathways")
    else:
        if any_dn_fired:
            print(f"  ⚠ DNS FIRED BUT NO TORQUE — pathway→actuator mapping issue")
            print(f"  DNs activated MNs but MNs didn't map to loaded joints")
        else:
            print(f"  ℹ️ NO DNS FIRED — scientifically valid result")
            print(f"  Spontaneous activity in {engine.array_size} neurons produced "
                  f"no DN activation")
            print(f"  This is expected: DNs require sensory input or targeted "
                  f"stimulation to fire")
    print(f"{'─'*60}")
    
    # ── Compile Video ───────────────────────────────────────────────────
    video_path = None
    if args.video and frames:
        print(f"\n  Compiling {len(frames)} frames to video...", flush=True)
        video_path = os.path.join(OUTPUT_DIR, "connectome_phase9.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-framerate', '30',
                '-i', os.path.join(OUTPUT_DIR, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'fast', '-crf', '23', video_path,
            ], capture_output=True, timeout=300)
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / 1e6
                print(f"  ✅ Video: {video_path} ({size_mb:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  [WARN] Video compile failed: {e}", flush=True)
    
    # ── Save Report ─────────────────────────────────────────────────────
    report_path = args.output or os.path.join(OUTPUT_DIR, "phase9_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'phase': 9,
        'phase_name': 'DN→MN Bridge — Connectome-Driven Movement',
        'connectome': {
            'neurons': engine.array_size,
            'synapses': n_syn,
            'dns_loaded': n_dns_loaded,
            'nt_distribution': dict(Counter(loader.neuron_nt_types.values())),
        },
        'bridge': bridge_stats,
        'vnc': decoder.summary(),
        'simfly': {
            'bodies': model.nbody,
            'actuators': model.nu,
            'joints': model.njnt,
        },
        'simulation': {
            **sim_summary,
            'avg_dn_matches_per_step': avg_dn,
            'avg_mns_activated_per_step': avg_mn,
            'any_dn_fired': any_dn_fired,
            'any_torque_applied': any_torque,
        },
        'engine': engine_stats,
        'wall_time_s': total_elapsed,
        'video': video_path,
        'scientific_result': (
            'CONNECTOME-DRIVEN_MOVEMENT' if any_torque
            else 'NO_DN_ACTIVATION' if not any_dn_fired
            else 'DN_FIRED_NO_TORQUE'
        ),
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  📄 Report: {report_path}", flush=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
