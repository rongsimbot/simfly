#!/usr/bin/env python3
"""
Phase 9: DN→MN Bridge — FlyWire DNs → MANC MN activations via VNC decoder.

Compatible with: connectome_simulation.py Phase 9 (BridgeAwareClosedLoop).

Architecture:
  NIRON.fire() → fired indices → idx_to_flywire → FlyWire root IDs
    → DN matches lookup → MANC body IDs → DN→MN pathways lookup
    → aggregated MN activations → VNC decoder.accumulate()
    → VNC decoder.decode() → torque array

API (matches original DNtoMNbridge):
  1. Constructor: bridge = DnMnBridge(dn_matches_path=..., pathways_path=..., vnc_decoder=..., model=..., idx_to_flywire=...)
  2. bridge.initialize()  — loads data, builds lookup
  3. bridge.translate(fired_engine_indices, loader)  — returns {manc_mn_id: activation}
  4. bridge.summary()  — stats dict
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class DnMnBridge:
    """Bridges FlyWire NIRON output to MANC MN activations.

    API compatible with the original DNtoMNbridge:
    - Constructor takes dn_matches_path, pathways_path, vnc_decoder, model, idx_to_flywire
    - initialize() loads data and builds lookup
    - translate() converts fired engine indices to MN activations
    - summary() returns stats
    """

    def __init__(
        self,
        dn_matches_path: str,
        pathways_path: str,
        vnc_decoder: Any = None,
        model: Any = None,
        idx_to_flywire: Optional[List[int]] = None,
        min_pathway_confidence: float = 0.01,
        global_gain: float = 0.1,
    ):
        self.dn_matches_path = dn_matches_path
        self.pathways_path = pathways_path
        self.vnc_decoder = vnc_decoder
        self.model = model
        self.idx_to_flywire = idx_to_flywire or []
        self.min_pathway_confidence = min_pathway_confidence
        self.global_gain = global_gain

        # Initialized after initialize()
        self.dn_matches: Dict[str, Dict[str, Any]] = {}
        self._dn_root_ids: Set[str] = set()
        self._fw_to_manc: Dict[int, int] = {}  # FlyWire root ID (int) → MANC body ID (int)
        self._known_dn_ids: Set[int] = set()  # FlyWire root IDs (int set) for fast lookup
        self._dn_to_mns: Dict[str, Dict[Tuple[int, str], float]] = defaultdict(dict)
        self._joint_to_act: Dict[str, int] = {}
        self._num_actuators: int = 0

        self._initialized = False
        self._total_dn_matches = 0
        self._total_unique_mns = 0
        self._build_time_s = 0.0

        # Runtime tracking
        self.dn_fire_count = 0
        self.mn_activation_count = 0
        self.last_fired_dns: Set[str] = set()
        self.last_activated_mns: Set[int] = set()
        self._total_dn_ever: int = 0
        self._torque_steps: int = 0

    # ── Initialization ──────────────────────────────────────────────────

    def initialize(self, verbose: bool = True) -> "DnMnBridge":
        """Load DN matches and pathways, build compact lookup."""
        t0 = time.perf_counter()

        # Step 1: Load DN matches
        if verbose:
            print(f"  [bridge] Loading DN matches from: {self.dn_matches_path}")
        with open(self.dn_matches_path) as f:
            dn_data = json.load(f)
        self.dn_matches = dn_data.get("matches", {})
        self._dn_root_ids = set(self.dn_matches.keys())
        self._total_dn_matches = len(self.dn_matches)

        # Build int→int lookup for fast translation
        self._fw_to_manc = {}
        for fw_root_id_str, match_info in self.dn_matches.items():
            try:
                fw_id = int(fw_root_id_str)
                manc_id = int(match_info["manc_body_id"])
                self._fw_to_manc[fw_id] = manc_id
            except (ValueError, KeyError, TypeError):
                pass
        self._known_dn_ids = set(self._fw_to_manc.keys())

        if verbose:
            print(f"  [bridge] Loaded {self._total_dn_matches} DN matches "
                  f"({len(self._fw_to_manc)} valid int→int)")

        # Step 2: Stream pathways to build DN→MN lookup
        if verbose:
            print(f"  [bridge] Streaming pathways: {self.pathways_path}")
            print(f"  [bridge] Building DN→MN lookup...")

        pathway_count = 0
        mn_set: Set[int] = set()
        dns_with_pathways = 0

        if os.path.exists(self.pathways_path) and os.path.getsize(self.pathways_path) < 6_000_000_000:
            try:
                with open(self.pathways_path) as f:
                    pw_data = json.load(f)
                pathways = pw_data.get("pathways", {})

                for fw_root_id, pw_list in pathways.items():
                    if fw_root_id not in self._dn_root_ids:
                        continue
                    dns_with_pathways += 1
                    for pw in pw_list:
                        pathway_count += 1
                        if pw.get("confidence", 0) < self.min_pathway_confidence:
                            continue
                        mn_id = pw.get("mn_id", 0)
                        segments = pw.get("segments", ["unknown"])
                        total_weight = pw.get("total_weight", 1)
                        mn_set.add(mn_id)
                        for seg in segments[:2]:
                            key = (mn_id, seg)
                            self._dn_to_mns[fw_root_id][key] = (
                                self._dn_to_mns[fw_root_id].get(key, 0.0) + total_weight
                            )
                self._total_unique_mns = len(mn_set)
            except (MemoryError, json.JSONDecodeError, OSError) as e:
                if verbose:
                    print(f"  [bridge] Pathways load failed: {e} — using DN→MANC direct only")
                self._total_unique_mns = 0
        else:
            if verbose:
                print(f"  [bridge] Pathways file too large or missing — using DN→MANC direct only")

        # Step 3: Build joint→actuator index from MuJoCo model
        if self.model is not None:
            import mujoco
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
                )
                if name:
                    self._joint_to_act[name] = i
            self._num_actuators = self.model.nu

        self._build_time_s = time.perf_counter() - t0
        self._initialized = True

        if verbose:
            print(f"  [bridge] Built lookup: {dns_with_pathways} DNs → "
                  f"{self._total_unique_mns} unique MNs "
                  f"({pathway_count} pathways) in {self._build_time_s:.1f}s")

        return self

    # ── Translation ─────────────────────────────────────────────────────

    def translate(
        self,
        fired_engine_indices: Set[int],
        loader: Any = None,
        activation_scale: float = 1.0,
        dn_classifier: Any = None,
        gait_mode: str = 'walk',
    ) -> Dict[int, float]:
        """Convert fired NIRON indices → MANC MN activations.

        M3: Weight DN contributions by subtype based on gait_mode.
        - In 'walk' mode: amplify walking (DNp) DNs, suppress stop (DNg/DNd) DNs.
        - In 'turn_left' mode: amplify turn-left (DNa) DNs.
        - In 'turn_right' mode: amplify turn-right (DNb) DNs.

        Args:
            fired_engine_indices: NIRON neuron indices that fired.
            loader: FlyWireConnectomeLoader (for idx_to_flywire).
            activation_scale: Activation gain.
            dn_classifier: Optional DNSubtypeClassifier for subtype weighting.
            gait_mode: Gait mode string from chemotaxis ('stance','walk','turn_left','turn_right').

        Returns:
            Dict mapping MANC body ID → activation strength [0, 1].
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")

        # Get idx_to_flywire from loader if available
        idx_to_fw = self.idx_to_flywire
        if loader is not None and hasattr(loader, 'idx_to_flywire'):
            idx_to_fw = loader.idx_to_flywire

        mn_activations: Dict[int, float] = defaultdict(float)
        matched_dns: Set[str] = set()

        for engine_idx in fired_engine_indices:
            if engine_idx >= len(idx_to_fw):
                continue
            fw_id = idx_to_fw[engine_idx]
            if fw_id <= 0:
                continue

            # Fast lookup: is this a known DN?
            if fw_id not in self._known_dn_ids:
                continue

            fw_id_str = str(fw_id)
            matched_dns.add(fw_id_str)
            manc_body_id = self._fw_to_manc[fw_id]

            # M3: Subtype-based activation weighting
            subtype_weight = 1.0
            if dn_classifier is not None and gait_mode != 'stance':
                subtype = dn_classifier.classify_dn(fw_id_str)
                if gait_mode in ('walk',):
                    if subtype == 'walking':
                        subtype_weight = 5.0  # Strong walking amplification
                    elif subtype == 'stop':
                        subtype_weight = 0.05  # Near-complete stop suppression
                    elif subtype in ('turn_left', 'turn_right'):
                        subtype_weight = 2.0
                elif gait_mode in ('turn_left', 'turn_right'):
                    if subtype == gait_mode:
                        subtype_weight = 5.0
                    elif subtype == 'stop':
                        subtype_weight = 0.05
                    elif subtype == 'walking':
                        subtype_weight = 2.0
            scale = activation_scale * subtype_weight

            # Check pathways for downstream MNs
            mn_entries = self._dn_to_mns.get(fw_id_str, {})
            if mn_entries:
                max_weight = max(mn_entries.values()) if mn_entries else 1.0
                if max_weight <= 0:
                    max_weight = 1.0
                for (mn_id, segment), weight in mn_entries.items():
                    normalized = (weight / max_weight) * scale
                    mn_activations[mn_id] = min(1.0, mn_activations.get(mn_id, 0.0) + normalized)
            else:
                # No pathways — just activate the MANC DN directly
                mn_activations[manc_body_id] = min(1.0, mn_activations.get(manc_body_id, 0.0) + scale)

        # Track stats
        self.dn_fire_count += len(matched_dns)
        self.mn_activation_count += len(mn_activations)
        self.last_fired_dns = matched_dns
        self.last_activated_mns = set(mn_activations.keys())
        if matched_dns:
            self._total_dn_ever += len(matched_dns)

        # Also accumulate in VNC decoder for torque generation
        if self.vnc_decoder is not None and self.last_activated_mns:
            self.vnc_decoder.accumulate(self.last_activated_mns)

        return dict(mn_activations)

    def get_torque_array(self) -> Optional[np.ndarray]:
        """Decode accumulated MN activations to a torque array.

        Returns:
            108-element numpy array if torques available, None otherwise.
        """
        if self.vnc_decoder is None or not self._joint_to_act or self._num_actuators == 0:
            return None

        joint_commands = self.vnc_decoder.decode()
        torques = np.zeros(self._num_actuators, dtype=np.float64)
        mapped = 0
        for joint_name, torque_value in joint_commands.items():
            if joint_name in self._joint_to_act:
                act_idx = self._joint_to_act[joint_name]
                torques[act_idx] = float(np.clip(torque_value, -1.0, 1.0))
                mapped += 1

        if mapped > 0:
            self._torque_steps += 1
            return torques
        return None

    # ── Query ───────────────────────────────────────────────────────────

    def is_dn(self, fw_root_id: str) -> bool:
        return fw_root_id in self._dn_root_ids

    def get_dn_info(self, fw_root_id: str) -> Optional[Dict[str, Any]]:
        return self.dn_matches.get(fw_root_id)

    def get_mn_set_for_dn(self, fw_root_id: str) -> Set[int]:
        entries = self._dn_to_mns.get(fw_root_id, {})
        return {mn_id for (mn_id, _) in entries.keys()}

    def get_last_dn_count(self) -> int:
        return len(self.last_fired_dns)

    def get_any_dn_ever_fired(self) -> bool:
        return self._total_dn_ever > 0

    def get_any_torque_applied(self) -> bool:
        return self._torque_steps > 0

    @property
    def num_dn_matches(self) -> int:
        return self._total_dn_matches

    @property
    def num_unique_mns(self) -> int:
        return self._total_unique_mns

    # ── Summary & Stats ─────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "dn_matches_loaded": self._total_dn_matches,
            "valid_int_mappings": len(self._fw_to_manc),
            "unique_mns_loaded": self._total_unique_mns,
            "build_time_s": self._build_time_s,
            "dns_with_pathways": len(self._dn_to_mns),
            "total_dn_fires": self.dn_fire_count,
            "total_mn_activations": self.mn_activation_count,
            "any_dn_ever_fired": self.get_any_dn_ever_fired(),
            "any_torque_applied": self.get_any_torque_applied(),
            "actuator_joints": len(self._joint_to_act),
            "num_actuators": self._num_actuators,
        }

    def stats(self) -> Dict[str, Any]:
        return self.summary()

    def reset_stats(self) -> None:
        self.dn_fire_count = 0
        self.mn_activation_count = 0
        self.last_fired_dns = set()
        self.last_activated_mns = set()
        self._total_dn_ever = 0
        self._torque_steps = 0
        if self.vnc_decoder and hasattr(self.vnc_decoder, 'reset'):
            self.vnc_decoder.reset()

    def __repr__(self) -> str:
        return (
            f"DnMnBridge(dns={self._total_dn_matches}, "
            f"mns={self._total_unique_mns}, "
            f"actuators={len(self._joint_to_act)}, "
            f"fired={self.dn_fire_count})"
        )
