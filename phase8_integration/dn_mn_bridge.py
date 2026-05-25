#!/usr/bin/env python3
"""
Phase 9: DN→MN Bridge — FlyWire Connectome → MANC Motor Neuron Translation

Translates FlyWire NIRON engine output (engine indices 0..N-1) → MANC motor
neuron torque commands via real connectome pathways.

Flow:
  1. NIRON fire cycle completes → get set of fired engine indices
  2. Map engine indices → FlyWire root IDs (via loader.idx_to_flywire)
  3. Check if any fired neurons are known DNs (from dn_matcher_fixed)
  4. For matched DNs, look up MANC motor neuron targets (from pathway_builder)
  5. Aggregate MANC MN spike counts weighted by pathway confidence
  6. Output: dict of {manc_mn_body_id: activation_strength} for VNC decoder

SCIENTIFIC RIGOR: NO scripted motor patterns. Every activation traces back to
real FlyWire connectome firing → real MANC pathways.

Usage:
    bridge = DNtoMNbridge(
        dn_matches_path="vnc_bridge/dn_matches.json",
        pathways_path="vnc_bridge/dn_mn_pathways.json",
    )
    bridge.initialize()  # Build compact lookup from 5GB pathways file
    
    # After NIRON fire cycle:
    mn_activations = bridge.translate(fired_engine_indices, loader)
    vnc_decoder.accumulate(mn_activations.keys())
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple, Any


class DNtoMNbridge:
    """Bridges FlyWire connectome firing → MANC motor neuron activation.
    
    The critical missing piece: NIRON engine fires FlyWire neurons (identified
    by sequential engine indices), but the VNC motor decoder expects MANC
    motor neuron body IDs. This bridge resolves that translation using:
    
    1. DN matches: FlyWire root IDs → MANC DN body IDs (953 pairs, 71.3% match rate)
    2. DN→MN pathways: MANC DN → interneuron → motor neuron (13.7M pathways)
    3. Compact MN lookup: per FlyWire DN → weighted set of downstream MANC MNs
    """
    
    def __init__(
        self,
        dn_matches_path: str,
        pathways_path: str,
        min_pathway_confidence: float = 0.01,
    ):
        """Initialize the bridge.
        
        Args:
            dn_matches_path: Path to dn_matches.json from DNMatcher.
            pathways_path: Path to dn_mn_pathways.json from PathwayBuilder.
            min_pathway_confidence: Minimum pathway confidence to include.
        """
        self.dn_matches_path = dn_matches_path
        self.pathways_path = pathways_path
        self.min_pathway_confidence = min_pathway_confidence
        
        # DN matches: flywire_root_id (str) → {manc_body_id, type, nt, ...}
        self.dn_matches: Dict[str, Dict[str, Any]] = {}
        
        # Set of FlyWire root IDs that are DNs (for fast lookup)
        self._dn_root_ids: Set[str] = set()
        
        # Compact lookup: flywire_root_id → {(manc_mn_id, segment): total_weight}
        # Built by streaming the 5GB pathways file once
        self._dn_to_mns: Dict[str, Dict[Tuple[int, str], float]] = defaultdict(dict)
        
        # Stats
        self._initialized = False
        self._total_dn_matches: int = 0
        self._total_pathways: int = 0
        self._total_unique_mns: int = 0
        self._build_time_s: float = 0.0
        
        # Runtime tracking
        self.dn_fire_count: int = 0
        self.mn_activation_count: int = 0
        self.last_fired_dns: Set[str] = set()
        self.last_activated_mns: Set[int] = set()
    
    def initialize(self, verbose: bool = True) -> 'DNtoMNbridge':
        """Load DN matches and build the compact MN lookup.
        
        Streams the 5GB pathways file once to build a compact dict:
        {flywire_root_id: {(manc_mn_id, segment): summed_weight}}
        
        Returns:
            self for chaining.
        """
        t0 = time.perf_counter()
        
        # Step 1: Load DN matches
        if verbose:
            print(f"  [bridge] Loading DN matches from: {self.dn_matches_path}")
        
        with open(self.dn_matches_path) as f:
            dn_data = json.load(f)
        
        self.dn_matches = dn_data.get("matches", {})
        self._dn_root_ids = set(self.dn_matches.keys())
        self._total_dn_matches = len(self.dn_matches)
        
        if verbose:
            print(f"  [bridge] Loaded {self._total_dn_matches} DN matches")
        
        # Step 2: Stream pathways to build compact MN lookup
        if verbose:
            print(f"  [bridge] Streaming pathways from: {self.pathways_path}")
            print(f"  [bridge] Building compact DN→MN lookup...")
        
        pathway_count = 0
        mn_set: Set[int] = set()
        
        with open(self.pathways_path) as f:
            pw_data = json.load(f)
        
        pathways = pw_data.get("pathways", {})
        
        for fw_root_id, pw_list in pathways.items():
            if fw_root_id not in self._dn_root_ids:
                continue  # Shouldn't happen, but be safe
            
            for pw in pw_list:
                pathway_count += 1
                
                # Skip low-confidence pathways
                if pw.get("confidence", 0) < self.min_pathway_confidence:
                    continue
                
                mn_id = pw.get("mn_id", 0)
                segments = pw.get("segments", ["unknown"])
                fn_types = pw.get("mn_fn_types", ["unclassified_MN"])
                total_weight = pw.get("total_weight", 1)
                
                mn_set.add(mn_id)
                
                # Aggregate by MN ID + primary segment
                for seg in segments[:2]:  # Cap at 2 segments to keep lookup size manageable
                    key = (mn_id, seg)
                    self._dn_to_mns[fw_root_id][key] = (
                        self._dn_to_mns[fw_root_id].get(key, 0.0) + total_weight
                    )
        
        self._total_pathways = pathway_count
        self._total_unique_mns = len(mn_set)
        self._build_time_s = time.perf_counter() - t0
        
        if verbose:
            dns_with_pathways = len(self._dn_to_mns)
            avg_mns_per_dn = sum(len(v) for v in self._dn_to_mns.values()) / max(1, dns_with_pathways)
            print(f"  [bridge] Built lookup: {dns_with_pathways} DNs → "
                  f"{self._total_unique_mns:,} unique MNs "
                  f"(avg {avg_mns_per_dn:.0f} MNs/DN)")
            print(f"  [bridge] Total pathways processed: {pathway_count:,}")
            print(f"  [bridge] Build time: {self._build_time_s:.1f}s")
        
        self._initialized = True
        return self
    
    def translate(
        self,
        fired_engine_indices: Set[int],
        loader,  # FlyWireConnectomeLoader
        activation_scale: float = 1.0,
    ) -> Dict[int, float]:
        """Translate fired NIRON engine indices → MANC MN activations.
        
        Args:
            fired_engine_indices: Set of engine neuron indices that fired.
            loader: FlyWireConnectomeLoader with idx_to_flywire mapping.
            activation_scale: Global activation scaling factor.
        
        Returns:
            Dict mapping MANC MN body IDs → activation strength [0, 1].
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")
        
        mn_activations: Dict[int, float] = defaultdict(float)
        matched_dns: Set[str] = set()
        total_dns_checked = 0
        
        # Map engine indices → FlyWire root IDs → check DN membership
        for engine_idx in fired_engine_indices:
            if engine_idx >= len(loader.idx_to_flywire):
                continue
            
            fw_root_id_int = loader.idx_to_flywire[engine_idx]
            fw_root_id = str(fw_root_id_int)
            
            total_dns_checked += 1
            
            # Check if this neuron is a known DN
            if fw_root_id not in self._dn_root_ids:
                continue
            
            matched_dns.add(fw_root_id)
            
            # Look up downstream MNs
            mn_entries = self._dn_to_mns.get(fw_root_id, {})
            if not mn_entries:
                continue
            
            # Normalize weights across all MNs for this DN
            max_weight = max(mn_entries.values()) if mn_entries else 1.0
            if max_weight <= 0:
                max_weight = 1.0
            
            for (mn_id, segment), weight in mn_entries.items():
                # Normalize and scale
                normalized = (weight / max_weight) * activation_scale
                # Accumulate (multiple DNs can activate same MN)
                mn_activations[mn_id] = min(1.0, mn_activations.get(mn_id, 0.0) + normalized)
        
        # Track stats
        self.dn_fire_count += len(matched_dns)
        self.mn_activation_count += len(mn_activations)
        self.last_fired_dns = matched_dns
        self.last_activated_mns = set(mn_activations.keys())
        
        return dict(mn_activations)
    
    def translate_batch(
        self,
        fired_engine_indices: Set[int],
        loader,
        activation_scale: float = 1.0,
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """Translate with detailed report.
        
        Returns:
            Tuple of (mn_activations dict, report dict).
        """
        mn_activations = self.translate(fired_engine_indices, loader, activation_scale)
        
        report = {
            "engine_neurons_fired": len(fired_engine_indices),
            "dn_matches_found": len(matched_dns := self.last_fired_dns),
            "mns_activated": len(mn_activations),
            "dn_match_rate": len(matched_dns) / max(1, len(fired_engine_indices)),
            "avg_mn_per_dn": len(mn_activations) / max(1, len(matched_dns)),
            "dn_types": [
                self.dn_matches.get(dn_id, {}).get("flywire_type", "?")
                for dn_id in list(matched_dns)[:5]
            ],
        }
        
        return mn_activations, report
    
    def get_dn_info(self, fw_root_id: str) -> Optional[Dict[str, Any]]:
        """Get DN match info for a FlyWire root ID."""
        return self.dn_matches.get(fw_root_id)
    
    def get_mn_set_for_dn(self, fw_root_id: str) -> Set[int]:
        """Get all MANC MN IDs downstream of a FlyWire DN."""
        entries = self._dn_to_mns.get(fw_root_id, {})
        return {mn_id for (mn_id, _) in entries.keys()}
    
    def is_dn(self, fw_root_id: str) -> bool:
        """Check if a FlyWire root ID is a known DN."""
        return fw_root_id in self._dn_root_ids
    
    @property
    def num_dn_matches(self) -> int:
        return self._total_dn_matches
    
    @property
    def num_unique_mns(self) -> int:
        return self._total_unique_mns
    
    def summary(self) -> Dict[str, Any]:
        """Get bridge summary statistics."""
        return {
            "dn_matches_loaded": self._total_dn_matches,
            "unique_mns_loaded": self._total_unique_mns,
            "pathways_processed": self._total_pathways,
            "build_time_s": self._build_time_s,
            "dns_with_pathways": len(self._dn_to_mns),
            "total_dn_fires": self.dn_fire_count,
            "total_mn_activations": self.mn_activation_count,
        }
    
    def reset_stats(self) -> None:
        """Reset runtime statistics."""
        self.dn_fire_count = 0
        self.mn_activation_count = 0
        self.last_fired_dns = set()
        self.last_activated_mns = set()


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    
    # Paths
    BASE = os.path.expanduser("~/simrobotics-storage/research/flywire/simfly-robotic-model")
    matches_path = os.path.join(BASE, "vnc_bridge", "dn_matches.json")
    pathways_path = os.path.join(BASE, "vnc_bridge", "dn_mn_pathways.json")
    
    if not os.path.exists(matches_path) or not os.path.exists(pathways_path):
        print("ERROR: Data files not found. Run from GB10.")
        exit(1)
    
    print("=" * 60)
    print("DN→MN BRIDGE — TEST")
    print("=" * 60)
    
    bridge = DNtoMNbridge(matches_path, pathways_path)
    bridge.initialize()
    
    summary = bridge.summary()
    print(f"\nBridge Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Test translation with a known DN
    sample_dn = list(bridge.dn_matches.keys())[0]
    dn_info = bridge.get_dn_info(sample_dn)
    print(f"\nSample DN: {sample_dn} → {dn_info}")
    
    mn_set = bridge.get_mn_set_for_dn(sample_dn)
    print(f"Downstream MNs: {len(mn_set)}")
    print(f"Sample MN IDs: {sorted(mn_set)[:10]}")
    
    print("\n✅ Bridge test complete")
