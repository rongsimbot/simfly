#!/usr/bin/env python3
"""
DN Subtype Classifier — classifies FlyWire Descending Neurons by functional subtype.

Reads dn_matches.json and categorizes each DN based on its flywire_type prefix
into behavioral subtypes that map to gait patterns.

Classification:
  DNp* → 'walking'    (protocerebral — locomotion)
  DNa* → 'turn_left'  (anterior — leftward bias)
  DNb* → 'turn_right' (posterior — rightward bias)
  DNg* → 'stop'        (gnathal — grooming, non-locomotion)
  DNd* → 'stop'        (descending — inhibitory)
  others → 'unknown'

Also maps chemosensory gradient direction to gait mode for chemotaxis.
"""
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math


# ── Prefix → subtype mapping ─────────────────────────────────────────────

PREFIX_MAP: Dict[str, str] = {
    'DNp': 'walking',      # protocerebral → locomotion
    'DNa': 'turn_left',     # anterior → leftward bias
    'DNb': 'turn_right',    # posterior → rightward bias
    'DNg': 'stop',          # gnathal → grooming/non-locomotion
    'DNd': 'stop',          # descending → inhibitory
}

# All expected subtype categories
ALL_SUBTYPES = ['walking', 'turn_left', 'turn_right', 'stop', 'unknown']


class DNSubtypeClassifier:
    """Classifies FlyWire DNs by functional subtype based on flywire_type prefix.

    Usage::

        classifier = DNSubtypeClassifier('dn_matches.json')
        print(classifier.subtype_counts)
        # {'walking': 241, 'turn_left': 52, 'turn_right': 34, 'stop': 594, 'unknown': 32}

        subtype = classifier.classify_dn('720575940603580960')
        # 'walking' (DNpe054 prefix)

        mode = classifier.determine_gait_mode((0.01, 0.04), 0.15)
        # 'walk'
    """

    def __init__(self, dn_matches_path: str):
        """Load DN matches and classify all DNs.

        Args:
            dn_matches_path: Path to dn_matches.json with 'matches' key.
        """
        self.dn_matches_path = dn_matches_path
        self._dn_subtype: Dict[str, str] = {}
        self._subtype_counts: Dict[str, int] = defaultdict(int)
        self._match_info: Dict[str, Dict] = {}
        self._total_dns = 0
        self._initialized = False

        self._load()

    def _classify_by_prefix(self, flywire_type: str) -> str:
        """Classify a DN by its flywire_type prefix.

        Args:
            flywire_type: Full type string like 'DNpe054', 'DNa02', 'DNb01'.

        Returns:
            Subtype category string.
        """
        if not flywire_type or not isinstance(flywire_type, str):
            return 'unknown'

        # Try 3-char prefix first (DNp, DNa, DNb, DNg, DNd)
        prefix3 = flywire_type[:3]
        if prefix3 in PREFIX_MAP:
            return PREFIX_MAP[prefix3]

        # Try 2-char prefix as fallback
        prefix2 = flywire_type[:2]
        if prefix2 == 'DN':
            # DN1*, DNx* etc. → unknown
            return 'unknown'

        return 'unknown'

    def _load(self) -> None:
        """Load DN matches JSON and classify all DNs."""
        if not os.path.exists(self.dn_matches_path):
            print(f"[classifier] WARNING: DN matches not found: {self.dn_matches_path}")
            self._initialized = True
            return

        with open(self.dn_matches_path) as f:
            data = json.load(f)

        matches = data.get('matches', {})
        self._total_dns = len(matches)

        for fw_root_id, info in matches.items():
            fw_type = info.get('flywire_type', 'unknown')
            subtype = self._classify_by_prefix(fw_type)
            self._dn_subtype[fw_root_id] = subtype
            self._subtype_counts[subtype] += 1
            self._match_info[fw_root_id] = info

        self._initialized = True

        # Print summary on load
        print(f"  [classifier] Loaded {self._total_dns} DNs:", flush=True)
        for st in ALL_SUBTYPES:
            count = self._subtype_counts.get(st, 0)
            if count > 0:
                print(f"    {st}: {count}", flush=True)

    # ── Classification ──────────────────────────────────────────────────

    def classify_dn(self, flywire_root_id: str) -> str:
        """Get subtype for a single DN.

        Args:
            flywire_root_id: FlyWire root ID string.

        Returns:
            Subtype: 'walking', 'turn_left', 'turn_right', 'stop', or 'unknown'.
        """
        return self._dn_subtype.get(str(flywire_root_id), 'unknown')

    def classify_dns(self, flywire_root_ids: List[str]) -> Dict[str, int]:
        """Count subtypes for a list of firing DN IDs.

        Args:
            flywire_root_ids: List of FlyWire root ID strings.

        Returns:
            Dict with counts per subtype, e.g.:
            {'walking': 5, 'turn_left': 3, 'turn_right': 2, 'stop': 10, 'unknown': 1}
        """
        counts: Dict[str, int] = {st: 0 for st in ALL_SUBTYPES}
        for dn_id in flywire_root_ids:
            subtype = self.classify_dn(str(dn_id))
            counts[subtype] += 1
        return counts

    def get_subtype_counts(self) -> Dict[str, int]:
        """Get total counts of all DNs by subtype (cached from load).

        Returns:
            Dict with total counts per subtype.
        """
        return dict(self._subtype_counts)

    def get_dn_info(self, flywire_root_id: str) -> Optional[Dict]:
        """Get the full match info for a DN.

        Args:
            flywire_root_id: FlyWire root ID string.

        Returns:
            Match info dict or None if not found.
        """
        return self._match_info.get(str(flywire_root_id))

    # ── Chemotaxis → Gait Mode ──────────────────────────────────────────

    def determine_gait_mode(
        self, 
        chemo_direction: Optional[Tuple[float, float]] = None,
        chemo_concentration: float = 0.0,
    ) -> str:
        """Determine gait mode from chemosensory gradient.

        Maps the direction and strength of the odor gradient to a gait mode
        that drives the fly toward the food source.

        Args:
            chemo_direction: (dx, dy) gradient direction vector.
            chemo_concentration: Odor concentration at fly position [0, 1].

        Returns:
            Gait mode: 'stance', 'walk', 'turn_left', or 'turn_right'.
        """
        # No odor detected → stay put (stance)
        if chemo_concentration < 0.005:
            return 'stance'

        if chemo_direction is None:
            return 'walk'  # Default: walk forward when odor detected

        dx, dy = chemo_direction

        grad_magnitude = math.sqrt(dx*dx + dy*dy)
        if grad_magnitude < 0.005:
            return 'stance'

        # Normalize
        dx_norm = dx / grad_magnitude
        dy_norm = dy / grad_magnitude

        # Food is roughly ahead → walk forward
        if abs(dx_norm) < 0.5 and dy_norm > -0.2:
            return 'walk'

        # Food is significantly to the right → turn left (toward food)
        if dx_norm > 0.15:
            return 'turn_left'

        # Food is significantly to the left → turn right
        if dx_norm < -0.25:
            return 'turn_right'

        # Behind → turn around (use the stronger direction)
        if dy_norm < -0.2:
            if dx_norm > 0:
                return 'turn_left'
            else:
                return 'turn_right'

        # Default: walk forward if any odor detected
        return 'walk'

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def total_dns(self) -> int:
        """Total number of DNs loaded."""
        return self._total_dns

    @property
    def subtype_counts(self) -> Dict[str, int]:
        """Cached counts of all DNs by subtype."""
        return dict(self._subtype_counts)

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return classifier summary."""
        return {
            'total_dns': self._total_dns,
            'subtypes': dict(self._subtype_counts),
            'source': os.path.basename(self.dn_matches_path),
        }


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Test with a minimal inline dataset
    test_data = {
        "matches": {
            "1001": {"flywire_type": "DNpe054", "flywire_nt": "ACH"},
            "1002": {"flywire_type": "DNae004", "flywire_nt": "ACH"},
            "1003": {"flywire_type": "DNb01",   "flywire_nt": "ACH"},
            "1004": {"flywire_type": "DNg63",   "flywire_nt": "GABA"},
            "1005": {"flywire_type": "DNd01",   "flywire_nt": "GLUT"},
            "1006": {"flywire_type": "DN123",   "flywire_nt": "ACH"},
        }
    }

    test_path = '/tmp/test_dn_matches.json'
    with open(test_path, 'w') as f:
        json.dump(test_data, f)

    c = DNSubtypeClassifier(test_path)
    print(f"\nTotal DNs: {c.total_dns}")
    print(f"Subtype counts: {c.subtype_counts}")

    # Test single classification
    assert c.classify_dn('1001') == 'walking', f"Expected walking, got {c.classify_dn('1001')}"
    assert c.classify_dn('1002') == 'turn_left', f"Expected turn_left, got {c.classify_dn('1002')}"
    assert c.classify_dn('1003') == 'turn_right', f"Expected turn_right, got {c.classify_dn('1003')}"
    assert c.classify_dn('1004') == 'stop', f"Expected stop, got {c.classify_dn('1004')}"
    assert c.classify_dn('1005') == 'stop', f"Expected stop, got {c.classify_dn('1005')}"
    assert c.classify_dn('1006') == 'unknown', f"Expected unknown, got {c.classify_dn('1006')}"
    print("  ✓ Single classification: PASS")

    # Test batch classification
    counts = c.classify_dns(['1001', '1002', '1003', '1004', '1005', '1006'])
    assert counts['walking'] == 1
    assert counts['turn_left'] == 1
    assert counts['turn_right'] == 1
    assert counts['stop'] == 2
    assert counts['unknown'] == 1
    print(f"  ✓ Batch classification: {counts}")

    # Test gait mode determination
    assert c.determine_gait_mode((0.01, 0.05), 0.10) == 'walk', "Ahead → walk"
    assert c.determine_gait_mode((0.30, 0.05), 0.10) == 'turn_left', "Right → turn_left"
    assert c.determine_gait_mode((-0.30, 0.05), 0.10) == 'turn_right', "Left → turn_right"
    assert c.determine_gait_mode((0.0, 0.0), 0.01) == 'stance', "No odor → stance"
    assert c.determine_gait_mode((0.0, 0.0), 0.0) == 'stance', "Zero → stance"
    print("  ✓ Gait mode determination: PASS")

    os.unlink(test_path)
    print("\n✅ All DNSubtypeClassifier tests passed!")
