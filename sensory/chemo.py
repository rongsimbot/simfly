#!/usr/bin/env python3
"""
Phase 10: Sensory-Driven Chemosensory Module — Sugar Gradient Detection
Models Drosophila olfactory and gustatory receptor neuron responses
to chemical gradients in the environment.

FIXED (2026-06-11):
  - Food moved closer: default (2.0, 0.0, 0.0) for 9× stronger chemo signal
  - Sugar sigma widened: default 4.0 (was 2.0) for gradient reaching further

Architecture:
  Fly position → distance to sugar source → concentration computation
  → ORN/GRN activation → sensory injector → NIRON → DNs
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class SugarGradient:
    """Models a sugar source with concentration gradient.
    
    Concentration follows: C(d) = C_max / (1 + (d / sigma)²)
    where d is distance from source, sigma is spread parameter.
    
    FIX: Default sigma widened to 4.0 (was 2.0) for wider gradient reach.
    FIX: Default source moved closer to (2.0, 0.0, 0.0) for stronger chemo signal.
    """
    
    def __init__(
        self,
        source_position: Tuple[float, float, float] = (2.0, 0.0, 0.0),
        max_concentration: float = 1.0,
        sigma: float = 4.0,  # FIX: Widened from 2.0 → 4.0 for farther reach
    ):
        self.source_pos = np.array(source_position, dtype=np.float64)
        self.max_concentration = max_concentration
        self.sigma = sigma
    
    def concentration_at(self, position: np.ndarray) -> float:
        """Compute sugar concentration at given position.
        
        Args:
            position: 3D position (x, y, z).
            
        Returns:
            Normalized concentration [0, 1].
        """
        dist = np.linalg.norm(position[:2] - self.source_pos[:2])
        return self.max_concentration / (1.0 + (dist / self.sigma) ** 2)
    
    def gradient_at(self, position: np.ndarray) -> np.ndarray:
        """Compute concentration gradient vector at given position.
        
        Args:
            position: 3D position.
            
        Returns:
            Gradient vector (dx, dy) — direction of increasing concentration.
        """
        pos_2d = position[:2]
        src_2d = self.source_pos[:2]
        dist_vec = src_2d - pos_2d
        dist = np.linalg.norm(dist_vec)
        
        if dist < 0.001:
            return np.zeros(2)
        
        d_sq = (dist / self.sigma) ** 2
        denom = (1.0 + d_sq) ** 2
        factor = -2.0 * self.max_concentration / (self.sigma ** 2) / denom
        
        return factor * dist_vec


class GustatorySensor:
    """Models Drosophila gustatory receptor neurons (GRNs) for sugar detection.
    
    Sugar-sensitive GRNs (Gr5a, Gr64f-expressing) in labellum and tarsi.
    """
    
    def __init__(self, num_grns: int = 50, gain: float = 100.0):
        self.num_grns = num_grns
        self.gain = gain  # Hz per concentration unit
        self._concentration = 0.0
        self._rates = np.zeros(num_grns)
    
    def update(self, concentration: float, dt_ms: float = 1.0) -> np.ndarray:
        """Update GRN responses to sugar concentration."""
        self._concentration = concentration
        
        for i in range(self.num_grns):
            sensitivity = 0.5 + 0.5 * i / max(1, self.num_grns - 1)
            rate = self.gain * concentration * sensitivity
            self._rates[i] = rate
        
        return self._rates
    
    @property
    def concentration(self) -> float:
        return self._concentration
    
    @property
    def rates(self) -> np.ndarray:
        return self._rates
    
    @property
    def max_rate(self) -> float:
        return np.max(self._rates) if len(self._rates) > 0 else 0.0


class OlfactorySensor:
    """Models olfactory receptor neurons (ORNs) for odor gradient tracking."""
    
    def __init__(
        self,
        num_orn_types: int = 10,
        orns_per_type: int = 10,
        gain: float = 80.0,
    ):
        self.num_orn_types = num_orn_types
        self.orns_per_type = orns_per_type
        self.total_orns = num_orn_types * orns_per_type
        self.gain = gain
        self._rates = np.zeros(self.total_orns)
    
    def update(self, odor_concentration: float, dt_ms: float = 1.0) -> np.ndarray:
        """Update ORN responses."""
        for t in range(self.num_orn_types):
            tuning = 0.5 + 0.5 * (t % 5) / 5.0
            for n in range(self.orns_per_type):
                idx = t * self.orns_per_type + n
                self._rates[idx] = self.gain * odor_concentration * tuning
        return self._rates
    
    @property
    def rates(self) -> np.ndarray:
        return self._rates


class ChemoSensorySystem:
    """Combined chemosensory system for sugar gradient navigation.
    
    Coordinates both olfactory and gustatory sensing.
    
    FIX: Default source at (2.0, 0.0, 0.0) — much closer than (8.0, 3.0, 0.0)
         Default sigma widened to 4.0 for wider gradient
    """
    
    def __init__(
        self,
        sugar_source_pos: Tuple[float, float, float] = (2.0, 0.0, 0.0),
        sugar_sigma: float = 4.0,  # FIX: widened from 2.0
        num_grns: int = 50,
        num_orn_types: int = 10,
        use_odor_plume: bool = True,
    ):
        self.gradient = SugarGradient(
            source_position=sugar_source_pos,
            sigma=sugar_sigma,
        )
        self.gustatory = GustatorySensor(num_grns=num_grns)
        self.olfactory = OlfactorySensor(
            num_orn_types=num_orn_types,
            orns_per_type=10,
        )
        self.use_odor_plume = use_odor_plume
        
        self._current_concentration = 0.0
        self._gradient_direction = np.zeros(2)
        self._distance_to_source = float('inf')
    
    def read(self, fly_position: np.ndarray, dt_ms: float = 1.0) -> Dict:
        """Read chemosensory data at fly position."""
        sugar_conc = self.gradient.concentration_at(fly_position)
        grad = self.gradient.gradient_at(fly_position)
        dist = np.linalg.norm(fly_position[:2] - self.gradient.source_pos[:2])
        
        grn_rates = self.gustatory.update(sugar_conc, dt_ms)
        odor_conc = sugar_conc * 0.3 if self.use_odor_plume else 0.0
        orn_rates = self.olfactory.update(odor_conc, dt_ms)
        
        self._current_concentration = sugar_conc
        self._gradient_direction = grad / (np.linalg.norm(grad) + 1e-10)
        self._distance_to_source = dist
        
        return {
            'sugar_concentration': sugar_conc,
            'odor_concentration': odor_conc,
            'gradient_x': grad[0],
            'gradient_y': grad[1],
            'gradient_direction': self._gradient_direction.tolist(),
            'distance_to_source': dist,
            'grn_max_rate': self.gustatory.max_rate,
            'grn_rates': grn_rates.tolist(),
            'orn_rates': orn_rates.tolist(),
            'oriented_toward_source': sugar_conc > 0.01,
        }
    
    def get_grn_input(self) -> float:
        """Get gustatory input level [0, 1] for sensory injector."""
        return self._current_concentration
    
    def get_orn_input(self) -> float:
        """Get olfactory input level [0, 1] for sensory injector."""
        return self._current_concentration * 0.3
    
    @property
    def distance_to_source(self) -> float:
        return self._distance_to_source
    
    @property
    def concentration(self) -> float:
        return self._current_concentration
