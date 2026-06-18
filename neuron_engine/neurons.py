"""Spiking neuron models for the BrainSimII simulation engine.

Implements 7 neuron model types:
- IF (0): Integrate & Fire — charges via synapses, fires at threshold 1.0
- COLOR (1): RGB passthrough — always fires, outputs an integer color value
- FLOATVALUE (2): Float passthrough — never fires, just stores a float
- LIF (3): Leaky I&F — decays each cycle, supports axon delay via bit-shift
- RANDOM (4): Random firing — intervals from Box-Muller normal distribution
- BURST (5): Burst firing — triggered spike trains
- ALWAYS (6): Continuous — fires every cycle

Port of NeuronEngine::NeuronBase (C++) to pure Python.
"""

import math
import random
import threading
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .synapses import Synapse

# ── Module Constants ─────────────────────────────────────────────────────────

THRESHOLD: float = 1.0
"""Firing threshold; neuron fires when current_charge >= THRESHOLD."""


class NeuronModel(IntEnum):
    """Spiking neuron model types (matches C# Neuron.modelType / C++ model enum)."""

    IF = 0        # Integrate & Fire
    COLOR = 1     # RGB value passthrough
    FLOATVALUE = 2  # Float value passthrough (never fires)
    LIF = 3       # Leaky Integrate & Fire
    RANDOM = 4    # Random firing with normal distribution
    BURST = 5     # Burst firing (spike trains)
    ALWAYS = 6    # Continuous firing


# ── NeuronBase ────────────────────────────────────────────────────────────────

class NeuronBase:
    """Core neuron class — the computational unit of the spiking network.

    Uses __slots__ for memory efficiency when scaling to millions of neurons.

    Port of C++ NeuronEngine::NeuronBase.
    """

    __slots__ = (
        'id', 'model', 'last_charge', 'current_charge',
        'leak_rate', 'axon_delay', 'axon_counter',
        'last_fired', 'refractory_delay', 'label',
        'synapses_out', 'synapses_from', 'gated',
        'next_firing', '_lock', '_has_plastic',
    )

    def __init__(
        self,
        neuron_id: int = 0,
        model: NeuronModel = NeuronModel.IF,
        leak_rate: float = 0.1,
        axon_delay: int = 0,
        refractory_delay: int = 0,
        label: str = "",
    ) -> None:
        """Initialize a neuron with default values.

        Args:
            neuron_id: Unique neuron identifier in the array.
            model: Neuron model type (IF, LIF, Color, etc.).
            leak_rate: Leak rate for LIF neurons; stddev for Random; ISI for Burst.
            axon_delay: Mean firing interval for Random; burst count for Burst.
            refractory_delay: Minimum cycles between firings.
            label: Human-readable label.
        """
        self.id: int = neuron_id
        self.model: NeuronModel = model
        self.last_charge: float = 0.0
        self.current_charge: float = 0.0
        self.leak_rate: float = leak_rate
        self.axon_delay: int = axon_delay
        self.axon_counter: int = 0
        self.last_fired: int = -9999  # Far in the past
        self.refractory_delay: int = refractory_delay
        self.label: str = label

        # Outgoing and incoming synapses
        self.synapses_out: List['Synapse'] = []
        self.synapses_from: List['Synapse'] = []

        # Gating state
        self.gated: bool = False

        # Optimization: tracks whether neuron has any plastic/learning synapses
        self._has_plastic: bool = False

        # Used by Always, Random, Burst models for timing
        self.next_firing: int = 0

        # Thread safety for concurrent charge addition
        self._lock: threading.Lock = threading.Lock()

    # ── Core Firing Algorithm ────────────────────────────────────────────────

    def fire1(self, cycle: int) -> bool:
        """Phase 1 — Evaluate: determine if the neuron fires this cycle.

        Called from NeuronArrayBase._process_neurons_1() in parallel.

        Args:
            cycle: Current simulation cycle number.

        Returns:
            True if the neuron emitted a spike, False otherwise.
        """
        # 1. DISABLED CHECK
        if self.leak_rate == -1.0 and self.current_charge == 0.0:
            return False

        # 2. MODEL-SPECIFIC PRE-FIRING
        model = self.model

        if model == NeuronModel.COLOR:
            # Color neurons always report as firing (display update)
            return True

        if model == NeuronModel.ALWAYS:
            if self.leak_rate >= 0:
                self.next_firing -= 1
            if self.next_firing <= 0:
                self.current_charge += 1.0
            return True

        if model == NeuronModel.RANDOM:
            if self.leak_rate >= 0:
                self.next_firing -= 1
            if self.next_firing <= 0:
                self.current_charge += 1.0

        if model == NeuronModel.BURST:
            if self.current_charge < 0:
                self.axon_counter = 0
            if self.axon_counter > 0:
                self.next_firing -= 1
                if self.next_firing <= 0:
                    self.axon_counter -= 1
                    self.current_charge += 1.0
                    if self.axon_counter > 0:
                        self.next_firing = max(1, int(self.leak_rate))
            if self.axon_counter == 0:
                self.axon_counter -= 1  # Keep track; goes negative when done

        # 3. REFRACTORY PERIOD
        if cycle < self.last_fired + self.refractory_delay:
            self.current_charge = 0.0
            return False

        # 4. CHARGE CLAMP (skip for FloatValue, it stores arbitrary values)
        if model != NeuronModel.FLOATVALUE:
            if self.current_charge < 0:
                self.current_charge = 0.0

        # 5. TRACK CHARGE CHANGES (display)
        if self.current_charge != self.last_charge:
            self.last_charge = self.current_charge

        # 6. LIF AXON DELAY PROPAGATION
        if model == NeuronModel.LIF and self.axon_counter != 0:
            self.axon_counter = self.axon_counter >> 1  # Shift delay counter
            if self.axon_counter & 0x1:
                return True  # Delayed spike arrives NOW

        # 7. FIRING THRESHOLD CHECK
        if self.current_charge >= THRESHOLD:
            if model == NeuronModel.LIF and self.axon_delay > 0:
                # Set up delayed firing via bit-shift counter
                self.axon_counter |= (1 << self.axon_delay)
                self.last_fired = cycle
                self.current_charge = 0.0
                return False  # Will fire later via axon delay

            if model == NeuronModel.BURST:
                self.next_firing = max(1, int(self.leak_rate))
                self.axon_counter = self.axon_delay - 1

            if model == NeuronModel.ALWAYS:
                self.next_firing = self.axon_delay

            if model == NeuronModel.RANDOM:
                self._set_next_firing()
                if self.next_firing < 1:
                    self.next_firing = 1

            # Common post-fire reset
            self.current_charge = 0.0
            self.last_fired = cycle
            return True

        # 8. LIF LEAKAGE (did not fire)
        if model == NeuronModel.LIF:
            self.current_charge *= (1.0 - self.leak_rate)

        return False

    def _set_next_firing(self) -> None:
        """Generate next firing interval using Box-Muller transform.

        Produces normally-distributed intervals with:
            mean = axon_delay
            stddev = leak_rate

        Clamped to minimum of 1 cycle.
        """
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(max(1e-10, u1)))
        theta = 2.0 * math.pi * u2
        self.next_firing = int(self.axon_delay + self.leak_rate * r * math.cos(theta))
        if self.next_firing < 1:
            self.next_firing = 1

    # ── Phase 2: Propagation ─────────────────────────────────────────────────

    def fire2(self, cycle: int) -> List[int]:
        """Phase 2 — Propagate: distribute spikes to target neurons.

        Called from NeuronArrayBase._process_neurons_2() in parallel.

        Returns list of target neuron IDs that received charge (for cascade).

        Args:
            cycle: Current simulation cycle number.
        """
        # FloatValue never propagates
        if self.model == NeuronModel.FLOATVALUE:
            return []

        # Color neurons do not propagate
        if self.model == NeuronModel.COLOR and self.last_charge != 0:
            return []

        # Check if gated
        if self.is_gated(cycle):
            return []

        # Propagate to all outgoing synapse targets
        charged: List[int] = []
        with self._lock:
            for synapse in self.synapses_out:
                tid = self._propagate_synapse(synapse, cycle)
                if tid is not None:
                    charged.append(tid)
        return charged

    def _propagate_synapse(self, synapse: 'Synapse', cycle: int) -> None:
        """Propagate a single synapse's weight to its target.

        Args:
            synapse: The outgoing synapse.
            cycle: Current cycle (for gating checks).
        """
        from .synapses import SynapseModel

        # Gate and Learn synapses don't propagate charge
        if synapse.model in (SynapseModel.GATE, SynapseModel.LEARN):
            return

        target = synapse.target_neuron
        if target is None or target.id == self.id:
            return  # Skip self-references and missing targets

        # Check target gating
        if target.is_gated(cycle):
            return

        # Deliver charge atomically
        target.add_to_current_value(synapse.weight)
        return target.id  # Cascade: report target for re-evaluation

    # ── Phase 3: Learning (Plasticity) ────────────────────────────────────────

    def fire3(self, cycle: int) -> None:
        """Phase 3 — Learn: synaptic plasticity and weight updates.

        Called from NeuronArrayBase._process_neurons_3() serially.

        Args:
            cycle: Current simulation cycle number.
        """
        from .synapses import SynapseModel

        # FloatValue and Color don't participate in learning
        if self.model == NeuronModel.FLOATVALUE:
            return
        if self.model == NeuronModel.COLOR and self.last_charge != 0:
            return

        if self.is_gated(cycle):
            return

        # Hebbian2 learning (competitive)
        self._hebbian2_learn(cycle)

        # Hebbian3 forgetting (exponential decay)
        self._hebbian3_forget()

        # Hebbian1/3 negative learning (target fired first)
        self._hebbian_negative(cycle)

        # Hebbian1/3 positive learning (source fired first)
        self._hebbian_positive(cycle)

    def _hebbian2_learn(self, cycle: int) -> None:
        """Competitive Hebbian learning: weights depend on active input count.

        Controlled by incoming Learn-type synapses.
        """
        from .synapses import SynapseModel

        # Check for Learn synapse signals
        learning_mode = 0  # 0=none, 1=learn, -1=erase
        for syn in self.synapses_from:
            if syn.model != SynapseModel.LEARN:
                continue
            source = syn.source_neuron
            if source is None:
                continue
            if cycle - source.last_fired <= 5:
                if syn.weight > 0:
                    learning_mode = 1
                elif syn.weight < 0:
                    learning_mode = -1

        if learning_mode == 0:
            return

        # Competitive weight lookup table
        competitive_weights = {1: 0.4, 2: 0.25, 3: 0.15, 4: 0.1, 5: 0.07}
        # 6+ maps to 0.06

        for syn in self.synapses_from:
            if syn.model != SynapseModel.HEBBIAN2:
                continue

            # Count how many Hebbian2 inputs were recently active
            active_count = 0
            for other in self.synapses_from:
                if other is syn or other.model != SynapseModel.HEBBIAN2:
                    continue
                src = other.source_neuron
                if src is not None and cycle - src.last_fired <= 5:
                    active_count += 1

            # Current synapse active too (adds 1)
            active_count += 1
            active_count = min(active_count, 6)

            new_weight = competitive_weights.get(active_count, 0.06)

            if learning_mode == 1:
                syn.weight = new_weight
            elif learning_mode == -1:
                syn.weight = 0.0

    def _hebbian3_forget(self) -> None:
        """Hebbian3 exponential weight decay (forgetting)."""
        from .synapses import SynapseModel

        for syn in self.synapses_from:
            if syn.model == SynapseModel.HEBBIAN3:
                syn.weight *= 0.99999
                if syn.weight < 0.2:
                    syn.weight = 0.2

    def _hebbian_negative(self, cycle: int) -> None:
        """Weaken outgoing Hebbian1/3 synapses when target fired first."""
        from .synapses import SynapseModel

        for syn in self.synapses_out:
            if syn.model not in (SynapseModel.HEBBIAN1, SynapseModel.HEBBIAN3):
                continue
            target = syn.target_neuron
            if target is None:
                continue

            # Target fired before source within 1-2 cycles
            delta = target.last_fired - self.last_fired
            if 1 <= delta <= 2:
                syn.weight *= (1.0 - 0.5 / delta)
                # Clamp
                if syn.model == SynapseModel.HEBBIAN1:
                    if syn.weight < 0.01:
                        syn.weight = 0.01
                else:  # HEBBIAN3
                    if syn.weight < 0.2:
                        syn.weight = 0.2

    def _hebbian_positive(self, cycle: int) -> None:
        """Strengthen incoming Hebbian1/3 synapses when source fired first."""
        from .synapses import SynapseModel

        for syn in self.synapses_from:
            if syn.model not in (SynapseModel.HEBBIAN1, SynapseModel.HEBBIAN3):
                continue
            source = syn.source_neuron
            if source is None:
                continue

            # Source fired before or at same time as target, within 5 cycles
            delta = source.last_fired - self.last_fired
            if 0 <= delta <= 5:
                delta = max(1, delta)
                syn.weight /= (1.0 - 0.5 / delta)

                # Clamp to model-specific ranges
                if syn.model == SynapseModel.HEBBIAN1:
                    if syn.weight > 0.6:
                        syn.weight = 0.6
                    if syn.weight < 0.01:
                        syn.weight = 0.01
                else:  # HEBBIAN3
                    if syn.weight > 1.1:
                        syn.weight = 1.1
                    if syn.weight < 0.2:
                        syn.weight = 0.2

    # ── Gating ────────────────────────────────────────────────────────────────

    def is_gated(self, cycle: int) -> bool:
        """Check if this neuron is gated (blocked from firing/propagating).

        Evaluates incoming Gate-type synapses:
        - Negative gate weight: unconditionally blocks if source was recently active
        - Positive gate weight: enables only if source was recently active
        - Negative gates take priority over positive gates

        Args:
            cycle: Current simulation cycle number.

        Returns:
            True if the neuron is blocked (gated).
        """
        from .synapses import SynapseModel

        gated_state = 0  # 0=normal, 1=enabled by positive gate, -1=disabled

        for syn in self.synapses_from:
            if syn.model != SynapseModel.GATE:
                continue
            source = syn.source_neuron
            if source is None:
                continue

            if syn.weight < 0:
                # Negative gate: blocks if source was active within 3 cycles
                if source.last_fired >= cycle - 3:
                    return True  # Unconditional block
            elif syn.weight > 0:
                # Positive gate: enables only when source was recently active
                if source.last_fired < cycle - 3:
                    gated_state = -1  # Source inactive → disable
                elif gated_state == 0:
                    gated_state = 1   # Source active → enable

        return gated_state < 0

    # ── Thread-Safe Charge Accumulation ───────────────────────────────────────

    def add_to_current_value(self, weight: float) -> None:
        """Thread-safe addition to the neuron's membrane potential.

        Used by fire2() propagation to accumulate incoming spikes.

        Args:
            weight: Synaptic weight to add to current_charge.
        """
        with self._lock:
            self.current_charge += weight

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the neuron to initial state."""
        self.current_charge = 0.0
        self.last_charge = 0.0
        self.axon_counter = 0
        self.next_firing = 0
        self.last_fired = -9999
        self.gated = False

    def __repr__(self) -> str:
        return (
            f"Neuron(id={self.id}, model={self.model.name}, "
            f"charge={self.current_charge:.3f}, last_fired={self.last_fired})"
        )
