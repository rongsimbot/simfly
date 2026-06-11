"""Synapse models for SimFly neuron engine (ported from C++ NeuronEngine)."""
from enum import IntEnum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .neurons import NeuronBase


class SynapseModel(IntEnum):
    """Synapse plasticity model types matching C++ NeuronEngine."""
    FIXED = 0
    BINARY = 1
    HEBBIAN1 = 2
    HEBBIAN2 = 3
    HEBBIAN3 = 4
    GATE = 5
    LEARN = 6


class Synapse:
    """A synapse connecting source to target neuron."""
    
    __slots__ = ("target_neuron_id", "source_neuron_id", "weight", "model",
                 "target_neuron", "source_neuron")
    
    def __init__(self,
                 target_neuron_id: int = 0,
                 source_neuron_id: int = 0,
                 weight: float = 0.0,
                 model: SynapseModel = SynapseModel.FIXED):
        self.target_neuron_id = target_neuron_id
        self.source_neuron_id = source_neuron_id
        self.weight = weight
        self.model = model
        self.target_neuron = None
        self.source_neuron = None
    
    def __repr__(self):
        return (f"Synapse(pre={self.source_neuron_id}->post={self.target_neuron_id}, "
                f"w={self.weight:.4f}, model={self.model.name})")
