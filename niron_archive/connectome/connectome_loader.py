"""Minimal connectome loader stub for SimFly web server."""
class FlyWireConnectomeLoader:
    def __init__(self, connections_path=None, config=None):
        self.connections_path = connections_path
        self.config = config or {}
        self.flywire_to_idx = {}
        self.idx_to_flywire = []
        self.neuron_nt_types = {}
        self.nt_weight_map = {}
