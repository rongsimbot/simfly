"""
cpp_engine.py — Python ctypes bindings for libneuronengine.so (C++ spiking neuron engine)

Usage:
    from cpp_engine import CppEngine
    eng = CppEngine("/path/to/libneuronengine.so")
    eng.create(30000, n_threads=4)
    for i in range(30000):
        eng.set_neuron(i, model=3, leak=0.03)
    for pre, post, weight in synapses:
        eng.add_synapse(pre, post, weight)
    eng.add_charge(0, 2.0)
    fired = eng.fire()  # list of fired neuron IDs
"""
import ctypes
import os
from typing import List, Optional


class CppEngine:
    """Python wrapper around libneuronengine.so"""
    
    MODEL_LIF = 3
    
    def __init__(self, lib_path: str | None = None):
        if lib_path is None:
            # Search in standard locations
            candidates = [
                "libneuronengine.so",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "libneuronengine.so"),
                "/tmp/cpp_engine/libneuronengine.so",
            ]
        else:
            candidates = [lib_path]
        
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        
        if found is None:
            raise FileNotFoundError(f"libneuronengine.so not found. Searched: {candidates}")
        
        self.lib = ctypes.CDLL(os.path.abspath(found))
        self._setup()
        self._eng: Optional[ctypes.c_void_p] = None
        self._n = 0
        self._buf: Optional[ctypes.Array] = None
    
    def _setup(self):
        L = self.lib
        L.engine_create.argtypes = [ctypes.c_int, ctypes.c_int]
        L.engine_create.restype = ctypes.c_void_p
        L.engine_destroy.argtypes = [ctypes.c_void_p]
        L.engine_destroy.restype = None
        L.engine_set_neuron.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
        L.engine_set_neuron.restype = None
        L.engine_add_synapse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        L.engine_add_synapse.restype = None
        L.engine_set_charge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        L.engine_set_charge.restype = None
        L.engine_add_charge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        L.engine_add_charge.restype = None
        L.engine_fire.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.engine_fire.restype = ctypes.c_int
        L.engine_get_size.argtypes = [ctypes.c_void_p]
        L.engine_get_size.restype = ctypes.c_int
        L.engine_reset.argtypes = [ctypes.c_void_p]
        L.engine_reset.restype = None
        L.engine_get_synapse_count.argtypes = [ctypes.c_void_p]
        L.engine_get_synapse_count.restype = ctypes.c_longlong
    
    def create(self, n_neurons: int, n_threads: int = 4):
        if self._eng: self.destroy()
        self._n = n_neurons
        self._eng = self.lib.engine_create(n_neurons, n_threads)
        self._buf = (ctypes.c_int * (n_neurons + 64))()
    
    def destroy(self):
        if self._eng:
            self.lib.engine_destroy(self._eng)
            self._eng = None
    
    def set_neuron(self, i: int, model: int = 3, leak: float = 0.03, refractory: int = 0):
        self.lib.engine_set_neuron(self._eng, i, model, leak, refractory)
    
    def add_synapse(self, pre: int, post: int, weight: float):
        self.lib.engine_add_synapse(self._eng, pre, post, weight)
    
    def set_charge(self, i: int, charge: float):
        self.lib.engine_set_charge(self._eng, i, charge)
    
    def add_charge(self, i: int, charge: float):
        self.lib.engine_add_charge(self._eng, i, charge)
    
    def fire(self) -> List[int]:
        n = self.lib.engine_fire(self._eng, self._buf, self._n + 64)
        return [self._buf[i] for i in range(n)]
    
    def reset(self):
        self.lib.engine_reset(self._eng)
    
    def size(self) -> int:
        return self.lib.engine_get_size(self._eng)
    
    def synapse_count(self) -> int:
        return self.lib.engine_get_synapse_count(self._eng)
    
    def __del__(self): self.destroy()
    def __enter__(self): return self
    def __exit__(self, *a): self.destroy()
