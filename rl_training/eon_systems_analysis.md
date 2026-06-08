# Eon Systems Competitive Analysis: DRL for Motor Control

**Date:** 2026-06-07
**Author:** SimTome (R&D Chief Scientist, Connectomics)
**References:** Eon Systems whitepapers, Brunton Lab NeuroMechFly, FlyGM paper (arXiv:2602.17997)

---

## 1. Eon Systems' Approach

Eon Systems uses **Deep Reinforcement Learning (PPO)** to train an ANN that maps
**motor neuron (MN) activations → leg actuator torques** for robotic/embodied systems.
Key characteristics:

| Aspect | Eon Systems | SimFLy (Us) |
|--------|-------------|-------------|
| **Motor signal source** | Trained policy (tabula rasa) | FlyWire connectome → DN → MANC VNC |
| **Network architecture** | MLP policy network | Biologically-verified connectome graph |
| **Training paradigm** | PPO with reward shaping | PPO as *torque calibration only* |
| **Constraint encoding** | Implicit in reward function | Explicit in connectome structure |
| **Generalization** | Requires domain randomization | Inherent from biological circuit |

## 2. Where RL Enhances Our Pipeline

Our pipeline has an advantage Eon Systems cannot match: the **connectome provides
the architecture**. The FlyWire brain → MANC VNC pathway specifies *which* DNs
control *which* motor neurons. RL should NOT replace this — it should **calibrate**
the final torque mapping.

### RL Enhancement Points (NOT Replacement):

```
FlyWire DN firing → MANC MN activation → VNC Decoder → RAW TORQUE
                                                           ↓
                                                    RL Policy Network
                                                    (gain/bias per joint)
                                                           ↓
                                                    CALIBRATED TORQUE
```

**What RL learns:**
- Per-joint gain multiplier (amplify/suppress connectome torque)
- Per-joint bias offset (resting tension adjustment)
- Temporal smoothing of torque commands

**What RL does NOT touch:**
- The DN→MN bridge (biological pathways)
- The VNC decoder (MN type classification)
- The connectome firing patterns (sensory→interneuron→DN chain)

### Honesty Control (from FlyGM paper):
We implement a `ShuffledConnectomePipeline` — degree-preserving but
structure-destroying permutation. Training RL on both real and shuffled
connectomes lets us **prove the connectome carries genuine motor signal**:

> If real connectome achieves higher reward than shuffled control,
> the connectome structure contributes actionable information.

## 3. Our Competitive Advantage

| Advantage | Why It Matters |
|-----------|---------------|
| **Connectome-driven architecture** | 139K real neurons, 22.3M synapses, 953 DNs matched to MANC |
| **Biological signal chain** | Sensory → interneuron → DN → MN → joint (verified chain) |
| **DN subtype classification** | Walking, turning, stopping DNs classified from connectome data |
| **Gait controller** | Tripod gait from DN firing patterns (not scripted) |
| **MN-type-aware decoding** | Fast/slow/flexor/extensor MN types with biologically-derived gains |
| **Phase 11 proven** | Connectome-driven movement demonstrated (May 2026) |

## 4. RL Integration Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| RL overrides connectome signal | High | Use affine modulation (not replacement); start from gain=1, bias=0 |
| Training instability (fly falls) | Medium | Early termination + reset on fall; reward shaping |
| Compute cost (MuJoCo sim) | Medium | Reduced neuron count (2K vs 139K); batch training |
| Policy doesn't generalize | Medium | Multiple random init positions; shuffled control baseline |
| Integration complexity | Low | rl_bridge.py already has protocol; plug-and-play adapter |

## 5. Brunton Lab Context

Brunton Lab (NeuroMechFly) showed "any random connectome + RL produces walking."
This is the **wrong lesson** to draw for our work. Their finding actually 
**validates our honesty control**: if random connectomes can walk, we need the
shuffled-control comparison to prove our connectome is *better* than random.

Our Phase A objective is NOT "make the fly walk" (Phase 11 already proved that).
It's "make the fly walk toward food more efficiently than the fixed-gain baseline,
using connectome-driven torques calibrated by RL."

## 6. Recommendations

1. **Start small:** 2000 neurons, 36 joints, 100 PPO iterations
2. **Run honesty control:** Compare real vs shuffled connectome reward curves
3. **Document torque smoothness:** RMS of torque changes as a key metric
4. **Chemotaxis benchmark:** Time-to-food as primary success metric
5. **Publish comparison:** Fixed-gain vs RL policy performance table
