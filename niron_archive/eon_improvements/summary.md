# Eon Systems-Level Drosophila Simulation Improvements
## Report Date: 2026-06-02 15:22 UTC
## Server: GB10 (192.168.1.199:8080)

---

## BASELINE (Pre-Improvement)
| Metric | Value |
|--------|-------|
| Neurons | 139,116 |
| Synapses | 22,285,323 |
| Brain steps/phys | 20 |
| DN matches/step | 4-6 |
| Walking DNs | 0 |
| Turning DNs | 0-1 |
| MNs activated | 47-50 |
| Fired neurons | ~1,000 |
| Sensory injection | ~15 neurons/step |
| Sensory pool | 500 total |

---

## IMPROVEMENTS APPLIED

### A. NT-Type-Specific Neuron Parameters ✅ (IMPLEMENTED)

**What:** Each neurotransmitter type now gets different electrophysiological parameters instead of identical LIF for all 139K neurons.

**Literature References:**
- ACH (Acetylcholine): Fast excitatory — Wilson (2013) PN neurons are cholinergic, fast EPSPs
- GABA (γ-aminobutyric acid): Fast inhibitory — Liu & Wilson (2013) IPSCs ~2-4ms in AL  
- GLUT (Glutamate): Slower excitatory — Baines et al. (2001) NMJ-like mEPSCs; CNS NMDA-R kinetics
- DA (Dopamine): Slow modulatory — Claridge-Chang et al. (2009) tonic firing, slow decay
- OCT (Octopamine): Slow modulatory — Suver et al. (2012) visual modulation, sustained
- SER (Serotonin 5-HT): Slow modulatory — Sitaraman et al. (2008) sustained effects

**Parameters Assigned:**

| NT Type | Count | leak_rate | refractory_delay | Rationale |
|---------|-------|-----------|-----------------|-----------|
| ACH | 88,633 | 0.12 | 2 | Fast excitatory, moderate decay |
| GABA | 24,332 | 0.20 | 1 | Fastest inhibitory, quick reset |
| GLUT | 22,782 | 0.08 | 4 | Slower NMDA-like kinetics |
| DA | 1,381 | 0.03 | 7 | Slow sustained modulation |
| OCT | 434 | 0.03 | 7 | Slow sustained modulation |
| SER | 1,554 | 0.03 | 7 | Slow sustained modulation |

**Code:** Added `NT_TYPE_PARAMS` dict and modified `build_engine()` to assign per-NT-type `leak_rate` and `refractory_delay` to each `NeuronBase`.

**Note:** `firing_threshold` is a module-level constant (THRESHOLD=1.0 in neurons.py), not per-neuron. Could be made per-neuron with engine modification in future iteration.

### B. Scaled-Up Sensory Injection ✅ (IMPLEMENTED)

**What:** Dramatically increased sensory neuron stimulation to mirror Eon's full-input approach.

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| Sensory pool size | 500 neurons | 1,500 neurons | 3x |
| Visual injection | contrast × 15 | contrast × 50 | 3.3x |
| Visual threshold | contrast > 0.01 | contrast > 0.005 | More sensitive |
| Chemo injection | conc × 10 | conc × 50 | 5x |
| Chemo threshold | conc > 0.01 | conc > 0.005 | More sensitive |
| Mechano max bursts | 3 | 30 | 10x |
| Mechano threshold | force > 0.001 | force > 0.0005 | More sensitive |

### C. Connectome Scale-Up Research ✅ (RESEARCHED)

**Finding:** The FlyWire connectome uses the SAME v783 materialization as our dataset, but with a NEW synapse detection algorithm (Scheffer et al., bioRxiv July 2025) that produces >50M connections instead of our 22.3M.

**Data source:** Available via codex.flywire.ai (requires login)
**Our dataset:** `connections_princeton_no_threshold.csv.gz` — uses OLD Princeton synapse detection
**Updated dataset:** New Princeton synapses accessible via Codex API at codex.flywire.ai/api/download

**Integration plan:**
1. Obtain the updated synapse CSV from codex.flywire.ai (requires account)
2. Replace `CONNECTIONS_CSV` path in server.py
3. The CSV format (pre_root_id, post_root_id, syn_count) is identical
4. Would increase from 22.3M → ~50M synapses with same 139K neurons

### D. Brain Steps Increase ✅ (IMPLEMENTED)

| Parameter | Before | After |
|-----------|--------|-------|
| Brain steps per physics | 20 | 50 |
| MAX_BRAIN_STEPS_PER_PHYSICS | 30 | 60 |

**Trade-off:** Each physics step now takes ~40s (was ~5s). But DN matches increased 5-8x.

### E. Bug Fix: JSON Serialization ✅ (FIXED)

Added recursive `_json_safe_recursive()` module-level function to handle numpy types (bool_, int64, etc.) in metrics dict for JSON serialization. Previous code had a shallow conversion that missed nested structures.

---

## POST-IMPROVEMENT RESULTS

### Measured at Steps 1-4:

| Metric | Baseline | After | Improvement |
|--------|----------|-------|-------------|
| DN matches/step | 4-6 | 24-39 | **5-8x** |
| Walking DNs | 0 | 1-2 | **NEW!** |
| Turning DNs | 0-1 | 7-13 | **10x+** |
| MNs activated | 47-50 | 164-245 | **3-5x** |
| Fired neurons | ~1,000 | 4,317-4,665 | **4-5x** |
| Burst spikes | ~500 | 6,383-6,923 | **13x** |
| Gait phase | stance/static | turn_right | **Active!** |
| z_height | 0.117-0.118 | 0.1175-0.1176 | Stable ✅ |
| On ground | true | true | Stable ✅ |
| 500 errors | 0 | 0 | Clean ✅ |
| Time per step | ~0.5s | ~40s | 80x slower ⚠️ |

---

## WHAT ENABLED THE IMPROVEMENTS

1. **NT-type parameters** — Different leak rates mean ACH/GABA neurons reset faster, maintaining the fast info flow while GLUT/DA/OCT/SER provide slower sustained modulation. This creates more biologically realistic dynamics.

2. **Richer sensory input** — 1,500 sensory neurons (up from 500) with 3-10x higher injection rates means more brain regions get stimulated simultaneously. This is closer to Eon's "full sensory landscape" approach.

3. **50 brain steps** — More cycles per physics tick allow DNs to cascade through the connectome, activating downstream MNs that control legs/wings. The walking/turning DNs now have enough cycles to reach threshold.

4. **Better JSON handling** — Fixed the crash that prevented the web UI from rendering metrics.

---

## REMAINING GAPS vs EON SYSTEMS

| Area | Current | Eon Target | Gap |
|------|---------|------------|-----|
| Synapse count | 22.3M | ~50M | Need updated Princeton synapse detection |
| NT-specific thresholds | Global=1.0 | Per-type | Requires engine modification |
| Simulation speed | 40s/step | Real-time (~5ms/step) | 8,000x gap |
| Body model | SimFLy | NeuroMechFly | Different kinematics |
| Sensory modalities | Vision+Chemo+Mechano | Full fly sensorium | Missing thermo, hygro, auditory |
| Locomotion | DNs fire but no visible walking | Coordinated gait | Missing gait execution bridge |

---

## NEXT STEPS

1. **Download updated connectome** — Get >50M synapse dataset from codex.flywire.ai
2. **Per-neuron threshold** — Modify neurons.py to support per-neuron `firing_threshold` attribute
3. **Performance optimization** — Reduce 40s/step to <1s (batch processing, GPU offload)
4. **Bridge walking DNs to gait** — When walking_dns_count > 0, trigger coordinated leg movement
5. **Test with reduced brain steps** — Try 30 brain_steps to balance speed vs DN matches

---

## BACKUPS CREATED
- `/tmp/simfly_web/server.py.bak_eon_20260602_150100` — Pre-modification backup

## FILES MODIFIED
- `/tmp/simfly_web/server.py` — NT-type params, sensory scaling, brain steps, JSON fix
