# Phase A: RL-Enhanced Torque Decoder — Status Report

**Date:** 2026-06-07
**Agent:** SimTome (subagent)
**Trello Cards:** [Phase A (6a233b0328cf9b6ff8424c02)](https://trello.com/c/6a233b0328cf9b6ff8424c02) + [duplicate (6a233b34c1a8f45566267ba6)](https://trello.com/c/6a233b34c1a8f45566267ba6)

---

## 📦 Deliverables Created

### 1. Real Pipeline Adapter — `rl_training/rl_simfly_pipeline.py` (GB10)
- **Path:** `/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/rl_training/rl_simfly_pipeline.py`
- **635 lines** — Full biological pipeline wrapped for RL
- Implements protocol expected by `rl_bridge.py`'s `SimFlyRLEnv`
- Functions: `reset()`, `get_observation()`, `get_connectome_torques()`, `apply_torques()`, `step_physics()`, `get_state()`
- Includes: BurstInjector, sensory neuron identification, engine builder, MuJoCo init
- Observation space: joint angles (36) + ground contact (36) + exteroceptive (3) = 75 dims

### 2. Training Script — `rl_training/train_rl_torque.py` (GB10)
- **Path:** `/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/rl_training/train_rl_torque.py`
- **427 lines** — Main training entry point
- Modes: training, eval-only (`--eval-only`), compare-only (`--compare-only`)
- Produces: `train_log.jsonl`, policy snapshots, `comparison_report.json`, `training_metrics.png`
- Run with: `MUJOCO_GL=egl python3 train_rl_torque.py --iterations 100 --neurons 2000`

### 3. Existing RL Framework — `rl_bridge.py` (GB10, pre-existing)
- **Path:** `/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/rl_bridge.py`
- Already complete: MLP, Adam, ConnectomeModulationPolicy, SimFlyRLEnv, PPO trainer
- `apply_modulation()` — scientific integrity: `torque[j] * gain[j] + bias[j]`
- ShuffledConnectomePipeline — honesty control (degree-preserving permutation)
- Smoke test: `python3 rl_bridge.py --smoke`

### 4. Eon Systems Competitive Analysis — `eon_systems_analysis.md` (workspace)
- Key finding: Eon uses PPO from scratch; we use connectome as architecture + PPO for calibration
- Our advantage: biological signal chain is the architecture; RL only tunes torque gain/bias
- Honesty control: shuffled connectome pipeline proves connectome carries signal

---

## 🔬 Current State: Active Phases

### Phase 2: Spiking Neuron Engine ✅ COMPLETE (card needs move)
- **Code:** `neuron_engine/engine.py`, `neuron_engine/neurons.py`, `neuron_engine/synapses.py` — ALL EXIST
- **Status:** NIRON engine is live and driving the pipeline
- **Action needed:** Move Trello card from "Active (Agents)" to "Done" (idList: `6a10517fafd83525e173cc17`)

### Phase 6: Virtual Fly Behavior Demos ⚠️ PARTIALLY COMPLETE
- **Two cards:** One in Done (✅) and one still in Active (Agents)
- **Code:** `demos/walking_demo.py`, `demos/feeding_demo.py`, `demos/grooming_demo.py`, `demos/escape_demo.py`, `demos/run_all_demos.py` — ALL EXIST
- **Status:** Demos exist but may use scripted motor patterns (need verification against SCIENTIFIC RIGOR DIRECTIVE)
- **Action needed:** Verify demos are connectome-driven, then move card to Done; archive duplicate

### Phase 7: Brain Activity Visualization 🔄 IN PROGRESS
- **Code:** `visualization/gui_main.py`, `visualization/raster_plot.py`, `visualization/network_graph.py`, `visualization/membrane_trace.py`, `visualizer/` — ALL EXIST
- **Status:** Visualization code is extensive (GUI, raster, network, membrane). Not yet integrated into web dashboard.
- **Action needed:** Continue integration with server; keep in Active

### Phase 8: White Paper Pipeline 🔄 IN PROGRESS
- **Papers exist:**
  - `white_papers/phase1_drosophila_foundation.md` (24KB)
  - `white_papers/NIRON-WP-002_Competitive_Analysis.md` (32KB)
  - `white_papers/NIRON-WP-003_Integration_Report.md` (15KB)
  - `white_papers/NIRON-WP-004_Gap_Analysis.md` (10KB)
  - `white_papers/NIRON-WP-007_RL_Bridge_Architecture.md` (18KB)
- **Missing:** WP-001 (core architecture), Phase 11 connectome-driven movement paper
- **Action needed:** Write Phase 11 milestone paper; keep in Active

---

## 🧪 How to Run RL Training

```bash
# On GB10 (192.168.1.199):
cd /home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model

# 1. Quick sanity check — mock pipeline smoke test:
/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 rl_bridge.py --smoke

# 2. Full RL training on connectome pipeline:
DISPLAY=:10 MUJOCO_GL=egl \
/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 \
rl_training/train_rl_torque.py \
  --iterations 100 \
  --neurons 2000 \
  --joints 36 \
  --rollout 1024 \
  --lr 3e-4 \
  --hidden 128 \
  --output rl_training/rl_training_output

# 3. Evaluate a saved policy:
DISPLAY=:10 MUJOCO_GL=egl \
/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 \
rl_training/train_rl_torque.py \
  --eval-only rl_training/rl_training_output/best_policy.npz
```

---

## 🎯 Success Criteria Status

| Criterion | Status | Details |
|-----------|--------|---------|
| MLP policy network built | ✅ | `rl_bridge.py` MLP class with analytic backprop |
| PPO trainer implemented | ✅ | `train_ppo()` in `rl_bridge.py` |
| Real pipeline integrated | ✅ | `rl_simfly_pipeline.py` wraps full connectome pipeline |
| Comparison framework ready | ✅ | `train_rl_torque.py --compare-only` mode |
| Food source reach test designed | ✅ | Chemotaxis reward shaping in SimFlyRLEnv |
| Training script deployable | ✅ | Can run on GB10 with venv |
| Honesty control implemented | ✅ | `ShuffledConnectomePipeline` in `rl_bridge.py` |

---

## ⏭️ Next Steps

1. **Run training on GB10** — Start with 100 iterations overnight
2. **Verify connectome-driven demos** — Phase 6 demos should use real connectome
3. **Move Phase 2 + Phase 6 duplicate to Done**
4. **Write Phase 11 milestone white paper** (WP-005)
5. **Integrate visualization into web dashboard** (Phase 7)
6. **Consider Phase B: LIF Parameter Optimization** (card already in Incoming)
