# SimFly Project Plan

> **Last Updated:** June 18, 2026
> **Current Phase:** J — Merge Brian2 + Real Sensory into Live Dashboard
> **Live Dashboard:** http://192.168.1.199:8080 (C++ NIRON, 139K neurons)
> **Best Model:** Brian2 C++ Standalone (Phase H) — 138K neurons, real sensory routing, 0.06× RT

---

## 📊 Current Architecture (June 18, 2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                     SIMFLY ARCHITECTURE                          │
│                                                                  │
│  Live Dashboard (:8080)          Research Pipeline              │
│  ┌────────────────────┐         ┌──────────────────────┐       │
│  │ C++ NIRON Engine   │         │ Brian2 C++ Standalone │       │
│  │ libneuronengine.so │         │ Phase H binary        │       │
│  │ 139K neurons       │         │ 138K neurons          │       │
│  │ 19.8M synapses     │         │ 15M synapses          │       │
│  │ Scripted sensory   │         │ Real sensory -> DN    │       │
│  │ 1x real-time       │         │ 0.06x real-time       │       │
│  │ Web + MJPEG         │         │ Batch + raw frames    │       │
│  └────────────────────┘         └──────────────────────┘       │
│           │                              │                       │
│           │      PHASE J (MERGE)          │                       │
│           └──────────────────────────────┘                       │
│                              │                                   │
│                    ┌─────────▼──────────┐                       │
│                    │ Unified Dashboard  │                       │
│                    │ Brian2 + Real Sens │                       │
│                    │ + MJPEG + Metrics  │                       │
│                    └────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Completed Phases (June 2026)

### Phase D: Brian2 Integration ✅ (June 15)
- Ported from custom NIRON engine to Brian2 (community-standard simulator)
- Full connectome: 138,639 neurons, 15M synapses loaded into Brian2 LIF network
- Integrated Eon Systems' fly-brain architecture (Apache 2.0)
- Created brian2_body_bridge.py for DN->MN->MuJoCo mapping
- EGL headless rendering preserved

### Phase E: Brian2 Closed-Loop Walking ✅ (June 17)
- First whole-connectome walking: 138,639 neurons -> MuJoCo body
- PoissonGroup DN injection (150Hz, 30 DNs)
- 47.9mm walking in 10s at 0.006x real-time (numpy backend)
- Catalogued 18,702 real FlyWire sensory afferents (9 modalities)
- White paper: NIRON-WP-008

### Phase F: Real Sensory Neuron Routing ✅ (June 17)
- Replaced PoissonGroup with real connectome pathways
- Mapped 18,162 sensory -> 38,506 interneurons -> 953 DNs
- 5,577,954 validated 2-hop pathways discovered
- Photoreceptors: ZERO direct DN (all through lamina/medulla)
- 29mm in 5s at 0.013x real-time
- White paper: NIRON-WP-009

### Phase G: Cython Attempt ❌ (June 18)
- Brian2 Cython shows 27x speedup on benchmarks
- Dynamic g-injection incompatible with Cython pre-compilation
- Decision: abandon Cython, go C++ standalone

### Phase H: Brian2 C++ Standalone Engine ✅ (June 18)
- Working C++ binary: 138K neurons, 15M synapses, MuJoCo body
- 6.1x faster than Python (0.06x real-time vs 0.013x)
- Memory: 250MB vs 4GB Python (16x less)
- Real spike cascade verified: 17->11K->24K propagating through connectome
- Food-driven chemotaxis: 61-89mm foraging, DN ramping 1->19,930
- 20-second sustained foraging demonstrated

### Phase I: Proboscis Extension Research ✅ (June 18)
- Found 11 gustatory sensory neurons -> 666 synapses -> 213 SEZ interneurons
- Scientific gap: Proboscis MNs in SEZ (not VNC) - not in MANC dataset
- Gustatory->DN pathways: ZERO (PER is local SEZ circuit)
- Documented honestly with virtual MN mapping

### Phase J: Merge Brian2 + Real Sensory into Live Dashboard 🔄 (In Progress)
- Goal: Replace NIRON engine with Brian2 C++ standalone
- Keep same dashboard UI (MJPEG video, real-time metrics)
- Real sensory routing in live web view
- SimTome active: agent:main:subagent:eaf14102

---

## 📋 Phase Comparison

| Metric | Phase E (PoissonGroup) | Phase F (Real Sensory) | Phase H (C++ Fast) |
|--------|:---:|:---:|:---:|
| DN stimulation | PoissonGroup | Real connectome | Real connectome |
| Sensory neurons | 0 | 18,702 | 18,702 |
| Interneurons | 0 | 38,506 | 38,506 |
| Validated pathways | 0 | 5,577,954 | 5,577,954 |
| DN avg active | 30.9 | 69.3 | 112 max |
| Real-time ratio | 0.006x | 0.013x | 0.059x |
| Memory | 14GB | 4GB | 250MB |
| Language | Python/Brian2 | Python/Brian2 | Pure C++ |
| Scientific validity | Low | High | High |
| Web dashboard | No | No | No (Phase J) |

---

## 🔬 DN-MN Bridge

The Descending Neuron -> Motor Neuron Bridge translates brain commands into muscle movements.

```
FLYWIRE BRAIN (Head)              MANC VNC (Ventral Nerve Cord)        SIMFLY BODY
+-------------------+           +------------------------+         +--------------+
|  139,256 neurons  |           |  Descending Neurons    |         |  78 motors   |
|                   |  --DN-->  |       953 DNs          |  --MN-->|  103 joints  |
|  Sensory -> NIRON |           |          |             |         |  36 active   |
|  -> Burst -> DN   |           |  Motor Neurons (MN)    |         |              |
|                   |           |       737 MNs          |         |  MuJoCo      |
|  22.3M connections|           |   13.7M pathways       |         |  physics     |
+-------------------+           +------------------------+         +--------------+
```

- 953 DNs matched (71% of 1,336 total FlyWire DNs)
- 700 unique MNs targeted
- 9,530 pathways (DN->MN connections)
- 6 NT types: ACHx1.5, GABAx-0.5, GLUTx-0.25, DAx0.75, OCTx0.75, SERx-0.15
- Burst encoding: 5 spikes/5ms, charge=2.0

---

## 🐛 Key Bugs Fixed Today (June 18)
- cv2 OpenCV missing in Brian2 venv -> installed
- python3-dev headers missing on ARM64 -> extracted from libpython3.12-dev .deb
- Camera name "front" doesn't exist in MuJoCo XML -> changed to "track1"
- Renderer(width, height) swapped in server_cpp.py -> fixed
- MJPEG video feed silent crash (except catching all) -> fixed camera
- Two servers colliding on port 8080 -> clean restart
- vnc_actuator_map.json Phase I metadata breaking parser -> removed
- C++ glClearColor for dark background in Phase H
- sens_scale tuning: 1.5->50.0 (Brian2 linear vs C++ Euler integration difference)

---

## 🔜 Next Steps
1. **Phase J** — Merge Brian2 + sensory routing into live dashboard (SimTome active)
2. **NEON SIMD** — 2-3x speedup for C++ standalone
3. **Neuron pruning** — Skip quiescent neurons, 5-10x speedup
4. **C++ standalone -> real-time** — Target 1.0x real-time
5. **Multi-environment** — Food, obstacles, uneven terrain
6. **Proprioceptive feedback** — Joint angle/contact -> gait self-correction
7. **SEZ dataset** — When available, close gustatory->proboscis MN loop

---

## Infrastructure
- **Server:** Dell GB10 (192.168.1.199)
- **Brian2 venv:** /home/simllm/.../eon-fly-brain/venv/
- **NIRON venv:** /home/simllm/.../virtual-fly/venv/
- **Live dashboard:** /tmp/simfly_web/server_cpp.py -> port 8080
- **Phase H engine:** /tmp/phaseH_cpp/phaseH_main.cpp + binary
- **Display:** DISPLAY=:10, MUJOCO_GL=egl
- **GitHub:** github.com/rongsimbot/simfly
- **White papers:** NIRON-WP-008, WP-009, master SIMFLY_WHITEPAPER.md v2.0
