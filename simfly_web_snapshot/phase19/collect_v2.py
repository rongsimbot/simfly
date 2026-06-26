#!/usr/bin/env python3
"""
Phase 19A v2: MN→Torque Data Collection + Inference Controller (CPU)
====================================================================
- Loads all 36 models to CPU (avoids CUDA OOM)
- Collects (MN_state, joint_state, target_torque) dataset
- Simultaneously applies torques to keep the simulation running
"""

import sys, os, time, json
from collections import defaultdict
import numpy as np
import torch
import requests

sys.path.insert(0, "/tmp/simfly_web/phase16")
from per_actuator_rl import PerActuatorRL

SERVER = "http://127.0.0.1:8080"
MODEL_DIR = "/tmp/simfly_web/phase17/models"
OUTPUT_DIR = "/tmp/simfly_web/phase19"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_JOINTS = [
    "coxa_T1_left", "coxa_abduct_T1_left", "coxa_twist_T1_left",
    "femur_T1_left", "femur_twist_T1_left", "tibia_T1_left",
    "coxa_T1_right", "coxa_abduct_T1_right", "coxa_twist_T1_right",
    "femur_T1_right", "femur_twist_T1_right", "tibia_T1_right",
    "coxa_T2_left", "coxa_abduct_T2_left", "coxa_twist_T2_left",
    "femur_T2_left", "femur_twist_T2_left", "tibia_T2_left",
    "coxa_T2_right", "coxa_abduct_T2_right", "coxa_twist_T2_right",
    "femur_T2_right", "femur_twist_T2_right", "tibia_T2_right",
    "coxa_T3_left", "coxa_abduct_T3_left", "coxa_twist_T3_left",
    "femur_T3_left", "femur_twist_T3_left", "tibia_T3_left",
    "coxa_T3_right", "coxa_abduct_T3_right", "coxa_twist_T3_right",
    "femur_T3_right", "femur_twist_T3_right", "tibia_T3_right",
]

print("=" * 60)
print("Phase 19A v2: MN-to-Torque Dataset Collection (CPU)")
print("=" * 60)

# Load all 36 models to CPU
print("\nLoading models (CPU)...")
agents = {}
for joint in ALL_JOINTS:
    path = MODEL_DIR + "/" + joint + ".pt"
    if not os.path.exists(path):
        print("  SKIP " + joint)
        continue
    try:
        # Force CPU load
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        agent = PerActuatorRL(
            joint_name=cfg["joint_name"],
            server_url=SERVER,
            hidden_dim=cfg.get("hidden_dim", 64),
            joint_angle_limit=cfg.get("joint_angle_limit", 0.8),
            angular_displacement_weight=cfg.get("angular_displacement_weight", 10.0),
        )
        agent.policy.load_state_dict(data["policy_state_dict"])
        agent.policy = agent.policy.to("cpu")  # Ensure CPU
        agents[joint] = agent
    except Exception as e:
        print("  FAIL " + joint + ": " + str(e))

print("  Loaded " + str(len(agents)) + "/36 models to CPU")
torch.set_num_threads(4)  # Limit CPU threads

# Verify server
try:
    r = requests.get(SERVER + "/api/status", timeout=5)
    s = r.json()
    print("  Server OK: step=" + str(s["metrics"]["step"]))
except Exception as e:
    print("ERROR: " + str(e))
    sys.exit(1)

# Enable all joints
try:
    requests.post(SERVER + "/api/target-joints", json={"joints": None}, timeout=3)
except Exception:
    pass

SAMPLES = 500
INTERVAL = 0.05  # match physics rate for torque application

print("\nCollecting " + str(SAMPLES) + " samples while applying torques...")
print("-" * 60)

dataset = []
step_count = 0
start_time = time.time()

for i in range(SAMPLES):
    sample = {
        "step": i,
        "timestamp": time.time(),
        "joints": {},
    }
    
    # Get global status
    try:
        r = requests.get(SERVER + "/api/status", timeout=2)
        status = r.json()
        m = status["metrics"]
        sample["global"] = {
            "fired_neurons": m.get("fired_neurons", 0),
            "dn_matches": m.get("dn_matches", 0),
            "mns_activated": m.get("mns_activated", 0),
            "z_height": round(m.get("z_height", 0), 4),
            "on_ground": m.get("on_ground", False),
        }
    except Exception:
        sample["global"] = {}
    
    # Per joint: get state, compute target torque, apply it
    nz = 0
    for joint_name, agent in agents.items():
        try:
            # Get actuator state
            r2 = requests.get(SERVER + "/api/actuator/state/" + joint_name, timeout=2)
            if r2.status_code == 200:
                act = r2.json()
                state = np.array([
                    float(act.get("dn_fired", 0)) / max(act.get("dn_total", 1), 1),
                    float(act.get("dn_fired", 0)),
                    float(act.get("mn_active", 0)) / max(agent.mn_count, 1),
                    min(float(act.get("mn_mean_activation", 0)), 10.0),
                    np.clip(float(act.get("joint_angle", 0)), -np.pi, np.pi),
                    np.clip(float(act.get("joint_velocity", 0)), -20.0, 20.0),
                ], dtype=np.float32)
            else:
                state = np.zeros(6, dtype=np.float32)
                act = {}
            
            # Compute target torque (CPU inference)
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = agent.policy(state_t)
            target_torque = float(mean.item())
            
            # Apply torque to server
            try:
                requests.post(
                    SERVER + "/api/actuator/torque",
                    json={"joint": joint_name, "torque": float(np.clip(target_torque, -1.0, 1.0))},
                    timeout=1.0
                )
            except Exception:
                pass
            
            if abs(target_torque) > 0.001:
                nz += 1
            
            sample["joints"][joint_name] = {
                "dn_fired": act.get("dn_fired", 0),
                "dn_total": act.get("dn_total", 0),
                "mn_active": act.get("mn_active", 0),
                "mn_mean_activation": round(act.get("mn_mean_activation", 0), 4),
                "angle": round(act.get("joint_angle", 0), 4),
                "velocity": round(act.get("joint_velocity", 0), 4),
                "target_torque": round(target_torque, 6),
            }
        except Exception as e:
            sample["joints"][joint_name] = {
                "target_torque": 0.0,
                "error": str(e)[:100],
            }
    
    sample["active_joints"] = nz
    dataset.append(sample)
    step_count += 1
    time.sleep(INTERVAL)
    
    if (i + 1) % 50 == 0:
        mean_abs = 0
        count = 0
        for jd in sample["joints"].values():
            mean_abs += abs(jd.get("target_torque", 0))
            count += 1
        elapsed = time.time() - start_time
        print("  Sample " + str(i+1) + "/" + str(SAMPLES) +
              " | active: " + str(nz) + "/36" +
              " | mean|t|: " + str(round(mean_abs/max(count,1), 4)) +
              " | " + str(round((i+1)/elapsed, 1)) + " s/s" +
              " | on_ground: " + str(sample["global"].get("on_ground")))

# Save dataset
output_path = OUTPUT_DIR + "/torque_dataset.json"
with open(output_path, "w") as f:
    json.dump({
        "description": "Phase 19A: MN activation -> target torque from 36 RL models (CPU inference)",
        "num_samples": len(dataset),
        "joints": ALL_JOINTS,
        "samples": dataset,
    }, f)

print("\n" + "=" * 60)
print("Dataset: " + output_path + " (" + str(len(dataset)) + " samples)")
print("Time: " + str(round(time.time() - start_time, 1)) + "s")
print("=" * 60)

# Summary stats
print("\n=== Per-Joint Target Torque Stats ===")
joint_stats = defaultdict(list)
for sample in dataset:
    for jname, jdata in sample["joints"].items():
        joint_stats[jname].append(jdata.get("target_torque", 0))

summary = {}
for jname in sorted(joint_stats.keys()):
    vals = joint_stats[jname]
    m = float(np.mean(vals))
    s = float(np.std(vals))
    ma = float(max(abs(v) for v in vals))
    summary[jname] = {"mean": round(m,4), "std": round(s,4), "max_abs": round(ma,4)}
    sign = "+" if m > 0 else ""
    print("  " + jname + ": mean=" + sign + str(round(m,4)) +
          " std=" + str(round(s,4)) + " max|=" + str(round(ma,4)))

with open(OUTPUT_DIR + "/torque_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPhase 19A complete.")
