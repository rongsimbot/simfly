#!/usr/bin/env python3
"""
Phase 19A: Collect MN Activation → Target Torque Pairs
======================================================
Runs alongside the live simulation. For each step:
1. Reads connectome MN activation state from server
2. Runs each of the 36 trained RL models to get target torque
3. Records (MN_state, joint_state, target_torque) → dataset

Output: /tmp/simfly_web/phase19/torque_dataset.json
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
print("Phase 19A: Target Torque Collection")
print("=" * 60)

# Load all 36 models
print("\nLoading models...")
agents = {}
for joint in ALL_JOINTS:
    path = MODEL_DIR + "/" + joint + ".pt"
    if not os.path.exists(path):
        print("  SKIP " + joint + ": no model")
        continue
    try:
        agent = PerActuatorRL.load(path, server_url=SERVER)
        agents[joint] = agent
    except Exception as e:
        print("  FAIL " + joint + ": " + str(e))

print("  Loaded " + str(len(agents)) + "/36 models")

if len(agents) < 30:
    print("ERROR: Too few models loaded. Exiting.")
    sys.exit(1)

# Verify server
try:
    r = requests.get(SERVER + "/api/status", timeout=5)
    s = r.json()
    print("  Server OK: step=" + str(s["metrics"]["step"]))
except Exception as e:
    print("ERROR: " + str(e))
    sys.exit(1)

# Collection loop
SAMPLES = 500
INTERVAL = 0.2  # 5 Hz sampling

dataset = []
print("\nCollecting " + str(SAMPLES) + " samples at 5 Hz...")
print("-" * 60)

for i in range(SAMPLES):
    sample = {
        "step": i,
        "timestamp": time.time(),
        "joints": {},
    }
    
    # Get global status for context
    try:
        r = requests.get(SERVER + "/api/status", timeout=2)
        status = r.json()
        m = status["metrics"]
        sample["global"] = {
            "fired_neurons": m.get("fired_neurons", 0),
            "dn_matches": m.get("dn_matches", 0),
            "mns_activated": m.get("mns_activated", 0),
            "z_height": m.get("z_height", 0),
            "on_ground": m.get("on_ground", False),
        }
    except Exception:
        sample["global"] = {}
    
    # For each joint: get MN state → RL target torque
    for joint_name, agent in agents.items():
        try:
            state = agent.get_state()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.policy.fc1.weight.device)
                mean, _, _ = agent.policy(state_t)
            target_torque = float(mean.item())
            
            # Get raw actuator state
            r2 = requests.get(SERVER + "/api/actuator/state/" + joint_name, timeout=2)
            if r2.status_code == 200:
                act = r2.json()
                joint_state = {
                    "dn_fired": act.get("dn_fired", 0),
                    "dn_total": act.get("dn_total", 0),
                    "mn_active": act.get("mn_active", 0),
                    "mn_mean_activation": act.get("mn_mean_activation", 0),
                    "angle": act.get("joint_angle", 0),
                    "velocity": act.get("joint_velocity", 0),
                }
            else:
                joint_state = {"dn_fired": 0, "dn_total": 0, "mn_active": 0, 
                               "mn_mean_activation": 0, "angle": 0, "velocity": 0}
            
            sample["joints"][joint_name] = {
                "state": joint_state,
                "target_torque": target_torque,
            }
        except Exception as e:
            sample["joints"][joint_name] = {
                "state": {"dn_fired": 0, "mn_active": 0, "angle": 0, "velocity": 0},
                "target_torque": 0.0,
                "error": str(e),
            }
    
    dataset.append(sample)
    
    if (i + 1) % 50 == 0:
        # Quick stats
        mean_abs_torque = 0
        count = 0
        for jd in sample["joints"].values():
            mean_abs_torque += abs(jd["target_torque"])
            count += 1
        mean_abs_torque /= max(count, 1)
        print("  Sample " + str(i+1) + "/" + str(SAMPLES) + 
              " | mean|target|: " + str(round(mean_abs_torque, 4)) +
              " | on_ground: " + str(sample["global"].get("on_ground")))
    
    time.sleep(INTERVAL)

# Save dataset
output_path = OUTPUT_DIR + "/torque_dataset.json"
with open(output_path, "w") as f:
    json.dump({
        "description": "Phase 19A: MN activation → target torque pairs from 36 trained RL models",
        "num_samples": len(dataset),
        "joints": ALL_JOINTS,
        "samples": dataset,
    }, f, indent=2)

print("\n" + "=" * 60)
print("Dataset saved: " + output_path)
print("Samples: " + str(len(dataset)))
print("=" * 60)

# Compute summary stats
print("\n=== Per-Joint Target Torque Stats ===")
joint_stats = defaultdict(list)
for sample in dataset:
    for jname, jdata in sample["joints"].items():
        joint_stats[jname].append(jdata["target_torque"])

summary = {}
for jname in sorted(joint_stats.keys()):
    vals = joint_stats[jname]
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    max_abs = float(max(abs(v) for v in vals))
    summary[jname] = {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "max_abs": round(max_abs, 4),
        "n_samples": len(vals),
    }
    sign = "+" if mean_val > 0 else ""
    print("  " + jname + ": mean=" + sign + str(round(mean_val, 4)) + 
          " std=" + str(round(std_val, 4)) + 
          " max|abs|=" + str(round(max_abs, 4)))

with open(OUTPUT_DIR + "/torque_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved: " + OUTPUT_DIR + "/torque_summary.json")
print("\nPhase 19A complete.")
