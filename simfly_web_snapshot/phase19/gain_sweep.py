#!/usr/bin/env python3
"""
Phase 19B v2: Connectome-Driven Gain Sweep
==========================================
Keep decoder architecture intact. Sweep global_gain upward.
At each level, measure body displacement and joint activity.
Find the minimum gain that produces visible movement while
preserving the connectome's natural signal patterns.
"""

import sys, os, time, json, math
import numpy as np
import requests

SERVER = "http://127.0.0.1:8080"
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

# ── Gain levels to test ──
# Current: 0.005 (no visible movement)
# Test: 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0
GAIN_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
SAMPLES_PER_GAIN = 40
INTERVAL = 0.05

print("=" * 60)
print("Phase 19B v2: Connectome Gain Sweep")
print("=" * 60)

# Verify server
try:
    r = requests.get(SERVER + "/api/status", timeout=5)
    s = r.json()
    print("Server: step=" + str(s["metrics"]["step"]) + 
          " neurons=" + str(s["neurons"]) +
          " running=" + str(s["running"]))
except Exception as e:
    print("ERROR: " + str(e))
    sys.exit(1)

# Ensure no external overrides
try:
    requests.post(SERVER + "/api/target-joints", json={"joints": None}, timeout=3)
except Exception:
    pass

results = []

for gain_idx, gain in enumerate(GAIN_LEVELS):
    print("\n" + "=" * 60)
    print("GAIN = " + str(gain) + " (" + str(gain_idx+1) + "/" + str(len(GAIN_LEVELS)) + ")")
    print("=" * 60)
    
    # Set decoder gain via API
    try:
        r = requests.post(SERVER + "/api/params",
                         json={"global_gain": gain, "tau_decay": 50.0},
                         timeout=3)
        resp = r.json()
        print("  Params set: " + str(resp))
    except Exception as e:
        print("  WARNING params: " + str(e))
    
    # Let decoder stabilize at new gain
    time.sleep(2.0)
    
    # Collect measurements
    torques = {j: [] for j in ALL_JOINTS}
    z_heights = []
    on_ground_count = 0
    fired_neurons_list = []
    dn_matches_list = []
    active_joints_list = []
    
    for i in range(SAMPLES_PER_GAIN):
        try:
            r = requests.get(SERVER + "/api/status", timeout=2)
            s = r.json()
            m = s["metrics"]
            
            z_heights.append(m.get("z_height", 0))
            if m.get("on_ground"):
                on_ground_count += 1
            fired_neurons_list.append(m.get("fired_neurons", 0))
            dn_matches_list.append(m.get("dn_matches", 0))
            active_joints_list.append(m.get("active_joints", 0))
        except Exception:
            pass
        
        try:
            r2 = requests.get(SERVER + "/api/torque", timeout=2)
            t_data = r2.json()
            joints = t_data.get("joints", {})
            for j in ALL_JOINTS:
                if j in joints:
                    torques[j].append(joints[j])
        except Exception:
            pass
        
        time.sleep(INTERVAL)
    
    # Compute stats
    mean_z = float(np.mean(z_heights)) if z_heights else 0
    std_z = float(np.std(z_heights)) if z_heights else 0
    ground_pct = on_ground_count / SAMPLES_PER_GAIN * 100
    mean_fired = float(np.mean(fired_neurons_list)) if fired_neurons_list else 0
    mean_dn = float(np.mean(dn_matches_list)) if dn_matches_list else 0
    mean_active = float(np.mean(active_joints_list)) if active_joints_list else 0
    
    # Per-joint torque stats
    joint_means = {}
    joint_activity = 0
    big_torque = 0
    for j in ALL_JOINTS:
        vals = torques.get(j, [])
        if vals:
            mj = float(np.mean(vals))
            joint_means[j] = round(mj, 4)
            if abs(mj) > 0.01:
                joint_activity += 1
            if abs(mj) > 0.1:
                big_torque += 1
    
    # Mean absolute torque across all joints
    all_abs_torques = []
    for vals in torques.values():
        all_abs_torques.extend([abs(v) for v in vals])
    mean_abs_torque = float(np.mean(all_abs_torques)) if all_abs_torques else 0
    max_abs_torque = float(np.max(all_abs_torques)) if all_abs_torques else 0
    
    result = {
        "gain": gain,
        "mean_abs_torque": round(mean_abs_torque, 4),
        "max_abs_torque": round(max_abs_torque, 4),
        "joints_active_gt_001": joint_activity,
        "joints_big_gt_01": big_torque,
        "mean_z_height": round(mean_z, 4),
        "std_z_height": round(std_z, 4),
        "ground_pct": round(ground_pct, 1),
        "mean_fired_neurons": round(mean_fired, 1),
        "mean_dn_matches": round(mean_dn, 1),
        "mean_active_joints": round(mean_active, 1),
    }
    results.append(result)
    
    # Display
    print("  Torque: mean|abs|=" + str(round(mean_abs_torque, 4)) +
          " max|abs|=" + str(round(max_abs_torque, 4)) +
          " active(>0.01)=" + str(joint_activity) + "/36" +
          " big(>0.1)=" + str(big_torque) + "/36")
    print("  Body: Z=" + str(round(mean_z, 4)) +
          " +/- " + str(round(std_z, 4)) +
          " ground=" + str(round(ground_pct)) + "%" +
          " fired=" + str(round(mean_fired)) +
          " DNs=" + str(round(mean_dn)))

# ── Summary ──
print("\n" + "=" * 60)
print("GAIN SWEEP SUMMARY")
print("=" * 60)
print("  Gain      |torque|   max|t|  active  big    Z       ground")
print("-" * 60)
for r in results:
    print("  " + str(r["gain"]).ljust(8) + " " +
          str(r["mean_abs_torque"]).rjust(8) + " " +
          str(r["max_abs_torque"]).rjust(8) + " " +
          str(r["joints_active_gt_001"]).rjust(5) + "/36 " +
          str(r["joints_big_gt_01"]).rjust(3) + "/36 " +
          str(r["mean_z_height"]).rjust(7) + " " +
          str(r["ground_pct"]).rjust(5) + "%")

# Find optimal gain: max torque while staying on ground
print("\n  Analysis:")
best_idx = 0
for i, r in enumerate(results):
    if r["ground_pct"] >= 80 and r["mean_abs_torque"] > results[best_idx]["mean_abs_torque"]:
        best_idx = i

best = results[best_idx]
print("  Recommended gain: " + str(best["gain"]) +
      " (|torque|=" + str(best["mean_abs_torque"]) +
      ", " + str(best["joints_big_gt_01"]) + " big joints, " +
      "ground=" + str(best["ground_pct"]) + "%)")

# Save
with open(OUTPUT_DIR + "/gain_sweep.json", "w") as f:
    json.dump({
        "description": "Phase 19B: Decoder gain sweep results",
        "results": results,
        "recommended_gain": best["gain"],
    }, f, indent=2)

print("\nResults: " + OUTPUT_DIR + "/gain_sweep.json")
print("Phase 19B complete.")
