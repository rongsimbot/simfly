#!/usr/bin/env python3
"""
Phase 19B: Direct Per-Joint Decoder Scale Calibration
=====================================================
1. Kill external torque override → let decoder run raw
2. Record decoder output per joint (with scales=1.0)
3. Compute optimal_scale = target_torque / decoder_output
4. Patch optimal scales into decoder
5. Test: fly moves from connectome → decoder → body (no external RL)
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

print("=" * 60)
print("Phase 19B: Decoder Scale Calibration")
print("=" * 60)

# Load target torques from 19A
with open(OUTPUT_DIR + "/torque_summary.json") as f:
    targets = json.load(f)

print("\nTarget torques loaded for " + str(len(targets)) + " joints")

# Step 1: Clear all external torques (let decoder run raw)
print("\nStep 1: Clearing external torque overrides...")
# The external torques are consumed per-step, so just wait for them to clear
# Also enable all joints
try:
    r = requests.post(SERVER + "/api/target-joints", json={"joints": None}, timeout=3)
    print("  All joints enabled: " + str(r.json().get("target_joints")))
except Exception as e:
    print("  WARNING: " + str(e))

# Wait for external torques to drain (they're consumed per-step)
time.sleep(1.0)

# Step 2: Record raw decoder outputs over N samples
SAMPLES = 100
INTERVAL = 0.05

print("\nStep 2: Recording raw decoder outputs (" + str(SAMPLES) + " samples)...")

decoder_outputs = {j: [] for j in ALL_JOINTS}

for i in range(SAMPLES):
    try:
        r = requests.get(SERVER + "/api/torque", timeout=2)
        data = r.json()
        joints = data.get("joints", {})
        for jname in ALL_JOINTS:
            if jname in joints:
                decoder_outputs[jname].append(joints[jname])
    except Exception:
        pass
    time.sleep(INTERVAL)
    
    if (i + 1) % 25 == 0:
        # Show a sample
        try:
            r2 = requests.get(SERVER + "/api/status", timeout=2)
            s = r2.json()
            print("  Sample " + str(i+1) + " | step=" + str(s["metrics"]["step"]) + 
                  " | on_ground=" + str(s["metrics"]["on_ground"]))
        except Exception:
            pass

# Step 3: Compute mean decoder output per joint
print("\nStep 3: Computing optimal per-joint scales...")
print("-" * 70)
print("  Joint                    target     decoder    scale     |target|")
print("-" * 70)

optimal_scales = {}
for jname in ALL_JOINTS:
    vals = decoder_outputs.get(jname, [])
    if not vals:
        optimal_scales[jname] = 1.0
        continue
    
    mean_decoder = float(np.mean(vals))
    
    # Get target
    target_data = targets.get(jname, {})
    target_mean = target_data.get("mean", 0.0)
    
    # Compute optimal scale
    if abs(mean_decoder) > 0.0001:
        scale = target_mean / mean_decoder
    else:
        scale = 1.0
    
    # Clamp to reasonable range
    scale = max(-50.0, min(50.0, scale))
    
    optimal_scales[jname] = round(scale, 4)
    
    # Display
    t_str = ("+" if target_mean >= 0 else "") + str(round(target_mean, 4))
    d_str = ("+" if mean_decoder >= 0 else "") + str(round(mean_decoder, 4))
    s_str = ("+" if scale >= 0 else "") + str(round(scale, 2))
    bar = "█" * min(50, int(abs(target_mean) * 50))
    
    print("  " + jname.ljust(23) + " " + t_str.rjust(10) + " " + 
          d_str.rjust(10) + " " + s_str.rjust(8) + "  " + bar)

# Step 4: Save optimal scales
scales_path = OUTPUT_DIR + "/optimal_scales.json"
with open(scales_path, "w") as f:
    json.dump({
        "description": "Phase 19B: Optimal per-joint decoder scales",
        "per_joint_scales": optimal_scales,
        "recommended_global_gain": 0.005,  # keep current
    }, f, indent=2)

print("\nStep 4: Optimal scales saved to " + scales_path)

# Step 5: Show stats
n_scales = len(optimal_scales)
mean_scale = float(np.mean(list(optimal_scales.values())))
max_scale = max(optimal_scales.values(), key=abs)
print("\n  " + str(n_scales) + " joints calibrated")
print("  Mean scale: " + str(round(mean_scale, 2)))
print("  Max scale: " + str(round(max_scale, 2)))
print("\nPhase 19B complete.")
