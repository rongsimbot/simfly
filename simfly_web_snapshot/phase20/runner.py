#!/usr/bin/env python3
"""
Phase 20 v4: CPG Walking — Connectome-Driven with CPG Rhythm
=============================================================
1. Capture baseline decoder torques from the connectome (once)
2. Apply CPG phase modulation: stance=keep, swing=reverse+reduce
3. The connectome controls torque DIRECTION. CPG controls WHICH LEGS.

This preserves the connectome's natural signal patterns 
while adding the rhythmic alternation needed for walking.
"""
import sys, time, requests

sys.stdout.reconfigure(line_buffering=True)
SERVER = "http://127.0.0.1:8080"
DT = 0.005

# Tripod groups
A_LEGS = ["T1_left","T2_right","T3_left"]
B_LEGS = ["T1_right","T2_left","T3_right"]

ALL_JOINTS = []
for leg in A_LEGS + B_LEGS:
    seg, side = leg.split("_")
    for p in ["coxa_","coxa_abduct_","coxa_twist_","femur_","femur_twist_","tibia_"]:
        ALL_JOINTS.append(p + seg + "_" + side)

A_JOINTS = [j for j in ALL_JOINTS for leg in A_LEGS 
            if j.endswith(leg.replace("_",""))]
# Simpler: filter by suffix
def joints_for(legs):
    result = []
    for j in ALL_JOINTS:
        for leg in legs:
            seg, side = leg.split("_")
            if j.endswith(seg + "_" + side):
                result.append(j)
                break
    return result

A_JOINTS = joints_for(A_LEGS)
B_JOINTS = joints_for(B_LEGS)

CYCLE_STEPS = 400  # 2s full stride
STANCE_MULT = 1.0
SWING_MULT = -0.3  # Reverse torque for swing (lift)

print("=" * 60, flush=True)
print("Phase 20 v4: Connectome + CPG Rhythm", flush=True)
print("=" * 60, flush=True)

# Verify server
try:
    r = requests.get(SERVER + "/api/status", timeout=5)
    print("Server: step=" + str(r.json()["metrics"]["step"]), flush=True)
except Exception as e:
    print("ERROR: " + str(e), flush=True)
    sys.exit(1)

requests.post(SERVER + "/api/params", json={"global_gain": 0.10}, timeout=3)

# Step 1: Clear any external torques and let decoder run raw
time.sleep(0.5)

# Step 2: Capture baseline decoder torques (connectome natural output)
print("\nCapturing baseline decoder torques...", flush=True)
baseline = {}
for i in range(20):
    try:
        r = requests.get(SERVER + "/api/torque", timeout=2)
        data = r.json().get("joints", {})
        for j, t in data.items():
            if j not in baseline:
                baseline[j] = []
            baseline[j].append(t)
    except Exception:
        pass
    time.sleep(0.01)

# Average
for j in baseline:
    baseline[j] = sum(baseline[j]) / len(baseline[j])

# Show baseline
print("Baseline torques (connectome-driven):", flush=True)
for j in ["femur_T1_left","femur_T1_right","tibia_T1_left","tibia_T1_right",
           "coxa_T1_left","coxa_T1_right"]:
    if j in baseline:
        print("  " + j + ": " + str(round(baseline[j],3)), flush=True)

print("\nStarting CPG walking loop...", flush=True)
print("Stance: " + str(STANCE_MULT) + "x  Swing: " + str(SWING_MULT) + "x", flush=True)
print("-" * 60, flush=True)

step = 0
t0 = time.time()

try:
    while True:
        phase = (step % CYCLE_STEPS) / CYCLE_STEPS
        a_stance = step % CYCLE_STEPS < CYCLE_STEPS // 2
        
        stance_joints = A_JOINTS if a_stance else B_JOINTS
        swing_joints = B_JOINTS if a_stance else A_JOINTS
        
        # Apply CPG modulation to connectome baseline
        for j in stance_joints:
            base = baseline.get(j, -0.5)
            torque = max(-1.0, min(1.0, base * STANCE_MULT))
            try:
                requests.post(SERVER + "/api/actuator/torque",
                            json={"joint": j, "torque": float(torque)}, timeout=1.0)
            except: pass
        
        for j in swing_joints:
            base = baseline.get(j, -0.5)
            torque = max(-1.0, min(1.0, base * SWING_MULT))
            try:
                requests.post(SERVER + "/api/actuator/torque",
                            json={"joint": j, "torque": float(torque)}, timeout=1.0)
            except: pass
        
        step += 1
        time.sleep(DT)
        
        if step % 100 == 0:
            e = time.time() - t0
            a = "A:ST" if a_stance else "A:SW"
            b = "B:SW" if a_stance else "B:ST"
            print("  " + str(step) + " | ph=" + str(round(phase,2)) + 
                  " | " + a + " " + b +
                  " | " + str(round(step/e,1)) + "Hz", flush=True)

except KeyboardInterrupt:
    pass
print("\nDone: " + str(step) + " steps", flush=True)
