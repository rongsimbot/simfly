#!/usr/bin/env python3
"""Phase 18: Multi-Actuator Inference Controller - Fixed"""
import sys, os, time, json
import numpy as np
import torch
import requests

sys.path.insert(0, "/tmp/simfly_web/phase16")
from per_actuator_rl import PerActuatorRL, MultiActuatorController

SERVER = "http://127.0.0.1:8080"
MODEL_DIR = "/tmp/simfly_web/phase17/models"
COLLECT_TIMEOUT = 0.05

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

def main():
    print("=" * 60)
    print("Phase 18: Multi-Actuator Inference Controller")
    print("=" * 60)
    
    # Enable all joints
    try:
        r = requests.post(SERVER + "/api/target-joints",
                         json={"joints": None}, timeout=3)
        resp = r.json()
        print("  Target joints: " + str(resp.get("target_joints", resp)))
    except Exception as e:
        print("  WARNING target-joints: " + str(e))
    
    # Load all 36 models
    controller = MultiActuatorController(server_url=SERVER)
    loaded = 0
    failed = []
    
    for joint in ALL_JOINTS:
        model_path = MODEL_DIR + "/" + joint + ".pt"
        if not os.path.exists(model_path):
            failed.append(joint)
            continue
        try:
            agent = PerActuatorRL.load(model_path, server_url=SERVER)
            controller.add_agent(agent)
            loaded += 1
        except Exception as e:
            print("  FAIL " + joint + ": " + str(e))
            failed.append(joint)
    
    print("\n  Loaded: " + str(loaded) + "/36 models")
    if failed:
        print("  Failed: " + str(failed))
    if loaded == 0:
        print("ERROR: No models loaded.")
        return
    
    # Verify server connection
    try:
        r = requests.get(SERVER + "/api/status", timeout=5)
        s = r.json()
        m = s.get("metrics", {})
        print("  Server OK: step=" + str(m.get("step", "?")) + 
              ", neurons=" + str(s.get("neurons", "?")) + 
              ", running=" + str(s.get("running", "?")))
    except Exception as e:
        print("ERROR: Cannot reach server: " + str(e))
        return
    
    # Main inference loop
    print("\n  Starting inference loop (Ctrl+C to stop)...")
    print("  Collect timeout: " + str(COLLECT_TIMEOUT) + "s")
    print("-" * 60)
    
    step_count = 0
    total_torque = 0.0
    start_time = time.time()
    
    try:
        while True:
            torques = controller.step()
            total_torque += sum(abs(t) for t in torques.values())
            step_count += 1
            time.sleep(COLLECT_TIMEOUT)
            
            if step_count % 200 == 0:
                elapsed = time.time() - start_time
                nz = sum(1 for t in torques.values() if abs(t) > 0.001)
                mean_t = total_torque / max(step_count, 1)
                print("  Step " + str(step_count) + " | "
                      "active: " + str(nz) + "/36 | "
                      "mean|torque|: " + str(round(mean_t, 4)) + " | "
                      "rate: " + str(round(step_count/elapsed, 1)) + " s/s | "
                      "elapsed: " + str(round(elapsed)) + "s")
    
    except KeyboardInterrupt:
        pass
    
    print("\n  Stopped after " + str(step_count) + " steps, " + str(round(time.time()-start_time, 1)) + "s")

if __name__ == "__main__":
    main()
