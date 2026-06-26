import json, urllib.request, time, os

os.makedirs("/tmp/simfly_web/phase15", exist_ok=True)

# 1. Capture status
r = urllib.request.urlopen("http://192.168.1.199:8080/api/status")
status = json.load(r)

# 2. Capture torque data  
r2 = urllib.request.urlopen("http://192.168.1.199:8080/api/torque")
torque_data = json.load(r2)

# 3. Build evidence
evidence = {
    "phase": "Phase 15: Per-Actuator Connectome Control",
    "target_actuator": "coxa_T1_left",
    "target_actuator_index": 2,
    "actuator_qpos_addr": 45,
    "actuator_joint_range": [-0.2, 1.7],
    "status_metrics": status.get("metrics", {}),
    "per_joint_scale": 50.0,
    "decoder_torque_coxa_T1_left": torque_data.get("joints", {}).get("coxa_T1_left", 0),
    "all_decoder_torques": torque_data.get("joints", {}),
    "active_joints": status.get("metrics", {}).get("active_joints", 1),
    "torque_applied": status.get("metrics", {}).get("torque_applied", False),
}

with open("/tmp/simfly_web/phase15/evidence_snapshot.json", "w") as f:
    json.dump(evidence, f, indent=2)

print("Evidence saved!")
print(f"Step: {evidence['status_metrics']['step']}")
print(f"Active joints: {evidence['active_joints']}")
print(f"Torque applied: {evidence['torque_applied']}")
print(f"coxa_T1_left torque: {evidence['decoder_torque_coxa_T1_left']}")
print(f"z_height: {evidence['status_metrics']['z_height']:.4f}")
print(f"food_distance: {evidence['status_metrics']['food_distance']:.4f}")
