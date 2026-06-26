#!/usr/bin/env python3
"""Add dynamic target-joints API endpoint to server_cpp.py"""
import shutil

server_path = "/tmp/simfly_web/server_cpp.py"
backup_path = "/tmp/simfly_web/server_cpp.py.bak_phase17_20260623"

# Backup
shutil.copy(server_path, backup_path)
print(f"Backup: {backup_path}")

with open(server_path) as f:
    content = f.read()

new_endpoint = '''
@app.route("/api/target-joints", methods=["GET", "POST"])
def api_target_joints():
    """Phase 17: Get/set target joints dynamically (no server restart needed)."""
    if not sim_server._initialized or not sim_server.loop:
        return jsonify({"error": "simulation not initialized"}), 503
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
            joints = data.get("joints", [])
            if isinstance(joints, str):
                joints = [j.strip() for j in joints.split(",") if j.strip()]
            sim_server.loop.target_joints = set(joints) if joints else None
            sim_server.loop._external_torques.clear()
            return jsonify({"target_joints": list(sim_server.loop.target_joints) if sim_server.loop.target_joints else None})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        tj = sim_server.loop.target_joints
        return jsonify({"target_joints": list(tj) if tj else None})
'''

insert_before = "# ── Phase 16: Per-Actuator RL API Endpoints ──"
if insert_before in content:
    content = content.replace(insert_before, new_endpoint + "\n" + insert_before)
    with open(server_path, "w") as f:
        f.write(content)
    print("Endpoint /api/target-joints ADDED to server_cpp.py")
else:
    print("ERROR: Insert marker not found")
    print("Looking for:", insert_before)
