import re

with open("/tmp/simfly_web/server_cpp.py", "r") as f:
    content = f.read()

# 1. Add target_joints to __init__ right after self.metrics
old_init = """        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)"""
new_init = """        self.step_count = 0
        self.sim_time_ms = 0.0
        self.metrics = defaultdict(list)
        # ── Phase 15: Per-Actuator Connectome Control ──
        # Set to a set of joint names to isolate single-actuator torque.
        # None (default) = normal operation (all actuators).
        self.target_joints: Optional[Set[str]] = None"""

if old_init not in content:
    print("ERROR: Could not find old_init pattern")
    print("Searching for step_count...")
    for i, line in enumerate(content.split("\n"), 1):
        if "step_count" in line:
            print(f"  Line {i}: {line}")
    exit(1)

content = content.replace(old_init, new_init)

# 2. Modify the torque application section to filter by target_joints
old_torque = """        self.data.ctrl[:] = 0.0
        nz = 0
        for jname, torque in joint_commands.items():
            if jname in self.act_map:
                self.data.ctrl[self.act_map[jname]] = float(np.clip(torque, -1.0, 1.0))
                if abs(torque) > 0.001:
                    nz += 1"""

new_torque = """        self.data.ctrl[:] = 0.0
        nz = 0
        target_set = self.target_joints  # Phase 15: per-actuator isolation
        for jname, torque in joint_commands.items():
            if target_set is not None and jname not in target_set:
                continue  # 🔴 Phase 15: SKIP non-target actuators
            if jname in self.act_map:
                self.data.ctrl[self.act_map[jname]] = float(np.clip(torque, -1.0, 1.0))
                if abs(torque) > 0.001:
                    nz += 1"""

if old_torque not in content:
    print("ERROR: Could not find old_torque pattern")
    exit(1)

content = content.replace(old_torque, new_torque)

with open("/tmp/simfly_web/server_cpp.py", "w") as f:
    f.write(content)

print("Patch applied successfully!")
print(f"Content size: {len(content)} bytes")
