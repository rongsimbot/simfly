import re, subprocess
coxa_mn_ids = {17664, 163719, 157320, 10892, 19468, 12686}
mn_pattern = "|".join(f'"mn_id": {x}\\b' for x in sorted(coxa_mn_ids))
result = subprocess.run(
    ["grep", "-P", mn_pattern, 
     "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/vnc_bridge/dn_mn_pathways.json"],
    capture_output=True, text=True, timeout=120
)
lines = result.stdout.strip().split("\n")
print(f"Found {len(lines)} lines")
for i, line in enumerate(lines[:3]):
    print(f"--- Line {i} ---")
    print(line[:500])
    print()
