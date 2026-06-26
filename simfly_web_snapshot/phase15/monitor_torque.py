import json, urllib.request, time

r = urllib.request.urlopen("http://192.168.1.199:8080/api/torque")
d = json.load(r)
print(f"Step: {d.get('step', '?')}")
print(f"Active joints: {d.get('active_joints', '?')}")
coxa = d.get("joints", {}).get("coxa_T1_left", "N/A")
print(f"coxa_T1_left decoder torque: {coxa}")
nonzero = {k: v for k, v in d.get("joints", {}).items() if abs(v) > 0.01}
print(f"Significant torques (>0.01): {nonzero}")
