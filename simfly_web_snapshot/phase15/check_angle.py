import json, urllib.request, time
# Check if we can access joint angles via the status API
# The qpos can be obtained from the simulation
# Let's check what data is available
r = urllib.request.urlopen("http://192.168.1.199:8080/api/status")
d = json.load(r)
# Print all top-level keys and metrics keys
print("Status keys:", list(d.keys()))
print("Metrics keys:", list(d.get("metrics", {}).keys()))
# Also check if joints endpoint exists
try:
    r2 = urllib.request.urlopen("http://192.168.1.199:8080/api/torque")
    d2 = json.load(r2)
    print("Torque response keys:", list(d2.keys()))
except:
    pass
