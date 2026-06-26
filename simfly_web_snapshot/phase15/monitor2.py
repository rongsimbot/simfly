import json, urllib.request, time

for i in range(6):
    r = urllib.request.urlopen("http://192.168.1.199:8080/api/torque")
    d = json.load(r)
    coxa = d.get("joints", {}).get("coxa_T1_left", 0)
    step = d.get("step", 0)
    print(f"  Step {step}: coxa_T1_left torque = {coxa:.6f}")
    if i < 5:
        time.sleep(5)
