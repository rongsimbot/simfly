import json, urllib.request, time, os

os.makedirs("/tmp/simfly_web/phase15", exist_ok=True)

torque_series = []
status_series = []

print("Starting 2-min monitoring of coxa_T1_left...")
start = time.time()
while time.time() - start < 120:
    try:
        r = urllib.request.urlopen("http://192.168.1.199:8080/api/torque", timeout=5)
        d = json.load(r)
        step = d.get("step", 0)
        coxa = d.get("joints", {}).get("coxa_T1_left", 0)
        active = d.get("active_joints", 0)
        torque_series.append({"t": round(time.time() - start, 1), "step": step, "coxa_t1_left_torque": coxa, "active_joints": active})
        
        r2 = urllib.request.urlopen("http://192.168.1.199:8080/api/status", timeout=5)
        s = json.load(r2)
        status_series.append({"t": round(time.time() - start, 1), "step": s["metrics"]["step"], "z_height": s["metrics"]["z_height"], "fired": s["metrics"]["fired_neurons"]})
        
        if len(torque_series) % 6 == 1:
            print(f"  t={torque_series[-1]['t']:.0f}s step={step} torque={coxa:.4f} z={s['metrics']['z_height']:.4f}")
    except Exception as e:
        print(f"  Error at t={time.time()-start:.0f}s: {e}")
    
    time.sleep(10)

# Save time series
with open("/tmp/simfly_web/phase15/torque_timeseries.json", "w") as f:
    json.dump(torque_series, f, indent=2)

with open("/tmp/simfly_web/phase15/status_timeseries.json", "w") as f:
    json.dump(status_series, f, indent=2)

print(f"\nMonitoring complete! {len(torque_series)} samples captured")
print(f"Final torque: {torque_series[-1]['coxa_t1_left_torque']:.4f}")
print(f"Torque range: {min(t['coxa_t1_left_torque'] for t in torque_series):.4f} to {max(t['coxa_t1_left_torque'] for t in torque_series):.4f}")
print(f"z_height range: {min(s['z_height'] for s in status_series):.4f} to {max(s['z_height'] for s in status_series):.4f}")
