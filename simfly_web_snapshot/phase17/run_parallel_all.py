#!/usr/bin/env python3
"""Phase 17: Run all 36 RL controllers in parallel from same connectome state."""
import sys, os, argparse, time, json, requests
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/tmp/simfly_web/phase16")
from per_actuator_rl import PerActuatorRL, MultiActuatorController

LEG_JOINTS = ["coxa", "coxa_abduct", "coxa_twist", "femur", "femur_twist", "tibia"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
ALL_ACTUATORS = [f"{j}_{l}" for l in LEGS for j in LEG_JOINTS]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--models-dir", default="/tmp/simfly_web/phase17/models")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--report", default="/tmp/simfly_web/phase17/phase17_report.json")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 17: Full Parallel Run - ALL 36 Leg Actuators")
    print(f"  Duration: {args.duration}s @ {args.interval}s")
    print(f"{'='*70}\n")
    
    controller = MultiActuatorController(server_url=args.server)
    
    loaded = 0
    missing = []
    for name in ALL_ACTUATORS:
        model_path = os.path.join(args.models_dir, f"{name}.pt")
        pathway = f"/tmp/simfly_web/phase17/pathways/{name}.json"
        if not os.path.exists(pathway):
            pathway = None
        if os.path.exists(model_path):
            try:
                agent = PerActuatorRL.load(model_path, server_url=args.server, pathway_json=pathway)
                controller.add_agent(agent)
                loaded += 1
                print(f"  [OK] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                missing.append(name)
        else:
            missing.append(name)
            print(f"  [MISS] {name}")
    
    print(f"\n  Loaded: {loaded}/{len(ALL_ACTUATORS)}")
    if missing:
        print(f"  Missing: {len(missing)}")
    
    if loaded == 0:
        print("  ABORT: No models loaded")
        return
    
    # Set server to all joints
    try:
        requests.post(f"{args.server}/api/target-joints",
                     json={"joints": ALL_ACTUATORS}, timeout=5)
        print("  Server set to all 36 leg joints")
    except Exception as e:
        print(f"  [WARN] Could not set target joints: {e}")
    
    print(f"\n{'='*70}")
    print(f"Starting {args.duration}s parallel run...")
    print(f"{'='*70}\n")
    
    start = time.perf_counter()
    step_count = 0
    torques_history = []
    
    try:
        while time.perf_counter() - start < args.duration:
            torques = controller.step()
            step_count += 1
            time.sleep(args.interval)
            
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - start
                sample = {k: v for i, (k, v) in enumerate(sorted(torques.items())) if i < 6}
                tstr = ", ".join(f"{j.split('_')[0][:5]}={t:+.3f}" for j, t in sample.items())
                print(f"  [{elapsed:6.1f}s] step={step_count:6d} | {tstr}...")
            
            torques_history.append(torques)
    except KeyboardInterrupt:
        print("\n  Interrupted")
    
    elapsed = time.perf_counter() - start
    
    joint_stats = defaultdict(list)
    for tq in torques_history:
        for j, v in tq.items():
            joint_stats[j].append(v)
    
    print(f"\n{'='*70}")
    print(f"Complete: {elapsed:.1f}s, {step_count} steps, {step_count/elapsed:.1f} Hz")
    
    report = {
        "phase": "phase17_full_parallel",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed,
        "steps": step_count,
        "rate_hz": step_count/elapsed if elapsed > 0 else 0,
        "actuators_loaded": loaded,
        "actuators_total": len(ALL_ACTUATORS),
        "actuators_missing": missing,
        "per_joint_stats": {
            j: {
                "mean": sum(v)/len(v),
                "min": min(v),
                "max": max(v),
                "std": (sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5
            }
            for j, v in joint_stats.items()
        } if joint_stats else {},
    }
    
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {args.report}")
    
    # Per-joint detail
    for jname in sorted(joint_stats.keys()):
        vals = joint_stats[jname]
        mean = sum(vals)/len(vals)
        std = report["per_joint_stats"][jname]["std"]
        print(f"  {jname:30s}: mean={mean:+7.4f}  std={std:.4f}  range=[{min(vals):+.4f}, {max(vals):+.4f}]")

if __name__ == "__main__":
    main()
