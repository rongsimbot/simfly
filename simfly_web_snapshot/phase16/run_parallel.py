#!/usr/bin/env python3
"""
Phase 16: Run multiple RL controllers in parallel from same connectome state.
Usage:
  python3 run_parallel.py --joints coxa_T1_left,coxa_T1_right --duration 60
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_actuator_rl import PerActuatorRL, MultiActuatorController, DEVICE

def main():
    parser = argparse.ArgumentParser(description="Run parallel per-actuator RL controllers")
    parser.add_argument("--joints", type=str, required=True,
                        help="Comma-separated joint names")
    parser.add_argument("--server", type=str, default="http://localhost:8080")
    parser.add_argument("--models-dir", type=str, default="/tmp/simfly_web/phase16/models")
    parser.add_argument("--duration", type=int, default=60,
                        help="Run duration in seconds")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="Control interval in seconds")
    args = parser.parse_args()

    joint_list = [j.strip() for j in args.joints.split(",")]
    print(f"\nPhase 16: Parallel Control — {len(joint_list)} actuators")
    print(f"  Joints: {joint_list}")
    print(f"  Server: {args.server}")
    print(f"  Duration: {args.duration}s @ {args.interval}s interval")

    controller = MultiActuatorController(server_url=args.server)

    for jname in joint_list:
        model_path = os.path.join(args.models_dir, f"{jname}.pt")
        pathway = f"/tmp/simfly_web/actuator_{jname}_pathway.json"
        if not os.path.exists(pathway):
            pathway = None

        if os.path.exists(model_path):
            agent = PerActuatorRL.load(model_path, server_url=args.server, pathway_json=pathway)
            controller.add_agent(agent)
            print(f"  ✅ Loaded {jname} from {model_path}")
        else:
            print(f"  ❌ Model not found: {model_path}")
            print(f"     Run: python3 train_actuator.py --joint {jname}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Starting parallel control loop ({args.duration}s)...")
    print(f"All {len(joint_list)} agents read SAME connectome state")
    print(f"Each produces DIFFERENT torque for its actuator")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    step_count = 0
    torques_history = []

    try:
        while time.perf_counter() - start < args.duration:
            torques = controller.step()
            step_count += 1
            time.sleep(args.interval)

            if step_count % 50 == 0:
                elapsed = time.perf_counter() - start
                tstr = ", ".join(f"{j}={t:+.3f}" for j, t in sorted(torques.items()))
                print(f"  [{elapsed:5.1f}s] step={step_count} | torques: {tstr}")
            torques_history.append(torques)

    except KeyboardInterrupt:
        print("\n  Interrupted")

    elapsed = time.perf_counter() - start
    print(f"\n{'='*60}")
    print(f"Parallel Control Complete")
    print(f"  Duration: {elapsed:.1f}s | Steps: {step_count}")
    print(f"  Rate: {step_count/elapsed:.1f} Hz")
    if torques_history:
        # Summary stats per joint
        from collections import defaultdict
        joint_stats = defaultdict(list)
        for tq in torques_history:
            for j, v in tq.items():
                joint_stats[j].append(v)
        for jname in sorted(joint_stats.keys()):
            vals = joint_stats[jname]
            print(f"  {jname}: mean={sum(vals)/len(vals):+.4f}, min={min(vals):+.4f}, max={max(vals):+.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
