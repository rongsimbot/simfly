#!/usr/bin/env python3
"""
Phase 17: Batch train ALL 34 remaining leg actuator RL controllers.
Uses dynamic target-joint switching (no server restart between actuators).
Training: 100 episodes x 200 steps PPO per actuator.
"""
import subprocess, os, sys, time, signal, argparse, json, urllib.request
from datetime import datetime, timedelta

# ── Config ──
VENV_PYTHON = "/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3"
SERVER_SCRIPT = "/tmp/simfly_web/server_cpp.py"
MODEL_DIR = "/tmp/simfly_web/phase17/models"
PATHWAY_DIR = "/tmp/simfly_web/phase17/pathways"
LOG_DIR = "/tmp/simfly_web/phase17/logs"
SERVER_URL = "http://localhost:8080"
SERVER_PORT = 8080

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── All 36 leg actuators ──
LEG_JOINTS = ["coxa", "coxa_abduct", "coxa_twist", "femur", "femur_twist", "tibia"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
ALL_ACTUATORS = [f"{j}_{l}" for l in LEGS for j in LEG_JOINTS]
ALREADY_TRAINED = {"coxa_T1_left", "coxa_T1_right"}
TRAINING_LIST = [a for a in ALL_ACTUATORS if a not in ALREADY_TRAINED]

def set_target_joint(joint_name: str) -> bool:
    """Switch server target to a single actuator (no restart)."""
    import urllib.request
    data = json.dumps({"joints": [joint_name]}).encode()
    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/api/target-joints",
            data=data, headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception as e:
        print(f"    [WARN] set_target_joint failed: {e}")
        return False

def check_server() -> bool:
    try:
        req = urllib.request.urlopen(f"{SERVER_URL}/api/status", timeout=5)
        return req.status == 200
    except:
        return False

def train_one_actuator(name: str) -> dict:
    """Train single actuator using Phase 16 train_actuator.py"""
    train_script = "/tmp/simfly_web/phase16/train_actuator.py"
    log_path = os.path.join(LOG_DIR, f"train_{name}.log")
    
    cmd = [
        VENV_PYTHON, train_script,
        "--joint", name,
        "--server", SERVER_URL,
        "--episodes", "100",
        "--steps", "200",
        "--hidden", "64",
        "--lr", "0.0003",
        "--timeout", "0.05",
        "--save-dir", MODEL_DIR,
    ]
    
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, timeout=900)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    elapsed = time.time() - start
    
    metrics = {"joint": name, "returncode": rc, "elapsed": elapsed}
    
    # Parse log
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in lines:
            if "Best reward:" in line:
                metrics["best_reward"] = float(line.split(":")[-1].strip())
            if "Final avg10" in line:
                metrics["final_avg10"] = float(line.split(":")[-1].strip())
            if "Training Complete" in line:
                parts = line.split("in")
                if len(parts) > 1:
                    metrics["train_duration"] = parts[-1].strip()
    except:
        pass
    
    model_path = os.path.join(MODEL_DIR, f"{name}.pt")
    metrics["model_saved"] = os.path.exists(model_path)
    if os.path.exists(model_path):
        metrics["model_size"] = os.path.getsize(model_path)
    
    return metrics

def start_server(target: str = "coxa_T1_left"):
    """Kill existing and start fresh server."""
    subprocess.run(["pkill", "-f", "server_cpp.py"], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "server_cpp.py"], capture_output=True)
    time.sleep(2)
    
    cmd = [
        VENV_PYTHON, SERVER_SCRIPT,
        "--port", str(SERVER_PORT),
        "--neurons", "0",
        "--global-gain", "0.005",
        "--tau-decay", "50.0",
        "--target-joint", target,
    ]
    env = os.environ.copy()
    env["DISPLAY"] = ":10"
    env["MUJOCO_GL"] = "egl"
    
    log_path = os.path.join(LOG_DIR, "server_master.log")
    with open(log_path, "w") as logf:
        subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    
    # Wait for ready
    start = time.time()
    while time.time() - start < 300:
        if check_server():
            elapsed = time.time() - start
            print(f"  Server ready in {elapsed:.0f}s")
            return True
        time.sleep(5)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", type=str, default=None, help="Test single actuator then exit")
    args = parser.parse_args()
    
    if args.test:
        training = [args.test]
    else:
        training = TRAINING_LIST[args.start_from:]
    
    print(f"\n{'='*70}")
    print(f"Phase 17: Batch RL Training")
    print(f"  Total actuators: {len(ALL_ACTUATORS)}")
    print(f"  Already trained: {len(ALREADY_TRAINED)}")
    print(f"  To train: {len(training)}")
    print(f"  Models dir: {MODEL_DIR}")
    print(f"{'='*70}\n")
    
    if args.dry_run:
        for name in training:
            print(f"  [DRY RUN] {name}")
        return
    
    # Start server once
    print("Starting Brain2 server (full connectome, ~2.5 min)...")
    if not start_server("coxa_T1_left"):
        print("FATAL: Server failed to start")
        sys.exit(1)
    
    results = []
    start_time = time.time()
    
    for i, name in enumerate(training):
        eta_sec = (len(training) - i) * 420  # ~7 min per actuator avg
        eta = str(timedelta(seconds=int(eta_sec)))
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(training)}] {name} | ETA: {eta}")
        print(f"{'='*60}")
        
        # Dynamic switch
        print(f"  Switching target to {name}...")
        if not set_target_joint(name):
            print(f"  [FAIL] Cannot switch target for {name}")
            results.append({"joint": name, "success": False, "error": "target_switch"})
            continue
        
        time.sleep(2)  # let sim stabilize
        
        # Train
        print(f"  Training ({name})...")
        metrics = train_one_actuator(name)
        print(f"  Result: rc={metrics.get('returncode')}, R={metrics.get('best_reward','N/A')}, saved={metrics.get('model_saved')}")
        results.append(metrics)
        
        elapsed = time.time() - start_time
        done = len(results)
        print(f"  Progress: {done}/{len(training)} in {elapsed/60:.1f} min")
    
    # ── Summary ──
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.get("model_saved"))
    failed = len(results) - successful
    
    print(f"\n{'='*70}")
    print(f"Phase 17 Batch Training Complete")
    print(f"  Duration: {str(timedelta(seconds=int(elapsed)))}")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}")
    print(f"{'='*70}")
    
    for r in results:
        status = "OK" if r.get("model_saved") else "FAIL"
        reward = r.get("best_reward", "N/A")
        elapsed_r = r.get("elapsed", 0)
        print(f"  {status:4s} {r['joint']:30s} R={str(reward):>8s} ({elapsed_r:.0f}s)")
    
    # Save batch results
    report = {
        "phase": "phase17_batch_training",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed,
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
    with open(os.path.join(MODEL_DIR, "batch_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {MODEL_DIR}/batch_results.json")
    
    # Set server to all joints for parallel run
    print(f"\nSetting server to ALL 36 leg joints for parallel run...")
    all_names = ",".join(ALL_ACTUATORS)
    urllib.request.urlopen(
        urllib.request.Request(
            f"{SERVER_URL}/api/target-joints",
            data=json.dumps({"joints": ALL_ACTUATORS}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        ),
        timeout=5
    )
    print("Server now accepting torques for all 36 leg actuators")

if __name__ == "__main__":
    main()
