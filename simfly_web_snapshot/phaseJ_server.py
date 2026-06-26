#!/usr/bin/env python3
"""
Phase J: Live Web Dashboard — Brian2 C++ Standalone + Real Sensory Routing
===========================================================================
Flask web dashboard wrapping the Phase H C++ streaming binary.
- 138,639 neurons, 15.1M synapses, 18,702 sensory → 953 DNs → 108 actuators
- Real connectome sensory routing (not scripted injection)
- Live MJPEG video feed from shared-memory RGB frames
- Food chemotaxis with interactive position control
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, threading, time
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, Response, render_template, jsonify, request

# ── Config ───────────────────────────────────────────────────────────
MUJOCO_LIB = "/home/simllm/simrobotics-storage/research/flywire/eon-fly-brain/venv/lib/python3.12/site-packages/mujoco"
STREAM_BIN = "/tmp/phaseH_cpp/phaseH_stream"
FRAME_RGB = "/dev/shm/phaseH_frame.rgb"
FOOD_POS_FILE = "/tmp/simfly_web/food_pos.json"
FRAME_W, FRAME_H = 640, 480
DEFAULT_PORT = 8082

# ── Global state ──
app = Flask(__name__)
g_state: dict = {
    "step": 0, "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0,
    "dist": 0.0, "food_dist": 0.0, "speed": 0.0,
    "n_spikes": 0, "active_dns": 0, "total_dns": 0, "total_sens": 0,
    "joints": [], "paused": False,
    "food_x": 0.05, "food_y": 0.0, "food_z": 0.03,
    "running": False, "rt_ratio": 0.0
}
g_state_lock = threading.Lock()
g_latest_jpeg: bytes = b""
g_jpeg_lock = threading.Lock()
g_proc: subprocess.Popen | None = None

# ── Frame reader thread ──
def frame_reader_thread():
    """Read RGB frames from shared memory, convert to JPEG."""
    global g_latest_jpeg
    rgb_size = FRAME_W * FRAME_H * 3
    last_mtime = 0
    while True:
        try:
            mtime = os.path.getmtime(FRAME_RGB)
            if mtime != last_mtime:
                with open(FRAME_RGB, 'rb') as f:
                    rgb_data = f.read(rgb_size)
                if len(rgb_data) == rgb_size:
                    img = Image.frombytes('RGB', (FRAME_W, FRAME_H), rgb_data)
                    # Mirror vertically (OpenGL reads bottom-up)
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    buf = BytesIO()
                    img.save(buf, 'JPEG', quality=75)
                    with g_jpeg_lock:
                        g_latest_jpeg = buf.getvalue()
                    last_mtime = mtime
            time.sleep(0.01)  # Check every 10ms
        except Exception as e:
            time.sleep(0.05)

# ── Stdout reader thread ──
def stdout_reader_thread():
    """Read JSON status lines from C++ subprocess stdout."""
    global g_state, g_proc
    while g_proc and g_proc.poll() is None:
        try:
            line = g_proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('{"type":"status"'):
                data = json.loads(line)
                with g_state_lock:
                    g_state.update(data)
                    g_state["running"] = not data.get("paused", False)
        except (json.JSONDecodeError, ValueError):
            pass
        except Exception:
            break
    with g_state_lock:
        g_state["running"] = False

# ── C++ subprocess management ──
def start_cpp_stream(sim_duration=0):
    """Launch the C++ streaming binary."""
    global g_proc, g_state
    if g_proc and g_proc.poll() is None:
        return  # Already running

    env = os.environ.copy()
    env["DISPLAY"] = ":10"
    env["MUJOCO_GL"] = "egl"
    env["LD_LIBRARY_PATH"] = f"{MUJOCO_LIB}:{env.get('LD_LIBRARY_PATH', '')}"

    # Ensure food pos file exists
    os.makedirs(os.path.dirname(FOOD_POS_FILE), exist_ok=True)
    write_food_pos(g_state["food_x"], g_state["food_y"], g_state["food_z"])

    cmd = [STREAM_BIN]
    if sim_duration > 0:
        cmd.append(str(sim_duration))

    print(f"[PhaseJ] Launching: {' '.join(cmd)}", flush=True)
    g_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1
    )

    with g_state_lock:
        g_state["running"] = True

    threading.Thread(target=stdout_reader_thread, daemon=True).start()
    threading.Thread(target=frame_reader_thread, daemon=True).start()
    print("[PhaseJ] C++ stream started, threads launched", flush=True)

def stop_cpp_stream():
    """Stop the C++ streaming binary."""
    global g_proc
    if g_proc and g_proc.poll() is None:
        g_proc.send_signal(signal.SIGINT)
        try:
            g_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            g_proc.kill()
        g_proc = None

def toggle_pause():
    """Toggle pause via SIGUSR1."""
    global g_proc
    if g_proc and g_proc.poll() is None:
        g_proc.send_signal(signal.SIGUSR1)
        with g_state_lock:
            g_state["paused"] = not g_state.get("paused", False)

def write_food_pos(x, y, z):
    """Write food position for C++ to read."""
    with open(FOOD_POS_FILE, 'w') as f:
        json.dump({"x": x, "y": y, "z": z}, f)
    with g_state_lock:
        g_state["food_x"] = x
        g_state["food_y"] = y
        g_state["food_z"] = z

# ── Flask routes ──
@app.route('/')
def index():
    return render_template('index_phaseJ.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG stream from shared-memory frames."""
    def generate():
        placeholder = None
        while True:
            with g_jpeg_lock:
                jpeg = g_latest_jpeg
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                # Generate placeholder
                if placeholder is None:
                    img = Image.new('RGB', (FRAME_W, FRAME_H), (15, 15, 25))
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img)
                    draw.text((FRAME_W//2-100, FRAME_H//2-10),
                             "Loading simulation...", fill=(100, 150, 255))
                    buf = BytesIO()
                    img.save(buf, 'JPEG', quality=50)
                    placeholder = buf.getvalue()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.04)  # ~25 FPS max
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with g_state_lock:
        s = dict(g_state)
    # Add the joints as a list
    return jsonify(s)

@app.route('/api/pause', methods=['POST'])
def api_pause():
    toggle_pause()
    with g_state_lock:
        return jsonify({"paused": g_state.get("paused", False)})

@app.route('/api/food', methods=['POST'])
def api_food():
    data = request.get_json() or {}
    x = float(data.get('x', g_state.get('food_x', 0.05)))
    y = float(data.get('y', g_state.get('food_y', 0.0)))
    z = float(data.get('z', g_state.get('food_z', 0.03)))
    write_food_pos(x, y, z)
    return jsonify({"food_x": x, "food_y": y, "food_z": z})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    stop_cpp_stream()
    time.sleep(0.5)
    with g_state_lock:
        g_state.update({"step": 0, "t": 0.0, "dist": 0.0, "total_dns": 0, "total_sens": 0})
    start_cpp_stream()
    return jsonify({"status": "restarting"})

# ── Main ──
def main():
    parser = argparse.ArgumentParser(description='Phase J Live Web Dashboard')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--duration', type=float, default=0,
                       help='Simulation duration in seconds (0=infinite)')
    parser.add_argument('--no-auto', action='store_true',
                       help='Do not auto-start simulation')
    args = parser.parse_args()

    if not args.no_auto:
        start_cpp_stream(sim_duration=args.duration)

    print(f"\n[PhaseJ] Dashboard at http://192.168.1.199:{args.port}", flush=True)
    print(f"[PhaseJ]   /video_feed — Live MJPEG stream", flush=True)
    print(f"[PhaseJ]   /api/status  — Real-time metrics", flush=True)
    print(f"[PhaseJ]   /api/food    — Set food position (POST JSON)", flush=True)
    print(f"[PhaseJ]   /api/pause   — Toggle pause (POST)", flush=True)
    print(f"[PhaseJ]   /api/reset   — Restart simulation (POST)", flush=True)

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    finally:
        stop_cpp_stream()

if __name__ == '__main__':
    main()
