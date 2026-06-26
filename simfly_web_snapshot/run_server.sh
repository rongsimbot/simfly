#!/bin/bash
export DISPLAY=:10
export MUJOCO_GL=egl
export PYTHONPATH=/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model:/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/sensory:/home/simllm/simrobotics-storage/research/flywire
cd /home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model
exec /home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3 /tmp/simfly_web/server.py --neurons 200 --port 8080
