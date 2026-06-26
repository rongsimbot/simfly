import json, urllib.request, time, sys
sys.path.insert(0, "/home/simllm/simrobotics-storage/research/flywire")
sys.path.insert(0, "/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/lib/python3.12/site-packages")
import mujoco

# Load the model to find the qpos index for coxa_T1_left
model = mujoco.MjModel.from_xml_path("/home/simllm/simrobotics-storage/research/flywire/virtual-fly/simfly_model/simfly_grounded.xml")
# Find the joint index
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name == "coxa_T1_left":
        qpos_addr = model.jnt_qposadr[i]
        print(f"Joint: {name}, qpos_addr={qpos_addr}, range={model.jnt_range[i]}")
        break
print(f"Total joints: {model.njnt}, Total qpos: {model.nq}, Total actuators: {model.nu}")
