#!/usr/bin/env python3
"""
Phase 17: Trace all 36 leg actuator pathways (DN -> IN -> MN -> joint).
Uses vnc_actuator_map.json for MN IDs, dn_mn_pathways.json for DN->MN tracing.
"""
import json, os, re, subprocess, sys
from collections import defaultdict

ACTUATOR_MAP = "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/vnc_bridge/vnc_actuator_map.json"
PATHWAY_DB = "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/vnc_bridge/dn_mn_pathways.json"
OUTPUT_DIR = "/tmp/simfly_web/phase17/pathways"

LEG_JOINTS = ["coxa", "coxa_abduct", "coxa_twist", "femur", "femur_twist", "tibia"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
SEGMENTS = {
    "T1_left": "prothoracic_L", "T1_right": "prothoracic_R",
    "T2_left": "mesothoracic_L", "T2_right": "mesothoracic_R",
    "T3_left": "metathoracic_L", "T3_right": "metathoracic_R",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ACTUATOR_MAP) as f:
    amap = json.load(f)

actuator_mns = {}
for leg in LEGS:
    segment = SEGMENTS[leg]
    seg_data = amap.get(segment, {})
    for joint in LEG_JOINTS:
        name = f"{joint}_{leg}"
        joint_data = seg_data.get(name, {})
        mn_ids = joint_data.get("agonists", [])
        actuator_mns[name] = set(mn_ids)

print(f"Actuators with MNs: {sum(1 for v in actuator_mns.values() if v)}/{len(actuator_mns)}")
for name, mns in sorted(actuator_mns.items()):
    print(f"  {name}: {len(mns)} MNs")

def trace_dn_pathways(mn_ids):
    mn_pattern = "|".join(f'"mn_id": {x}\\b' for x in sorted(mn_ids))
    result = subprocess.run(
        ["grep", "-P", "-B", "4", "-A", "6", mn_pattern, PATHWAY_DB],
        capture_output=True, text=True, timeout=300
    )
    blocks = result.stdout.split("--\n")

    dn_info = {}
    for block in blocks:
        if not block.strip():
            continue
        m_dn = re.search(r'"manc_dn_id": (\d+)', block)
        m_in = re.search(r'"in_id": (\d+)', block)
        m_mn = re.search(r'"mn_id": (\d+)', block)
        m_dnt = re.search(r'"dn_type": "([^"]*)"', block)
        m_fw = re.search(r'"flywire_dn_id": "([^"]*)"', block)
        if m_dn and m_in and m_mn:
            dn_id = int(m_dn.group(1))
            in_id = int(m_in.group(1))
            mn_id = int(m_mn.group(1))
            if mn_id in mn_ids:
                if dn_id not in dn_info:
                    dn_info[dn_id] = {
                        "dn_type": m_dnt.group(1) if m_dnt else "unknown",
                        "flywire_dn_id": m_fw.group(1) if m_fw else "",
                        "interneurons": set(),
                        "mns": set()
                    }
                dn_info[dn_id]["interneurons"].add(in_id)
                dn_info[dn_id]["mns"].add(mn_id)

    sorted_dns = sorted(dn_info.items(), key=lambda x: len(x[1]["mns"]), reverse=True)
    total_ins = sum(len(v["interneurons"]) for v in dn_info.values())
    mn_matched = set()
    for info in dn_info.values():
        mn_matched |= info["mns"]

    return {
        "total_dns": len(dn_info),
        "total_interneurons": total_ins,
        "motor_neurons_matched": len(mn_matched),
        "motor_neurons_missing": sorted(mn_ids - mn_matched),
        "descending_neurons": [
            {
                "manc_dn_id": dn_id,
                "dn_type": info["dn_type"],
                "flywire_dn_id": info["flywire_dn_id"],
                "mn_count": len(info["mns"]),
                "interneuron_count": len(info["interneurons"]),
                "mns": sorted(info["mns"]),
                "interneurons": sorted(info["interneurons"])
            }
            for dn_id, info in sorted_dns
        ]
    }

total_traced = 0
for leg in LEGS:
    for joint in LEG_JOINTS:
        name = f"{joint}_{leg}"
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")

        if os.path.exists(output_path):
            print(f"  [OK] {name}: already exists")
            total_traced += 1
            continue

        mn_ids = actuator_mns.get(name, set())
        if not mn_ids:
            print(f"  [WARN] {name}: no MNs found, skipping")
            continue

        print(f"  [TRACE] {name}: {len(mn_ids)} MNs...", end=" ", flush=True)
        try:
            dn_data = trace_dn_pathways(mn_ids)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        pathway = {
            "actuator": name,
            "segment": SEGMENTS[leg],
            "leg": leg,
            "joint": joint,
            "motor_neurons": sorted(mn_ids),
            **dn_data,
        }

        with open(output_path, "w") as f:
            json.dump(pathway, f, indent=2)
        print(f"DNs={dn_data['total_dns']}, matched={dn_data['motor_neurons_matched']}/{len(mn_ids)}")
        total_traced += 1

print(f"\n{'='*60}")
print(f"Pathways traced: {total_traced}/36")
print(f"{'='*60}")
