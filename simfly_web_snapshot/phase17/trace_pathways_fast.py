#!/usr/bin/env python3
"""
Phase 17: Fast pathway tracer using jq streaming.
One pass through the 5GB DN-MN database to build a reverse index,
then generate all 36 actuator pathways.
"""
import json, os, subprocess, sys

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

# Load actuator map
with open(ACTUATOR_MAP) as f:
    amap = json.load(f)

# Build MN->actuator lookup
all_mn_ids = set()
mn_to_actuators = {}
actuator_mns = {}
for leg in LEGS:
    for joint in LEG_JOINTS:
        name = f"{joint}_{leg}"
        segment = SEGMENTS[leg]
        seg_data = amap.get(segment, {})
        jdata = seg_data.get(name, {})
        mn_ids = set(jdata.get("agonists", []))
        actuator_mns[name] = mn_ids
        for mn in mn_ids:
            all_mn_ids.add(mn)
            if mn not in mn_to_actuators:
                mn_to_actuators[mn] = set()
            mn_to_actuators[mn].add(name)

print(f"Unique MNs across all 36 actuators: {len(all_mn_ids)}")
print(f"Scanning 5.1GB DN-MN pathway database...")

# Stream through the JSON with jq to extract DN->MN pathways
# jq -c '.pathways | to_entries[] | {dn_id: .key, type: .value.dn_type, fw: .value.flywire_dn_id, mns: [.value.pathways[]?.mn_id]}' file.json
# But this is slow. Let's use a targeted approach with grep and memory limits.

# Alternative: process in manageable chunks using Python's json incremental parsing
# The structure: {"metadata": {...}, "pathways": {"DN_ID": {"pathways": [{...}], ...}, ...}}

# Use jq to extract just MN IDs referenced, filtered by our known MNs
# Build a grep pattern of all MN IDs
print(f"Using jq streaming extraction...")
mn_list_str = ",".join(str(m) for m in sorted(all_mn_ids)[:200])  # limit grep pattern size

# Use a two-phase approach:
# Phase 1: Run jq to extract pathways filtered by MN IDs  
# Phase 2: Parse the extracted data into actuator pathways
print(f"Total MN IDs to search: {len(all_mn_ids)}")
print("Using chunked grep approach...")

# Split MN IDs into batches of 50 to avoid huge grep patterns
from collections import defaultdict
dn_actuator_map = defaultdict(lambda: defaultdict(lambda: {"interneurons": set(), "mns": set()}))

mn_sorted = sorted(all_mn_ids)
batch_size = 30
total_batches = (len(mn_sorted) + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start = batch_idx * batch_size
    batch = mn_sorted[start:start + batch_size]
    mn_pattern = "|".join(f'"mn_id": {x}\\b' for x in batch)
    
    print(f"  Batch {batch_idx+1}/{total_batches}: searching {len(batch)} MNs...", end=" ", flush=True)
    
    try:
        result = subprocess.run(
            ["grep", "-P", "-B", "2", "-A", "1", mn_pattern, PATHWAY_DB],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "LC_ALL": "C"}
        )
        blocks = result.stdout.split("--\n")
        
        parsed = 0
        for block in blocks:
            if not block.strip():
                continue
            mn_match = None
            in_match = None
            dn_match = None
            for line in block.split("\n"):
                if '"mn_id"' in line:
                    import re as _re
                    m = _re.search(r'"mn_id": (\d+)', line)
                    if m: mn_match = int(m.group(1))
                if '"in_id"' in line:
                    m = _re.search(r'"in_id": (\d+)', line)
                    if m: in_match = int(m.group(1))
                if '"manc_dn_id"' in line:
                    m = _re.search(r'"manc_dn_id": (\d+)', line)
                    if m: dn_match = int(m.group(1))
                if '"dn_type"' in line:
                    m = _re.search(r'"dn_type": "([^"]+)"', line)
                    if m and dn_match:
                        for act_name in mn_to_actuators.get(mn_match, set()):
                            dn_actuator_map[act_name][dn_match]["dn_type"] = m.group(1)
                if '"flywire_dn_id"' in line:
                    m = _re.search(r'"flywire_dn_id": "([^"]+)"', line)
                    if m and dn_match:
                        for act_name in mn_to_actuators.get(mn_match, set()):
                            dn_actuator_map[act_name][dn_match]["flywire_dn_id"] = m.group(1)
            
            if mn_match and in_match and dn_match and mn_match in mn_to_actuators:
                for act_name in mn_to_actuators[mn_match]:
                    dn_actuator_map[act_name][dn_match]["interneurons"].add(in_match)
                    dn_actuator_map[act_name][dn_match]["mns"].add(mn_match)
                parsed += 1
        
        print(f"{parsed} matches")
    except subprocess.TimeoutExpired:
        print("timeout")
    except Exception as e:
        print(f"error: {e}")

print(f"\nDone scanning. Generating pathway JSONs...")

for leg in LEGS:
    for joint in LEG_JOINTS:
        name = f"{joint}_{leg}"
        mn_ids = actuator_mns.get(name, set())
        dn_data = dn_actuator_map.get(name, {})
        
        sorted_dns = sorted(dn_data.items(), key=lambda x: len(x[1]["mns"]), reverse=True)
        total_ins = sum(len(v["interneurons"]) for v in dn_data.values())
        mn_matched = set()
        for info in dn_data.values():
            mn_matched |= info["mns"]
        
        pathway = {
            "actuator": name,
            "segment": SEGMENTS[leg],
            "leg": leg,
            "joint": joint,
            "motor_neurons": sorted(mn_ids),
            "motor_neurons_matched": len(mn_matched),
            "motor_neurons_missing": sorted(mn_ids - mn_matched),
            "total_dns": len(dn_data),
            "total_interneurons": total_ins,
            "descending_neurons": [
                {
                    "manc_dn_id": dn_id,
                    "dn_type": info.get("dn_type", "unknown"),
                    "flywire_dn_id": info.get("flywire_dn_id", ""),
                    "mn_count": len(info["mns"]),
                    "interneuron_count": len(info["interneurons"]),
                    "mns": sorted(info["mns"]),
                    "interneurons": sorted(info["interneurons"]),
                }
                for dn_id, info in sorted_dns
            ],
        }
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(output_path, "w") as f:
            json.dump(pathway, f, indent=2)
        
        print(f"  {name}: {len(mn_ids)} MNs, {len(dn_data)} DNs, {total_ins} INs")

print(f"\nAll 36 pathways saved to {OUTPUT_DIR}/")
