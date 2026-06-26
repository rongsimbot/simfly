import re, subprocess, json

# coxa_T1_left MN IDs
coxa_mn_ids = {17664, 163719, 157320, 10892, 19468, 12686, 24975, 102542, 19092, 13974, 
               17176, 47642, 153756, 156960, 178721, 166309, 24747, 26924, 29487, 13490, 
               11059, 15923, 10548, 21944, 156857, 12603, 13628, 12096, 19652, 26566, 
               10694, 10825, 25674, 10571, 24139, 155853, 19667, 22358, 10200, 12891, 
               13920, 10592, 226276, 14949, 13156, 10088, 10473, 12009, 12396, 155633, 
               24050, 10482, 29554, 17653, 14069, 19319, 21749, 21753, 219894, 20860, 29689}

# Build grep pattern and use context to capture full entries
mn_pattern = "|".join(f'"mn_id": {x}\\b' for x in sorted(coxa_mn_ids))
result = subprocess.run(
    ["grep", "-P", "-B", "4", "-A", "6", mn_pattern, 
     "/home/simllm/simrobotics-storage/research/flywire/simfly-robotic-model/vnc_bridge/dn_mn_pathways.json"],
    capture_output=True, text=True, timeout=300
)
output = result.stdout
print(f"Output length: {len(output)}")

# Parse blocks separated by '--'
blocks = output.split('--\n')

import json

dn_info = {}
entries_parsed = 0

for block in blocks:
    if not block.strip():
        continue
    # Wrap in braces and try to parse
    # But the grep context includes surrounding JSON artifacts
    # Instead, extract fields with regex from the block
    m_dn = re.search(r'"manc_dn_id": (\d+)', block)
    m_in = re.search(r'"in_id": (\d+)', block)
    m_mn = re.search(r'"mn_id": (\d+)', block)
    m_dnt = re.search(r'"dn_type": "([^"]+)"', block)
    m_fw = re.search(r'"flywire_dn_id": "([^"]+)"', block)
    if m_dn and m_in and m_mn:
        dn_id = int(m_dn.group(1))
        in_id = int(m_in.group(1))
        mn_id = int(m_mn.group(1))
        if mn_id in coxa_mn_ids:
            if dn_id not in dn_info:
                dn_info[dn_id] = {
                    "dn_type": m_dnt.group(1) if m_dnt else "unknown",
                    "flywire_dn_id": m_fw.group(1) if m_fw else "",
                    "interneurons": set(),
                    "mns": set()
                }
            dn_info[dn_id]["interneurons"].add(in_id)
            dn_info[dn_id]["mns"].add(mn_id)
            entries_parsed += 1

print(f"Blocks: {len(blocks)}, Entries parsed: {entries_parsed}")
print(f"Unique DNs: {len(dn_info)}")
total_ins = sum(len(v["interneurons"]) for v in dn_info.values())
print(f"Total interneurons: {total_ins}")

sorted_dns = sorted(dn_info.items(), key=lambda x: len(x[1]["mns"]), reverse=True)
for dn_id, info in sorted_dns[:15]:
    print(f"  DN{dn_id} ({info['dn_type']}): {len(info['mns'])} MNs, {len(info['interneurons'])} INs")
if len(sorted_dns) > 15:
    print(f"  ... and {len(sorted_dns)-15} more")

mn_matched = set()
for info in dn_info.values():
    mn_matched |= info["mns"]
print(f"MNs matched: {len(mn_matched)}/{len(coxa_mn_ids)}")

# Save pathway JSON
pathway_data = {
    "actuator": "coxa_T1_left",
    "actuator_index": 2,
    "segment": "prothoracic_L",
    "motor_neurons": sorted(coxa_mn_ids),
    "motor_neurons_matched": len(mn_matched),
    "motor_neurons_missing": sorted(coxa_mn_ids - mn_matched),
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
    ],
    "total_dns": len(dn_info),
    "total_interneurons": total_ins
}

with open("/tmp/simfly_web/actuator_coxa_t1_left_pathway.json", "w") as f:
    json.dump(pathway_data, f, indent=2)
print(f"Pathway saved to /tmp/simfly_web/actuator_coxa_t1_left_pathway.json")
