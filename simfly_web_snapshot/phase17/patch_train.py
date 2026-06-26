import sys
with open("/tmp/simfly_web/phase16/train_actuator.py") as f:
    content = f.read()

old = """    if not os.path.exists(pathway):
        print(f"\\u26a0  Pathway not found: {pathway}")"""

new = """    if pathway and not os.path.exists(pathway):
        print(f"\\u26a0  Pathway not found: {pathway}")"""

content = content.replace(old, new)

# Also check phase17 pathways
after = """        pathway = None"""

phase17_check = """
    # Also check Phase 17 pathways directory
    if not pathway:
        p17 = f"/tmp/simfly_web/phase17/pathways/{args.joint}.json"
        if os.path.exists(p17):
            pathway = p17
            print(f"  Using Phase 17 pathway: {p17}")"""

content = content.replace(after, after + phase17_check)

with open("/tmp/simfly_web/phase16/train_actuator.py", "w") as f:
    f.write(content)
print("Patched train_actuator.py")
