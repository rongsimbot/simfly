import re

with open("/tmp/simfly_web/server_cpp.py", "r") as f:
    content = f.read()

# 1. Add --target-joint CLI argument
old_args = """    parser.add_argument('--tau-decay', type=float, default=50.0)"""
new_args = """    parser.add_argument('--tau-decay', type=float, default=50.0)
    parser.add_argument('--target-joint', type=str, default=None,
                        help='Phase 15: isolate a single joint (e.g. coxa_T1_left). All other joints get zero torque.')"""

content = content.replace(old_args, new_args)

# 2. Add target_joint to SimFlyServer.initialize signature
old_init_sig = """    def initialize(self, num_neurons=200, global_gain=0.002, tau_decay=50.0):"""
new_init_sig = """    def initialize(self, num_neurons=200, global_gain=0.002, tau_decay=50.0, target_joint=None):"""

content = content.replace(old_init_sig, new_init_sig)

# 3. Pass target_joints to SimFlyLoop after creation
old_loop = """            self.loop = SimFlyLoop(
                engine=self.engine, sensor_idx=sensor_idx, bridge=self.bridge, decoder=self.decoder,
                loader=self.loader, model=self.model, data=self.data, act_map=self.act_map,
                vision=self.vision, chemo=chemo, mechano=mechano, burst_inj=burst_inj,
                food_pos=FOOD_POS,
            )"""

new_loop = """            self.loop = SimFlyLoop(
                engine=self.engine, sensor_idx=sensor_idx, bridge=self.bridge, decoder=self.decoder,
                loader=self.loader, model=self.model, data=self.data, act_map=self.act_map,
                vision=self.vision, chemo=chemo, mechano=mechano, burst_inj=burst_inj,
                food_pos=FOOD_POS,
            )
            # Phase 15: Per-Actuator Connectome Control
            if target_joint:
                self.loop.target_joints = {target_joint}
                print(f"  PHASE15: Isolated actuator mode - ONLY {target_joint} receives torque")
                print(f"  All other {self.model.nu - 1} actuators = ZERO torque (neutral position)")"""

if old_loop not in content:
    print("ERROR: Could not find old_loop pattern")
    exit(1)

content = content.replace(old_loop, new_loop)

# 4. Pass target_joint from CLI to initialize call
old_main_call = """    ok = sim_server.initialize(num_neurons=args.neurons, global_gain=args.global_gain, tau_decay=args.tau_decay)"""
new_main_call = """    ok = sim_server.initialize(num_neurons=args.neurons, global_gain=args.global_gain, tau_decay=args.tau_decay, target_joint=args.target_joint)"""

content = content.replace(old_main_call, new_main_call)

with open("/tmp/simfly_web/server_cpp.py", "w") as f:
    f.write(content)

print("Patch2 applied successfully!")
print(f"Content size: {len(content)} bytes")
