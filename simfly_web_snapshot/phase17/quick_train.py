#!/usr/bin/env python3
"""
Phase 18: Quick Per-Joint RL Training
Uses Phase16's proven PerActuatorRL with joint-specific angle limits.
"""
import sys, os, argparse, json, time, subprocess

sys.stdout.reconfigure(line_buffering=True)
# Joint-specific params (angle limit, displacement weight)
JOINT_PARAMS = {
    'coxa_T':       {'limit': 0.8,  'disp_w': 10.0},
    'coxa_abduct':  {'limit': 0.7,  'disp_w': 6.0},
    'coxa_twist':   {'limit': 0.6,  'disp_w': 4.0},
    'femur_T':      {'limit': 1.2,  'disp_w': 10.0},
    'femur_twist':  {'limit': 0.7,  'disp_w': 5.0},
    'tibia_T':      {'limit': 1.0,  'disp_w': 8.0},
}

def get_params(joint_name):
    name = joint_name.lower()
    if 'coxa_abduct' in name: return JOINT_PARAMS['coxa_abduct']
    elif 'coxa_twist' in name: return JOINT_PARAMS['coxa_twist']
    elif 'coxa_' in name: return JOINT_PARAMS['coxa_T']
    elif 'femur_twist' in name: return JOINT_PARAMS['femur_twist']
    elif 'femur_' in name: return JOINT_PARAMS['femur_T']
    elif 'tibia_' in name: return JOINT_PARAMS['tibia_T']
    return {'limit': 0.8, 'disp_w': 10.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--server', default='http://localhost:8080')
    parser.add_argument('--save-dir', default='/tmp/simfly_web/phase17/models')
    parser.add_argument('--timeout', type=float, default=0.05)
    args = parser.parse_args()

    params = get_params(args.joint)
    joint_limit = params['limit']
    disp_weight = params['disp_w']
    os.makedirs(args.save_dir, exist_ok=True)

    # Use Phase16 PerActuatorRL directly, just with correct limit
    sys.path.insert(0, '/tmp/simfly_web/phase16')
    from per_actuator_rl import PerActuatorRL, DEVICE

    pathway = f'/tmp/simfly_web/phase17/pathways/{args.joint}.json'
    if not os.path.exists(pathway):
        pathway = None

    agent = PerActuatorRL(
        joint_name=args.joint,
        server_url=args.server,
        pathway_json=pathway,
        hidden_dim=64,
        lr=3e-4,
        joint_angle_limit=joint_limit,
        angular_displacement_weight=disp_weight,
    )

    print(f"\n{'='*60}")
    print(f"Phase 18: Training {args.joint}")
    print(f"  Joint angle limit: {joint_limit}, disp_weight: {disp_weight}")
    print(f"  Episodes: {args.episodes} x {args.steps} steps")
    print(f"  State dim: {agent.state_dim} | Device: {DEVICE}")
    print(f"{'='*60}\n")

    metrics = agent.train(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        collect_timeout=args.timeout,
        verbose=True,
    )

    # Save
    save_path = os.path.join(args.save_dir, f'{args.joint}.pt')
    agent.save(save_path)

    # Print summary
    rewards = metrics['episode_reward']
    print(f"\n{'='*60}")
    print(f"Training Complete: {args.joint}")
    print(f"  Best: {max(rewards):.2f}")
    print(f"  Final avg10: {sum(rewards[-10:])/10:.2f}")
    print(f"  Model: {save_path}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
