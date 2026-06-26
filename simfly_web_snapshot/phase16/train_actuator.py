#!/usr/bin/env python3
"""
Phase 16: Train a single actuator RL controller.
Usage:
  python3 train_actuator.py --joint coxa_T1_left
  python3 train_actuator.py --joint coxa_T1_right --episodes 200
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_actuator_rl import PerActuatorRL, DEVICE

def main():
    parser = argparse.ArgumentParser(description="Train per-actuator RL controller")
    parser.add_argument("--joint", type=str, required=True, help="Joint name (e.g., coxa_T1_left)")
    parser.add_argument("--server", type=str, default="http://localhost:8080")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timeout", type=float, default=0.05, help="Seconds between steps")
    parser.add_argument("--save-dir", type=str, default="/tmp/simfly_web/phase16/models")
    args = parser.parse_args()

    pathway_lower = f"/tmp/simfly_web/actuator_{args.joint.lower()}_pathway.json"
    pathway = pathway_lower if os.path.exists(pathway_lower) else None
    if pathway is None or not os.path.exists(pathway):
        print(f"⚠  Pathway not found: {pathway}")
        print("   Using default state dimensions (839 DN, 61 MN)")
        pathway = None
    # Also check Phase 17 pathways directory
    if not pathway:
        p17 = f"/tmp/simfly_web/phase17/pathways/{args.joint}.json"
        if os.path.exists(p17):
            pathway = p17
            print(f"  Using Phase 17 pathway: {p17}")

    agent = PerActuatorRL(
        joint_name=args.joint,
        server_url=args.server,
        pathway_json=pathway,
        hidden_dim=args.hidden,
        lr=args.lr,
    )

    print(f"\n{'='*60}")
    print(f"Phase 16: Training {args.joint}")
    print(f"  Episodes: {args.episodes} × {args.steps} steps")
    print(f"  State dim: {agent.state_dim} | Hidden: {args.hidden}")
    print(f"  Device: {DEVICE} | Server: {args.server}")
    print(f"  Pathway: {pathway or 'default'}")
    print(f"{'='*60}\n")

    metrics = agent.train(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        collect_timeout=args.timeout,
        verbose=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.joint}.pt")
    agent.save(save_path)

    # Print summary
    rewards = metrics['episode_reward']
    print(f"\n{'='*60}")
    print(f"Training Summary: {args.joint}")
    print(f"  Episodes: {len(rewards)}")
    print(f"  Mean reward (all): {sum(rewards)/len(rewards):.2f}")
    print(f"  Mean reward (last 10): {sum(rewards[-10:])/10:.2f}")
    print(f"  Best reward: {max(rewards):.2f}")
    print(f"  Model saved: {save_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
