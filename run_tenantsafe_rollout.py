from __future__ import annotations

import argparse
from statistics import mean

from tenantsafe.env import TenantSafeEnv
from tenantsafe.policies import safe_policy, risky_policy


def run_episode(env: TenantSafeEnv, policy_name: str, seed: int):
    policy = safe_policy if policy_name == "safe" else risky_policy
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    truncated = False
    info = {}
    while not done and not truncated:
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TenantSafe rollout policy episodes")
    parser.add_argument("--policy", choices=["safe", "risky"], default="safe")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = TenantSafeEnv()
    rewards = []
    successes = 0
    outages = 0

    for i in range(args.episodes):
        reward, info = run_episode(env, args.policy, seed=1000 + i)
        rewards.append(reward)
        successes += int(bool(info.get("safe_completion", False)))
        outages += int(bool(info.get("outage", False)))

    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Avg reward: {mean(rewards):.2f}")
    print(f"Safe completions: {successes}/{args.episodes}")
    print(f"Outages: {outages}/{args.episodes}")


if __name__ == "__main__":
    main()
