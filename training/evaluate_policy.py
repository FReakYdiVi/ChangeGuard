"""Judge-friendly fixed-seed evaluation for ChangeGuard policies.

Compares baseline vs candidate policy over fixed seed packs.
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Any, Callable, Dict, List

from changeguard.client import EnvClient
from changeguard.models import TenantId
from training.changeguard_tool_env import ChangeGuardToolEnv
from training.train_grpo import SEED_PACKS

PolicyFn = Callable[[ChangeGuardToolEnv], None]


def baseline_policy(env: ChangeGuardToolEnv) -> None:
    """Unsafe baseline: fast promotion, minimal checks."""
    env.canary_upgrade(TenantId.A)
    env.promote_upgrade(TenantId.B)
    env.promote_upgrade(TenantId.C)


def trained_like_policy(env: ChangeGuardToolEnv) -> None:
    """Safer policy: inspect -> staged rollout -> compat -> approval -> finalize."""
    env.inspect_compatibility()
    env.inspect_logs()
    env.canary_upgrade(TenantId.A)
    env.promote_upgrade(TenantId.B)
    env.enable_compat_mode(TenantId.C)
    env.request_approval(TenantId.C)
    env.promote_upgrade(TenantId.C)


def _run_policy_on_pack(
    base_url: str,
    policy_name: str,
    policy_fn: PolicyFn,
    seed_pack: str,
    prompt_style: str,
) -> Dict[str, Any]:
    pack = SEED_PACKS[seed_pack]
    rows: List[Dict[str, Any]] = []

    for item in pack:
        env = ChangeGuardToolEnv(client=EnvClient(base_url=base_url))
        env.reset(
            difficulty=item["difficulty"],
            seed=item["seed"],
            scenario_id=item["scenario_id"],
            prompt_style=prompt_style,
        )
        try:
            policy_fn(env)
        except Exception:
            # Keep eval robust; wrapper will still produce metrics from current episode state.
            pass
        rows.append(env.build_episode_metrics())

    agg = {
        "policy": policy_name,
        "episodes": len(rows),
        "safe_full_completion": mean(r["safe_full_completion"] for r in rows),
        "safe_partial_completion": mean(r["safe_partial_completion"] for r in rows),
        "unsafe_completion": mean(r["unsafe_completion"] for r in rows),
        "approval_violation": mean(r["approval_violation"] for r in rows),
        "data_integrity_violation": mean(r["data_integrity_violation"] for r in rows),
        "blast_radius": mean(r["blast_radius"] for r in rows),
        "invalid_action_count": mean(r["invalid_action_count"] for r in rows),
        "mean_steps": mean(r["mean_steps"] for r in rows),
        "reward_total": mean(r["reward_total"] for r in rows),
    }

    # Reward components (judge-friendly visibility).
    reward_keys = [
        "progress_reward",
        "inspection_reward",
        "safety_reward",
        "invalid_action_penalty",
        "loop_penalty",
        "terminal_bonus_or_penalty",
    ]
    for key in reward_keys:
        agg[key] = mean(r[key] for r in rows)

    return {"aggregate": agg, "per_episode": rows}


def _print_comparison(base: Dict[str, Any], cand: Dict[str, Any]) -> None:
    b = base["aggregate"]
    c = cand["aggregate"]

    print("\n=== ChangeGuard Fixed-Seed Evaluation ===")
    print(f"Baseline: {b['policy']} | Candidate: {c['policy']}")
    print("\nHeadline (higher is better unless noted):")
    print(f"- safe_full_completion: {b['safe_full_completion']:.2f} -> {c['safe_full_completion']:.2f}")
    print(f"- safe_partial_completion: {b['safe_partial_completion']:.2f} -> {c['safe_partial_completion']:.2f}")
    print(f"- unsafe_completion (lower better): {b['unsafe_completion']:.2f} -> {c['unsafe_completion']:.2f}")
    print(f"- approval_violation (lower better): {b['approval_violation']:.2f} -> {c['approval_violation']:.2f}")
    print(f"- reward_total: {b['reward_total']:.2f} -> {c['reward_total']:.2f}")
    print("\nSafety diagnostics:")
    print(f"- data_integrity_violation: {b['data_integrity_violation']:.2f} -> {c['data_integrity_violation']:.2f}")
    print(f"- invalid_action_count: {b['invalid_action_count']:.2f} -> {c['invalid_action_count']:.2f}")
    print(f"- blast_radius: {b['blast_radius']:.2f} -> {c['blast_radius']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs candidate on fixed seeds")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--seed-pack", default="final_demo", choices=["smoke", "short_train", "final_demo"])
    parser.add_argument("--prompt-style", default="tool_json")
    parser.add_argument("--candidate", default="trained_like", choices=["trained_like", "baseline"])
    args = parser.parse_args()

    policy_map: Dict[str, PolicyFn] = {
        "baseline": baseline_policy,
        "trained_like": trained_like_policy,
    }

    baseline = _run_policy_on_pack(
        base_url=args.base_url,
        policy_name="baseline",
        policy_fn=baseline_policy,
        seed_pack=args.seed_pack,
        prompt_style=args.prompt_style,
    )
    candidate = _run_policy_on_pack(
        base_url=args.base_url,
        policy_name=args.candidate,
        policy_fn=policy_map[args.candidate],
        seed_pack=args.seed_pack,
        prompt_style=args.prompt_style,
    )

    _print_comparison(baseline, candidate)


if __name__ == "__main__":
    main()
