"""Minimal GRPO training entrypoint for ChangeGuard.

Defaults to a cheap, debuggable dry-run. Use `--no-dry-run` to attempt actual
GRPO training with TRL's GRPOTrainer.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

from changeguard.client import EnvClient
from changeguard.models import TenantId
from training.changeguard_tool_env import ChangeGuardToolEnv

SEED_PACKS: Dict[str, List[Dict[str, Any]]] = {
    "smoke": [
        {"seed": 101, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 102, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 103, "difficulty": "easy", "scenario_id": "default"},
    ],
    "short_train": [
        {"seed": 201, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 202, "difficulty": "easy", "scenario_id": "default"},
        {"seed": 203, "difficulty": "easy", "scenario_id": "default"},
        {"seed": 204, "difficulty": "medium", "scenario_id": "medium_mixed"},
        {"seed": 205, "difficulty": "medium", "scenario_id": "default"},
        {"seed": 206, "difficulty": "medium", "scenario_id": "default"},
        {"seed": 207, "difficulty": "hard", "scenario_id": "hard_fragile"},
        {"seed": 208, "difficulty": "hard", "scenario_id": "default"},
    ],
    "final_demo": [
        {"seed": 301, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 302, "difficulty": "easy", "scenario_id": "default"},
        {"seed": 303, "difficulty": "easy", "scenario_id": "default"},
        {"seed": 304, "difficulty": "medium", "scenario_id": "medium_mixed"},
        {"seed": 305, "difficulty": "medium", "scenario_id": "default"},
        {"seed": 306, "difficulty": "medium", "scenario_id": "default"},
        {"seed": 307, "difficulty": "medium", "scenario_id": "default"},
        {"seed": 308, "difficulty": "hard", "scenario_id": "hard_fragile"},
        {"seed": 309, "difficulty": "hard", "scenario_id": "default"},
        {"seed": 310, "difficulty": "hard", "scenario_id": "default"},
        {"seed": 311, "difficulty": "hard", "scenario_id": "default"},
        {"seed": 312, "difficulty": "hard", "scenario_id": "hard_fragile"},
    ],
}


@dataclass
class TrainConfig:
    base_url: str = "http://127.0.0.1:8080"
    difficulty: str = "easy"
    seed: int = 101
    scenario_id: str = "tenantsafe_v1"
    prompt_style: str = "tool_json"
    dry_run: bool = True
    output_dir: str = "./artifacts/grpo_debug"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_steps: int = 8
    prompt_repeats: int = 8
    seed_pack: str = "smoke"


def build_environment_factory(config: TrainConfig):
    """Return TRL-compatible environment factory closure."""

    def _factory() -> ChangeGuardToolEnv:
        client = EnvClient(base_url=config.base_url)
        return ChangeGuardToolEnv(client=client)

    return _factory


def build_tiny_prompt_dataset(prompt_repeats: int) -> List[Dict[str, str]]:
    """Create tiny repeated-prompt dataset for fast debug runs."""
    prompt = (
        "You are a cautious upgrade agent. Use tools to migrate A then B, "
        "handle Tenant C compatibility and approval, and avoid unsafe finalize."
    )
    return [{"prompt": prompt} for _ in range(prompt_repeats)]


def run_dry_run(config: TrainConfig) -> Dict[str, Any]:
    """Execute one deterministic episode through the tool wrapper and print metrics."""
    env = build_environment_factory(config)()
    obs = env.reset(
        difficulty=config.difficulty,
        seed=config.seed,
        scenario_id=config.scenario_id,
        prompt_style=config.prompt_style,
    )

    print("[dry-run] start:", obs.summary_text)

    # Canonical safe path for visible logs and quick sanity check.
    actions = [
        ("inspect_compatibility", {}),
        ("canary_upgrade", {"tenant": TenantId.A}),
        ("promote_upgrade", {"tenant": TenantId.B}),
        ("enable_compat_mode", {"tenant": TenantId.C}),
        ("request_approval", {"tenant": TenantId.C}),
        ("promote_upgrade", {"tenant": TenantId.C}),
    ]

    step_logs: List[Dict[str, Any]] = []
    for idx, (tool_name, kwargs) in enumerate(actions, start=1):
        result = env.call_tool(tool_name, kwargs)
        log = {
            "step": idx,
            "tool": tool_name,
            "reward": result.reward_total,
            "done": result.done,
            "verdict": result.verifier_flags.verdict.value,
        }
        step_logs.append(log)
        print(f"[dry-run] step={idx} tool={tool_name} reward={result.reward_total:.2f} done={result.done}")
        if result.done:
            break

    summary = env.get_episode_summary().to_dict()
    metrics = {
        "reward_total": env.reward_total,
        "reward_components": env.reward_components,
        "violation_flags": env.violation_flags,
        "episode_metrics": env.build_episode_metrics(),
        "done": env.done,
        "episode_summary": summary,
        "step_logs": step_logs,
    }
    print("[dry-run] final verdict:", summary.get("final_verdict"))
    print("[dry-run] reward components:", env.reward_components)
    return metrics


def run_curriculum_smoke(config: TrainConfig) -> Dict[str, Any]:
    """Run quick curriculum episodes and emit training-visible aggregated metrics."""
    pack = SEED_PACKS.get(config.seed_pack, SEED_PACKS["smoke"])
    rows: List[Dict[str, Any]] = []
    for item in pack:
        env = build_environment_factory(config)()
        env.reset(
            difficulty=item["difficulty"],
            seed=item["seed"],
            scenario_id=item["scenario_id"],
            prompt_style=config.prompt_style,
        )
        # Simple safe baseline policy for cheap metric visibility.
        env.inspect_compatibility()
        env.canary_upgrade(TenantId.A)
        env.promote_upgrade(TenantId.B)
        env.enable_compat_mode(TenantId.C)
        env.request_approval(TenantId.C)
        env.promote_upgrade(TenantId.C)
        rows.append(env.build_episode_metrics())

    agg = {
        "episodes": len(rows),
        "safe_full_completion_rate": mean(r["safe_full_completion"] for r in rows),
        "safe_partial_completion_rate": mean(r["safe_partial_completion"] for r in rows),
        "unsafe_completion_rate": mean(r["unsafe_completion"] for r in rows),
        "approval_violation_rate": mean(r["approval_violation"] for r in rows),
        "data_integrity_violation_rate": mean(r["data_integrity_violation"] for r in rows),
        "mean_blast_radius": mean(r["blast_radius"] for r in rows),
        "mean_invalid_action_count": mean(r["invalid_action_count"] for r in rows),
        "mean_steps": mean(r["mean_steps"] for r in rows),
        "mean_reward_total": mean(r["reward_total"] for r in rows),
    }
    print("[curriculum-smoke]", agg)
    return {"aggregate": agg, "per_episode": rows}


def run_grpo_training(config: TrainConfig) -> Dict[str, Any]:
    """Run TRL GRPOTrainer with environment_factory (if dependencies are installed)."""
    try:
        datasets_mod = importlib.import_module("datasets")
        trl_mod = importlib.import_module("trl")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GRPO dependencies missing. Install `trl` and `datasets`, or run with --dry-run."
        ) from exc

    Dataset = getattr(datasets_mod, "Dataset")
    GRPOConfig = getattr(trl_mod, "GRPOConfig")
    GRPOTrainer = getattr(trl_mod, "GRPOTrainer")

    dataset_rows = build_tiny_prompt_dataset(config.prompt_repeats)
    train_dataset = Dataset.from_list(dataset_rows)

    trainer_args = GRPOConfig(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=config.model_name,
        args=trainer_args,
        train_dataset=train_dataset,
        environment_factory=build_environment_factory(config),
    )

    train_result = trainer.train()
    # Keep visible reward metrics easy to inspect.
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics.update(
        {
            "status": "trained",
            "max_steps": config.max_steps,
            "prompt_repeats": config.prompt_repeats,
        }
    )
    print("[train] metrics:", metrics)
    return metrics


def run_training(config: TrainConfig) -> Dict[str, Any]:
    if config.dry_run:
        result = run_dry_run(config)
        result["curriculum_smoke"] = run_curriculum_smoke(config)
        return result
    return run_grpo_training(config)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="ChangeGuard GRPO trainer")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--scenario-id", default="tenantsafe_v1")
    parser.add_argument("--prompt-style", default="tool_json")
    parser.add_argument("--output-dir", default="./artifacts/grpo_debug")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--prompt-repeats", type=int, default=8)
    parser.add_argument("--seed-pack", default="smoke", choices=["smoke", "short_train", "final_demo"])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def main() -> None:
    config = parse_args()
    print("[config]", asdict(config))
    run_training(config)


if __name__ == "__main__":
    main()
