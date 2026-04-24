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
from changeguard.training.changeguard_tool_env import ChangeGuardToolEnv

SEED_PACKS: Dict[str, List[Dict[str, Any]]] = {
    # Backcompat-only: tests + dry-run visual sanity. Fixed diff = [RENAME_COL].
    "smoke": [
        {"seed": 101, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 102, "difficulty": "easy", "scenario_id": "easy_stable"},
        {"seed": 103, "difficulty": "easy", "scenario_id": "default"},
    ],
    # Procedural: varied diff ops + tenant deps for generalization training.
    "short_train": [
        {"seed": 201, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 202, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 203, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 204, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 205, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 206, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 207, "difficulty": "hard", "scenario_id": "procedural_hard"},
        {"seed": 208, "difficulty": "hard", "scenario_id": "procedural_hard"},
    ],
    # Mixed procedural + backcompat so the eval demo can compare.
    "final_demo": [
        {"seed": 301, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 302, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 303, "difficulty": "easy", "scenario_id": "procedural_easy"},
        {"seed": 304, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 305, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 306, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 307, "difficulty": "medium", "scenario_id": "procedural_medium"},
        {"seed": 308, "difficulty": "hard", "scenario_id": "procedural_hard"},
        {"seed": 309, "difficulty": "hard", "scenario_id": "procedural_hard"},
        {"seed": 310, "difficulty": "hard", "scenario_id": "procedural_hard"},
        {"seed": 311, "difficulty": "hard", "scenario_id": "procedural_hard"},
        {"seed": 312, "difficulty": "hard", "scenario_id": "procedural_hard"},
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
    # Qwen2.5-1.5B (vs 0.5B) holds the tool-call template more stably under LoRA
    # updates, which keeps tools/failure_frequency low across GRPO steps.
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_steps: int = 8
    prompt_repeats: int = 8
    seed_pack: str = "smoke"
    lora: bool = False
    # Stability knobs (P1 fixes for the step-6+ collapse seen in the 0.5B runs).
    num_generations: int = 4        # larger group -> reliable group-relative signal
    learning_rate: float = 5e-7     # slower drift away from valid-JSON basin
    max_grad_norm: float = 1.0      # clip spiky updates
    beta: float = 0.04              # KL penalty toward the reference model
    temperature: float = 1.0        # sampling diversity for GRPO rollouts
    top_p: float = 0.95


def _make_env_reward_func(trainer_ref: Dict[str, Any]):
    """Build a reward_func that reads cumulative env reward from the trainer.

    TRL's `environment_factory` provides tool dynamics but not reward — each
    trainer.environments[i] accumulates `reward_total` during its rollout, and
    we surface that as the GRPO reward signal. Mutable dict is used to sidestep
    the chicken-and-egg of reward_funcs being required at trainer construction.
    """

    def _env_reward_func(prompts, completions, **_kwargs):
        trainer = trainer_ref.get("trainer")
        if trainer is None or getattr(trainer, "environments", None) is None:
            return [0.0] * len(completions)
        envs = trainer.environments
        n_envs = len(envs)
        if n_envs == 0:
            return [0.0] * len(completions)
        return [float(envs[i % n_envs].reward_total) for i in range(len(completions))]

    _env_reward_func.__name__ = "env_reward"
    return _env_reward_func


def _build_processing_class(model_name: str):
    """Build tokenizer/processing class compatible with TRL tool parsing.

    For some Qwen2.5 variants, TRL may fail to infer a response schema from the
    bundled chat template. In that case, we apply a compatible Qwen tool template
    and retry schema registration.
    """
    transformers_mod = importlib.import_module("transformers")
    AutoTokenizer = getattr(transformers_mod, "AutoTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "response_schema", None) is not None:
        return tokenizer

    chat_utils_mod = importlib.import_module("trl.chat_template_utils")
    add_response_schema = getattr(chat_utils_mod, "add_response_schema")

    try:
        return add_response_schema(tokenizer)
    except Exception as exc:
        # Qwen2.5 fallback: swap to a known Qwen tools template then retry.
        if "Qwen/Qwen2.5" not in model_name:
            raise RuntimeError(
                f"Failed to prepare response schema for model '{model_name}'."
            ) from exc

        qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
        if not getattr(qwen3_tokenizer, "chat_template", None):
            raise RuntimeError("Qwen3 fallback tokenizer has no chat_template.") from exc
        tokenizer.chat_template = qwen3_tokenizer.chat_template
        return add_response_schema(tokenizer)


def build_environment_factory(config: TrainConfig):
    """Return TRL-compatible environment factory closure."""

    def _factory() -> ChangeGuardToolEnv:
        client = EnvClient(base_url=config.base_url)
        return ChangeGuardToolEnv(client=client)

    return _factory


def build_prompt_dataset(seed_pack: str, repeats: int = 1) -> List[Dict[str, Any]]:
    """Build a seeded training dataset from a SEED_PACK.

    Each row carries `seed`, `difficulty`, and `scenario_id` alongside the
    chat-format `prompt`. TRL forwards every non-prompt column to
    `env.reset(**reset_kwargs)`, giving us deterministic per-rollout worlds.
    """
    pack = SEED_PACKS.get(seed_pack, SEED_PACKS["smoke"])
    system_msg = (
        "You are a cautious upgrade agent. The V2 migration has a schema diff "
        "(shown in 'diff=[...]'). Each op kind needs a mitigation ONLY if some "
        "tenant has the matching sensitivity (revealed by inspect_tenant). "
        "Mapping: RENAME_COL->enable_compat_mode, ADD_NOT_NULL_COL->apply_backfill, "
        "DROP_COL->apply_announce_deprecation, CHANGE_TYPE->apply_dual_write; "
        "ADD_NULL_COL is always safe. Plan: inspect_compatibility, inspect_tenant "
        "each tenant, apply required mitigations, then canary_upgrade(A), "
        "promote_upgrade(B), request_approval(C), promote_upgrade(C). If in doubt, "
        "defer_tenant(C). Current state: "
    )
    messages = [{"role": "user", "content": system_msg}]
    rows = [
        {
            "prompt": messages,
            "seed": item["seed"],
            "difficulty": item["difficulty"],
            "scenario_id": item["scenario_id"],
        }
        for item in pack
    ]
    return rows * max(1, repeats)


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
        result = env._call_tool(tool_name, kwargs)
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

    summary = env._get_episode_summary().to_dict()
    metrics = {
        "reward_total": env.reward_total,
        "reward_components": env.reward_components,
        "violation_flags": env.violation_flags,
        "episode_metrics": env._build_episode_metrics(),
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
        rows.append(env._build_episode_metrics())

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


def _prepare_lora_model(config: TrainConfig):
    """Load base model in bf16/fp16 and wrap with peft LoRA for T4-friendly GRPO.

    Returns (model, use_bf16) where use_bf16 indicates which half-precision dtype was
    selected (T4 lacks proper bf16 support — we fall back to fp16 when bf16 is
    unavailable).
    """
    torch_mod = importlib.import_module("torch")
    transformers_mod = importlib.import_module("transformers")
    peft_mod = importlib.import_module("peft")
    AutoModelForCausalLM = getattr(transformers_mod, "AutoModelForCausalLM")
    LoraConfig = getattr(peft_mod, "LoraConfig")
    get_peft_model = getattr(peft_mod, "get_peft_model")

    cuda_available = bool(getattr(torch_mod.cuda, "is_available", lambda: False)())
    bf16_supported = False
    if cuda_available and hasattr(torch_mod.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch_mod.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False

    dtype = torch_mod.bfloat16 if bf16_supported else torch_mod.float16
    print(f"[lora] base dtype={'bf16' if bf16_supported else 'fp16'} cuda={cuda_available}")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, bf16_supported


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

    dataset_rows = build_prompt_dataset(config.seed_pack, repeats=config.prompt_repeats)
    train_dataset = Dataset.from_list(dataset_rows)
    processing_class = _build_processing_class(config.model_name)

    lora_model = None
    use_bf16 = False
    if config.lora:
        lora_model, use_bf16 = _prepare_lora_model(config)

    # Batch size must be divisible by num_generations. With num_generations=4
    # we set per_device_train_batch_size=4 so each step has 1 distinct prompt
    # with 4 rollouts (clean group-relative signal, avoids the dead-group trap
    # seen with num_generations=2).
    per_device_train_batch_size = max(config.num_generations, 2)

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=config.num_generations,
        gradient_accumulation_steps=1,
        logging_steps=1,
        report_to=[],
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        beta=config.beta,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    if config.lora:
        grpo_kwargs["gradient_checkpointing"] = True
        if use_bf16:
            grpo_kwargs["bf16"] = True
        else:
            grpo_kwargs["fp16"] = True

    trainer_args = GRPOConfig(**grpo_kwargs)

    # Deferred trainer ref: the reward_func needs to read trainer.environments,
    # but trainer construction requires reward_funcs up front.
    trainer_ref: Dict[str, Any] = {"trainer": None}
    reward_func = _make_env_reward_func(trainer_ref)

    trainer_kwargs: Dict[str, Any] = dict(
        processing_class=processing_class,
        reward_funcs=[reward_func],
        args=trainer_args,
        train_dataset=train_dataset,
        environment_factory=build_environment_factory(config),
    )
    if config.lora and lora_model is not None:
        trainer_kwargs["model"] = lora_model
    else:
        trainer_kwargs["model"] = config.model_name

    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "environment_factory" in str(exc):
            raise RuntimeError(
                "Your TRL version does not support environment_factory. "
                "Use Meta's TRL fork or downgrade."
            ) from exc
        raise

    trainer_ref["trainer"] = trainer

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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--prompt-repeats", type=int, default=8)
    parser.add_argument("--seed-pack", default="smoke", choices=["smoke", "short_train", "final_demo"])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="Use peft LoRA (r=8, alpha=16) + bf16/fp16 + gradient checkpointing. "
        "Required for fitting Qwen2.5-1.5B GRPO on a T4 16GB GPU.",
    )
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Rollouts per prompt (group size for GRPO advantage).")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient toward reference model.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def main() -> None:
    config = parse_args()
    print("[config]", asdict(config))
    run_training(config)


if __name__ == "__main__":
    main()
