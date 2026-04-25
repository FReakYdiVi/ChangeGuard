"""GRPO smoke training against Dispatch Arena.

Spins up the FastAPI server in-process, exposes Dispatch Arena's mini actions
as TRL tool methods, and runs ``max_steps`` GRPO updates with Qwen3-0.6B on
the local GPU. The goal is to confirm the training loop is wired correctly
end-to-end, not to produce a useful policy.
"""

from __future__ import annotations

import os
import time
from typing import Any, List

# Silence the experimental-feature warning before importing TRL.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.server.app import run_local_server_in_thread

MODEL_NAME = "Qwen/Qwen3-0.6B"
SEEDS = [7, 11, 13, 17]
SYSTEM_PROMPT = (
    "You are a courier dispatcher controlling one courier delivering one order in a "
    "small mini-mode dispatch simulator. Use the tools to drive the courier from the "
    "hub to the pickup, pick up the order once it is ready, then drive to dropoff and "
    "complete the delivery. Call exactly one tool per turn. Tools available: wait, "
    "go_pickup, pickup, go_dropoff, dropoff. Stop when the order is delivered."
)


def _start_shared_server() -> str:
    server, _thread = run_local_server_in_thread(port=0, max_concurrent_envs=32)
    host, port = server.server_address
    # Give uvicorn a moment to bind.
    time.sleep(0.2)
    return f"http://{host}:{port}"


SERVER_URL = _start_shared_server()


def _summarize(obs) -> str:
    state = obs.state
    courier = state.couriers[0] if state.couriers else None
    order = state.orders[0] if state.orders else None
    parts = [
        f"tick={state.tick}/{state.max_ticks}",
        f"verdict={obs.verifier_status.value}",
        f"reward_total={obs.reward:.2f}",
        f"legal={','.join(obs.legal_actions)}",
    ]
    if courier:
        parts.append(f"courier@{courier.node_id} load={courier.load or 'none'}")
    if order:
        parts.append(f"order={order.id} status={order.status.value} deadline={order.deadline_tick}")
    if obs.done:
        parts.append("DONE")
    return " | ".join(parts)


class DispatchToolEnv:
    """Tool-calling wrapper over Dispatch Arena mini mode."""

    def __init__(self) -> None:
        self.client = DispatchArenaClient(base_url=SERVER_URL, timeout_seconds=20)
        self.metrics: dict = {
            "total_reward": 0.0,
            "step_total": 0.0,
            "invalid_count": 0,
            "delivered": 0,
            "ticks": 0,
            "verdict": "in_progress",
        }

    def reset(self, seed: int | None = None, **_: Any) -> str:  # noqa: D401 - tool docstring not needed
        seed = int(seed) if seed is not None else 0
        obs = self.client.reset(seed=seed, config={"mode": "mini", "max_ticks": 12})
        self.metrics = {
            "total_reward": float(obs.reward),
            "step_total": float(obs.reward),
            "invalid_count": 0,
            "delivered": 0,
            "ticks": int(obs.state.tick),
            "verdict": obs.verifier_status.value,
        }
        return "Initial dashboard:\n" + _summarize(obs)

    def _step(self, action: str) -> str:
        obs = self.client.step(action)
        self.metrics["step_total"] += float(obs.reward)
        self.metrics["total_reward"] = float(obs.state.total_reward)
        self.metrics["ticks"] = int(obs.state.tick)
        self.metrics["verdict"] = obs.verifier_status.value
        if obs.info.get("invalid_action"):
            self.metrics["invalid_count"] += 1
        if obs.verifier_status.value == "delivered_successfully":
            self.metrics["delivered"] = 1
        return _summarize(obs)

    def wait(self) -> str:
        """Wait one tick. Use when the order is not ready yet."""
        return self._step("wait")

    def go_pickup(self) -> str:
        """Move the courier to the pickup node."""
        return self._step("go_pickup")

    def pickup(self) -> str:
        """Pick up the order. Only legal at pickup once the order is ready."""
        return self._step("pickup")

    def go_dropoff(self) -> str:
        """Move the courier to the dropoff node while carrying the order."""
        return self._step("go_dropoff")

    def dropoff(self) -> str:
        """Drop off the order. Only legal at dropoff while carrying."""
        return self._step("dropoff")


def reward_total(environments: List[DispatchToolEnv], **_: Any) -> List[float]:
    return [float(env.metrics.get("step_total", 0.0)) for env in environments]


def reward_validity(environments: List[DispatchToolEnv], **_: Any) -> List[float]:
    # Penalize each invalid call, mirror the env's invalid penalty magnitude.
    return [-1.0 * float(env.metrics.get("invalid_count", 0)) for env in environments]


def reward_delivery(environments: List[DispatchToolEnv], **_: Any) -> List[float]:
    return [float(env.metrics.get("delivered", 0)) for env in environments]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; smoke run requires a GPU.")

    train_dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Begin the shift. "},
                ],
                "seed": seed,
            }
            for seed in SEEDS
        ]
    )

    config = GRPOConfig(
        output_dir="dispatch_arena/scripts/_grpo_smoke_out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=192,
        max_tool_calling_iterations=8,
        learning_rate=1e-5,
        max_steps=2,
        beta=0.0,                   # no KL → skips reference model
        log_completions=True,
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        bf16=False,
        fp16=True,
        model_init_kwargs={"torch_dtype": "float16"},
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_total, reward_validity, reward_delivery],
        args=config,
        train_dataset=train_dataset,
        environment_factory=DispatchToolEnv,
    )

    train_output = trainer.train()
    print("\n=== TRAIN DONE ===")
    print("metrics:", train_output.metrics)


if __name__ == "__main__":
    main()
