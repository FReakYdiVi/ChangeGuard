"""GRPO training against Dispatch Arena (normal mode, catalog-driven).

End-to-end pipeline:
  1. Load `catalog.json` -> stratified 70/30 train/eval split.
  2. Spin up the FastAPI server in-process; one DispatchToolEnv per rollout.
  3. Each env is configured per-row from the catalog spec (mode=normal,
     plus rolling_arrivals / traffic_noise / scenario_bucket / etc.).
  4. Tool methods exposed to the LLM:
        view_dashboard, assign, reposition, hold, prioritize, finish_shift
  5. Reward function: reward_total (sum of env's per-step
     RewardBreakdown.total_reward over the rollout). Single function — the
     env already decomposes the reward; we report the scalar to GRPO.
  6. Training: TRL GRPOTrainer + LoRA (r=16). Smoke-friendly defaults for
     a Tesla T4 (16 GB) with grad checkpointing on.

Hyperparameters confirmed by user before run:
  max_steps=50, num_generations=2, max_completion_length=512,
  max_tool_calling_iterations=20, LoRA on, beta=0.0, fp16,
  per_device_train_batch_size=2, lr=1e-5.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Silence the experimental-feature warning before importing TRL.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


# NOTE: Qwen2.5-Instruct required a chat-template patch (TRL 1.2.0 ships the
# template but never wired it into add_response_schema). We dropped that path
# in favor of Qwen3-0.6B, which TRL recognizes natively — simpler stack, no
# monkey-patching, and the original smoke script used this model successfully.
# The git history of train_grpo_smoke.py shows the patch logic if it's needed
# again later.

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from dispatch_arena.catalog.dataset import load_catalog_datasets
from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.server.app import run_local_server_in_thread

# ---------------------------------------------------------------------------
# Model / paths
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"  # Natively recognized by TRL 1.2.0 (no patch needed). Pre-flight verified: tools render into prompt, envelope identical to 0.6B, ~6.7 GB total VRAM on T4.
CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "catalog.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "_grpo_normal_out"

# ---------------------------------------------------------------------------
# System prompt for normal-mode dispatcher
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a real-time delivery dispatcher running one shift over a small fleet of couriers. Your job is to dispatch each order to the right courier and keep the shift moving so orders are delivered before their deadlines.

# Tool calling

Always reply with EXACTLY ONE tool call per turn, in this format (no other text):
<tool_call>
{"name": "<tool_name>", "arguments": {<args-json>}}
</tool_call>

# Available tools

```json
[
  {
    "name": "view_dashboard",
    "description": "Refresh the dashboard. Returns courier statuses, order list, deadlines, and travel times.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },
  {
    "name": "assign",
    "description": "Dispatch an idle courier to an unassigned order whose status is queued or ready.",
    "parameters": {
      "type": "object",
      "properties": {
        "courier_id": {"type": "string", "description": "e.g. courier_0, courier_1, ..."},
        "order_id":   {"type": "string", "description": "e.g. order_0, order_1, ..."}
      },
      "required": ["courier_id", "order_id"]
    }
  },
  {
    "name": "reposition",
    "description": "Pre-stage an idle courier near a busy store or upcoming dropoff.",
    "parameters": {
      "type": "object",
      "properties": {
        "courier_id": {"type": "string"},
        "node_id":    {"type": "string", "description": "hub, store_0..3, or customer_0..N"}
      },
      "required": ["courier_id", "node_id"]
    }
  },
  {
    "name": "hold",
    "description": "Wait one tick. Use when prep is not done and no good action exists.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },
  {
    "name": "prioritize",
    "description": "Mark an order as priority. Safe even if not yet assigned.",
    "parameters": {
      "type": "object",
      "properties": {"order_id": {"type": "string"}},
      "required": ["order_id"]
    }
  },
  {
    "name": "finish_shift",
    "description": "End the shift early once all visible orders are delivered.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
]
```

# Examples

Refresh the dashboard:
<tool_call>
{"name": "view_dashboard", "arguments": {}}
</tool_call>

Dispatch courier_0 to order_1:
<tool_call>
{"name": "assign", "arguments": {"courier_id": "courier_0", "order_id": "order_1"}}
</tool_call>

# Rules

- Prep time is hidden. Queued orders flip to "ready" when prep completes; the courier you dispatch may have to wait briefly at the store.
- Travel times shown are BASE estimates. With traffic noise, real ETAs can run longer.
- The shift ends automatically at max_ticks. Maximize on-time deliveries.
- One tool per turn. Output the tool call in the format above and nothing else."""

USER_KICKOFF = "Begin the shift. Call view_dashboard first to see the state, then dispatch."


# ---------------------------------------------------------------------------
# Server boot + helpers
# ---------------------------------------------------------------------------


def _start_shared_server() -> str:
    server, _thread = run_local_server_in_thread(port=0, max_concurrent_envs=64)
    host, port = server.server_address
    time.sleep(0.2)  # let uvicorn bind
    return f"http://{host}:{port}"


SERVER_URL = _start_shared_server()


def _render_dashboard(obs) -> str:
    """Compact textual dashboard rendered for the LLM.

    Lists couriers, orders (with deadline + status), an excerpt of the travel
    matrix, last events, and the legal action shape. Designed to fit inside
    ~300 tokens so the agent has room for tool-call output too.
    """
    state = obs.state
    parts = [
        f"tick={state.tick}/{state.max_ticks}  verdict={obs.verifier_status.value}  "
        f"backlog={state.backlog}  sla_pressure={state.sla_pressure:.2f}",
    ]

    # Couriers
    parts.append("couriers:")
    for c in state.couriers:
        load = c.load or "none"
        target = f" -> {c.target_node_id}(eta {c.eta_remaining})" if c.target_node_id else ""
        parts.append(f"  {c.id} @ {c.node_id} {c.status.value}{target} carrying={load}")

    # Orders
    parts.append("orders:")
    for o in state.orders:
        assigned = o.assigned_courier_id or "-"
        parts.append(
            f"  {o.id} {o.kind} {o.pickup_node_id}->{o.dropoff_node_id} "
            f"status={o.status.value} deadline=t{o.deadline_tick} assigned={assigned}"
        )

    # Travel times — compact: one line per node showing top-K nearest
    parts.append("travel_times (base, may run longer with traffic):")
    for src in [n.id for n in state.nodes]:
        row = state.travel_time_matrix.get(src, {})
        # Show all destinations in a compact format
        edges = ", ".join(f"{dst}={t}" for dst, t in row.items() if dst != src)
        parts.append(f"  {src}: {edges}")

    # Last events
    if obs.info.get("events"):
        parts.append("last_events: " + " | ".join(obs.info["events"][-4:]))

    if obs.done:
        parts.append("DONE")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool-calling environment (one per rollout via environment_factory)
# ---------------------------------------------------------------------------


class DispatchToolEnv:
    """Normal-mode dispatcher wrapper exposing 6 tools to the LLM.

    The TRL trainer instantiates one DispatchToolEnv per generation. Public
    methods become the LLM's callable tools (per TRL OpenEnv integration).

    `metrics` is read by the reward functions after the rollout finishes.
    """

    def __init__(self) -> None:
        self.client = DispatchArenaClient(base_url=SERVER_URL, timeout_seconds=30)
        self.metrics: Dict[str, Any] = {
            "step_total": 0.0,
            "invalid_count": 0,
            "delivered": 0,
            "ticks": 0,
            "verdict": "in_progress",
            "rollout_done": False,
        }

    # The trainer passes dataset row fields here (seed + _config + ...).
    # We accept **kwargs to ignore _difficulty / _skill_focus / _name without
    # leaking them into env state.
    def reset(
        self,
        seed: Optional[int] = None,
        _config: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> str:
        seed_int = int(seed) if seed is not None else 0
        config = _config or {"mode": "normal", "max_ticks": 16, "num_couriers": 3, "num_orders": 5}
        obs = self.client.reset(seed=seed_int, config=config)
        self.metrics = {
            "step_total": float(obs.reward),
            "invalid_count": 0,
            "delivered": 0,
            "ticks": int(obs.state.tick),
            "verdict": obs.verifier_status.value,
            "rollout_done": False,
        }
        return "Initial dashboard:\n" + _render_dashboard(obs)

    def _step(self, action: Dict[str, Any]) -> str:
        if self.metrics.get("rollout_done"):
            return "Shift already finished — call finish_shift to stop or stop calling tools."
        obs = self.client.step(action)
        self.metrics["step_total"] += float(obs.reward)
        self.metrics["ticks"] = int(obs.state.tick)
        self.metrics["verdict"] = obs.verifier_status.value
        self.metrics["delivered"] = sum(
            1 for o in obs.state.orders if o.status.value == "delivered"
        )
        if obs.info.get("invalid_action"):
            self.metrics["invalid_count"] += 1
        if obs.done:
            self.metrics["rollout_done"] = True
        return _render_dashboard(obs)

    # ---- Tools (each is exposed to the LLM as a callable) -----------------

    def view_dashboard(self) -> str:
        """Refresh the dashboard with the latest courier/order state."""
        # No-op step semantically — but our env doesn't separate "look" from
        # "act", so we issue a hold instead to advance one tick.
        return self._step({"action_type": "hold"})

    def assign(self, courier_id: str, order_id: str) -> str:
        """Dispatch a courier to an order. Both must be valid + free.

        Args:
            courier_id: e.g. "courier_0".
            order_id:   e.g. "order_3".
        """
        return self._step(
            {"action_type": "assign", "courier_id": courier_id, "order_id": order_id}
        )

    def reposition(self, courier_id: str, node_id: str) -> str:
        """Move an idle courier to a node to pre-stage near a busy store.

        Args:
            courier_id: e.g. "courier_1".
            node_id:    e.g. "store_0", "hub", "customer_2".
        """
        return self._step(
            {"action_type": "reposition", "courier_id": courier_id, "node_id": node_id}
        )

    def hold(self) -> str:
        """Wait one tick. Use when prep is unfinished and no good move exists."""
        return self._step({"action_type": "hold"})

    def prioritize(self, order_id: str) -> str:
        """Signal that an order is priority. Safe even if not assigned.

        Args:
            order_id: e.g. "order_2".
        """
        return self._step({"action_type": "prioritize", "order_id": order_id})

    def finish_shift(self) -> str:
        """End the shift early. Returns the final summary."""
        # Mark rollout done; TRL will stop tool-calling once the next
        # iteration sees the rollout flag. We also issue a hold to advance
        # the tick so the env can finalize.
        if not self.metrics.get("rollout_done"):
            self._step({"action_type": "hold"})
        self.metrics["rollout_done"] = True
        return (
            f"Shift finished. tick={self.metrics['ticks']} delivered={self.metrics['delivered']} "
            f"verdict={self.metrics['verdict']} reward={self.metrics['step_total']:.2f}"
        )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def reward_total(environments: List[DispatchToolEnv], **_: Any) -> List[float]:
    """Sum of env's per-step RewardBreakdown.total_reward across the rollout.

    Already includes step_cost, progress, success, invalid_penalty, on-time
    bonus, late penalty, timeout penalty, idle penalty, churn, fairness — so
    additional reward functions would be double-counts of components inside
    this scalar.
    """
    return [float(env.metrics.get("step_total", 0.0)) for env in environments]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; this trainer requires a GPU.")

    train_ds, eval_ds, train_specs, eval_specs = load_catalog_datasets(
        catalog_path=CATALOG_PATH,
        system_prompt=SYSTEM_PROMPT,
        eval_fraction=0.30,
        master_seed=0,
    )
    print(f"Catalog loaded: train={len(train_ds)} eval={len(eval_ds)}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,    # OOM fix: only 1 prompt per micro-batch
        gradient_accumulation_steps=4,    # generation_batch_size = 1*4*1 = 4, divisible by G=4
        num_generations=4,                # bump from 2 for better advantage variance
        max_completion_length=384,        # OOM fix: was 512, less KV cache + caps rambling earlier
        max_tool_calling_iterations=20,
        learning_rate=1e-5,
        max_steps=80,                     # longer horizon for the policy to actually move
        beta=0.0,                   # no KL -> skips reference model
        log_completions=True,
        report_to=["tensorboard"],
        logging_dir=str(OUTPUT_DIR / "tb"),
        save_strategy="no",         # smoke-friendly; no checkpoints to disk
        eval_strategy="no",         # post-training eval is a separate script
        logging_steps=1,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        model_init_kwargs={"torch_dtype": "float16"},
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_total],
        args=config,
        train_dataset=train_ds,
        environment_factory=DispatchToolEnv,
        peft_config=lora_config,
    )

    print("Starting training...")
    train_output = trainer.train()
    print("\n=== TRAIN DONE ===")
    print("metrics:", train_output.metrics)

    # Persist the trained LoRA adapter so eval can load it later.
    # GRPOConfig was set with save_strategy="no" to avoid mid-run checkpoints,
    # but we explicitly save the final state here. trainer.save_model() writes
    # the adapter (since peft is in use) — base model weights are not duplicated.
    final_dir = OUTPUT_DIR / "final_lora"
    trainer.save_model(str(final_dir))
    print(f"LoRA adapter saved -> {final_dir}")


if __name__ == "__main__":
    main()
