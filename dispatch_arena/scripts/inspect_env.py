"""Print exactly what the env shows the model.

Walks one mini episode through the same client the smoke trainer uses,
dumping the full observation at reset and after every step so we can
verify nothing leaks and the labels match the values.
"""

from __future__ import annotations

import json
import time
from typing import Any

from transformers import AutoTokenizer

from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.scripts.train_grpo_smoke import (
    MODEL_NAME,
    SYSTEM_PROMPT,
    DispatchToolEnv,
    _summarize,
)
from dispatch_arena.server.app import run_local_server_in_thread


def _dump(label: str, obs) -> None:
    print(f"\n=== {label} ===")
    print(" summary_text:", obs.summary_text)
    print(" reward (this step):", obs.reward)
    print(" reward_breakdown:", obs.reward_breakdown.to_dict())
    print(" verifier_status:", obs.verifier_status.value)
    print(" done:", obs.done, "truncated:", obs.truncated)
    print(" legal_actions:", obs.legal_actions)
    print(" action_mask:", obs.action_mask)
    print(" info:", obs.info)
    print(" state.tick:", obs.state.tick, "/", obs.state.max_ticks)
    print(" state.total_reward (cumulative):", obs.state.total_reward)
    courier = obs.state.couriers[0]
    order = obs.state.orders[0]
    print(" courier:", courier.to_dict())
    print(" order:", order.to_dict())
    blob = json.dumps(obs.to_dict())
    leak = "prep_remaining" in blob
    print(" leak('prep_remaining' present):", leak)


def main() -> None:
    server, _thread = run_local_server_in_thread(port=0, max_concurrent_envs=4)
    host, port = server.server_address
    time.sleep(0.2)
    base_url = f"http://{host}:{port}"
    client = DispatchArenaClient(base_url=base_url)

    print("### Raw HTTP-client view of the env ###")
    obs = client.reset(seed=7, config={"mode": "mini", "max_ticks": 12})
    _dump("RESET (seed=7)", obs)

    plan = ["go_pickup", "wait", "wait", "pickup", "go_dropoff", "dropoff"]
    for i, action in enumerate(plan, 1):
        if obs.done:
            print(f"\n(stop: episode ended before action {i})")
            break
        if action not in obs.legal_actions:
            print(f"\n(stop: '{action}' not legal at step {i}, legal={obs.legal_actions})")
            break
        obs = client.step(action)
        _dump(f"STEP {i}: {action}", obs)

    print("\n### What DispatchToolEnv.reset returns to TRL ###")
    tool_env = DispatchToolEnv()
    tool_env.client = client  # reuse the same server
    initial = tool_env.reset(seed=7)
    print(initial)
    print("metrics after reset:", tool_env.metrics)

    print("\n### One tool-call's text return ###")
    out = tool_env.go_pickup()
    print(out)
    print("metrics after go_pickup:", tool_env.metrics)

    print("\n### Full prompt the model actually sees (after TRL appends reset string) ###")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tools_schema = []
    for name in ("wait", "go_pickup", "pickup", "go_dropoff", "dropoff"):
        method = getattr(DispatchToolEnv, name)
        tools_schema.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (method.__doc__ or "").strip(),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    user_content = "Begin the shift. " + initial  # mirrors what TRL does after reset
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=tools_schema,
        add_generation_prompt=True,
        tokenize=False,
    )
    print(rendered)


if __name__ == "__main__":
    main()
