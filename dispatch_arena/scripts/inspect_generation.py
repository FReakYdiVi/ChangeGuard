"""Generate a few raw completions from the smoke trainer's exact prompt.

If the model emits ``<tool_call>{...}</tool_call>``, the trainer would have
driven the env. If it doesn't, training-time reward=0 is a model limitation,
not an env bug.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.scripts.train_grpo_smoke import (
    MODEL_NAME,
    SYSTEM_PROMPT,
    DispatchToolEnv,
)
from dispatch_arena.server.app import run_local_server_in_thread


def main() -> None:
    server, _t = run_local_server_in_thread(port=0, max_concurrent_envs=4)
    host, port = server.server_address
    time.sleep(0.2)
    client = DispatchArenaClient(base_url=f"http://{host}:{port}")

    env = DispatchToolEnv()
    env.client = client
    initial = env.reset(seed=7)

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

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
    model.eval()

    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Begin the shift. " + initial},
        ],
        tools=tools_schema,
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tok(rendered, return_tensors="pt").to("cuda")
    input_ids = enc.input_ids

    print("prompt tokens:", input_ids.shape[-1])
    for trial in range(3):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=192,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=False)
        print(f"\n--- TRIAL {trial} ---")
        print(completion)
        print("contains <tool_call>:", "<tool_call>" in completion)


if __name__ == "__main__":
    main()
