"""Post-training evaluation against the held-out eval split.

Three modes:
  --baseline   greedy heuristic (handcoded, no model)
  --trained    base LLM + LoRA adapter, full tool-calling rollout per scenario
  --untrained  base LLM (no adapter) — measures whether GRPO actually improved over zero-shot

Aggregates per-difficulty and per-skill_focus, writes JSON. Same `master_seed`
as the trainer so the eval split is identical to training holdout.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dispatch_arena.catalog.builder import load_catalog
from dispatch_arena.catalog.dataset import stratified_split
from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.server.app import run_local_server_in_thread

CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "catalog.json"
ADAPTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "_grpo_normal_out" / "final_lora"
BASE_MODEL = "Qwen/Qwen3-1.7B"


def heuristic_action(obs) -> Dict[str, Any]:
    """Greedy baseline: assign first idle courier to first unassigned order."""
    state = obs.state
    courier = next(
        (c for c in state.couriers if c.status.value == "idle" and c.load is None),
        None,
    )
    order = next(
        (
            o
            for o in state.orders
            if o.status.value in {"queued", "ready"} and o.assigned_courier_id is None
        ),
        None,
    )
    if courier and order:
        return {"action_type": "assign", "courier_id": courier.id, "order_id": order.id}
    return {"action_type": "hold"}


def evaluate_heuristic(eval_specs, server_url: str) -> List[Dict[str, Any]]:
    """Run the greedy heuristic against every eval scenario and collect metrics.

    Used as the baseline curve in the README. The trained-model eval lives in
    a separate function once we have a checkpoint to load.
    """
    client = DispatchArenaClient(base_url=server_url, timeout_seconds=30)
    results = []
    for spec in eval_specs:
        config = {
            "mode": "normal",
            "max_ticks": spec.max_ticks,
            "num_couriers": spec.num_couriers,
            "num_orders": spec.num_orders,
            "scenario_bucket": spec.scenario_bucket,
            "rolling_arrivals": spec.rolling_arrivals,
            "traffic_noise": spec.traffic_noise,
            "visible_prep": spec.visible_prep,
        }
        obs = client.reset(seed=spec.seed, config=config)
        total_reward = float(obs.reward)
        invalid = 0
        cap = spec.max_ticks * 3
        steps = 0
        while not obs.done and steps < cap:
            steps += 1
            action = heuristic_action(obs)
            obs = client.step(action)
            total_reward += float(obs.reward)
            if obs.info.get("invalid_action"):
                invalid += 1
        summary = client.fetch_summary()
        metrics = summary.get("metrics", {})
        results.append(
            {
                "name": spec.name,
                "difficulty": spec.difficulty,
                "skill_focus": list(spec.skill_focus),
                "total_reward": total_reward,
                "delivered": metrics.get("delivered", 0),
                "orders": metrics.get("orders", spec.num_orders),
                "success_rate": metrics.get("success_rate", 0.0),
                "on_time_rate": metrics.get("on_time_rate", 0.0),
                "expired_rate": metrics.get("expired_rate", 0.0),
                "mean_delivery_ticks": metrics.get("mean_delivery_ticks", 0.0),
                "mean_lateness": metrics.get("mean_lateness", 0.0),
                "invalid_rate": metrics.get("invalid_rate", 0.0),
                "invalid_count": invalid,
                "verdict": summary.get("final_verdict", "unknown"),
                "ticks_taken": summary.get("ticks_taken", 0),
            }
        )
    return results


def aggregate(results: List[Dict[str, Any]], group_key: str) -> Dict[str, Dict[str, float]]:
    """Aggregate per-scenario results into per-bucket means."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if group_key == "skill_focus":
        for r in results:
            for tag in r["skill_focus"]:
                buckets[tag].append(r)
    else:
        for r in results:
            buckets[str(r[group_key])].append(r)

    agg = {}
    metric_keys = [
        "total_reward",
        "success_rate",
        "on_time_rate",
        "expired_rate",
        "mean_delivery_ticks",
        "mean_lateness",
        "invalid_rate",
    ]
    for key, group in buckets.items():
        n = len(group)
        agg[key] = {"n": n}
        for mk in metric_keys:
            agg[key][mk] = sum(r[mk] for r in group) / n if n else 0.0
    return agg


# ───────────────────────────────────────────────────────────────────────
# Trained-model evaluation: load base + LoRA, run a tool-calling loop per
# scenario, collect rewards.
# ───────────────────────────────────────────────────────────────────────


# Same prompt the trainer used; tool schemas are embedded so the model can
# emit valid <tool_call>{json}</tool_call> blocks even if the chat template's
# tools= injection isn't picked up.
INFERENCE_SYSTEM_PROMPT = """You are a real-time delivery dispatcher running one shift over a small fleet of couriers. Your job is to dispatch each order to the right courier and keep the shift moving so orders are delivered before their deadlines.

# Tool calling

Always reply with EXACTLY ONE tool call per turn, in this format (no other text):
<tool_call>
{"name": "<tool_name>", "arguments": {<args-json>}}
</tool_call>

# Available tools

  view_dashboard()                        — refresh the dashboard
  assign(courier_id, order_id)            — dispatch an idle courier to a queued/ready unassigned order
  reposition(courier_id, node_id)         — pre-stage an idle courier at a node
  hold()                                  — wait one tick
  prioritize(order_id)                    — mark an order as priority
  finish_shift()                          — end early once everything is delivered

# Rules
- One tool per turn, in the exact <tool_call> format above. Output nothing else.
- Prep time is hidden; orders flip from queued -> ready when ready.
- Travel times shown are baseline; with traffic, real ETAs run longer."""


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_call(completion: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract (name, arguments) from a Qwen-style <tool_call>{json}</tool_call> block."""
    m = _TOOL_CALL_RE.search(completion)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    args = data.get("arguments") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args


def _tool_call_to_action(tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map (tool_name, args) -> simulator Action dict. Returns None if invalid."""
    if tool_name == "view_dashboard":
        return {"action_type": "hold"}  # dashboard refresh costs one tick
    if tool_name == "hold":
        return {"action_type": "hold"}
    if tool_name == "assign":
        c, o = args.get("courier_id"), args.get("order_id")
        if not c or not o:
            return None
        return {"action_type": "assign", "courier_id": c, "order_id": o}
    if tool_name == "reposition":
        c, n = args.get("courier_id"), args.get("node_id")
        if not c or not n:
            return None
        return {"action_type": "reposition", "courier_id": c, "node_id": n}
    if tool_name == "prioritize":
        o = args.get("order_id")
        if not o:
            return None
        return {"action_type": "prioritize", "order_id": o}
    if tool_name == "finish_shift":
        return {"action_type": "hold"}  # treated as exit-with-hold
    return None  # unknown tool


def _render_dashboard(obs) -> str:
    """Compact text dashboard fed back to the model after each tool call."""
    s = obs.state
    lines = [f"tick={s.tick}/{s.max_ticks} verdict={obs.verifier_status.value} backlog={s.backlog} sla_pressure={s.sla_pressure:.2f}"]
    lines.append("couriers:")
    for c in s.couriers:
        load = c.load or "none"
        target = f" -> {c.target_node_id}(eta {c.eta_remaining})" if c.target_node_id else ""
        lines.append(f"  {c.id} @ {c.node_id} {c.status.value}{target} carrying={load}")
    lines.append("orders:")
    for o in s.orders:
        a = o.assigned_courier_id or "-"
        lines.append(f"  {o.id} {o.kind} {o.pickup_node_id}->{o.dropoff_node_id} status={o.status.value} deadline=t{o.deadline_tick} assigned={a}")
    lines.append("travel_times (base, may run longer with traffic):")
    for src in [n.id for n in s.nodes]:
        row = s.travel_time_matrix.get(src, {})
        edges = ", ".join(f"{dst}={t}" for dst, t in row.items() if dst != src)
        lines.append(f"  {src}: {edges}")
    if obs.info.get("events"):
        lines.append("last_events: " + " | ".join(obs.info["events"][-4:]))
    if obs.done:
        lines.append("DONE")
    return "\n".join(lines)


def load_inference_model(use_adapter: bool):
    """Load Qwen3-1.7B base model + (optionally) the trained LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {BASE_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda"
    )
    if use_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter {ADAPTER_PATH} ...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    return tok, model


def evaluate_with_model(
    eval_specs,
    server_url: str,
    tok,
    model,
    max_tool_calls: int = 20,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Run a tool-calling rollout for every eval scenario; collect metrics."""
    import torch

    client = DispatchArenaClient(base_url=server_url, timeout_seconds=30)
    results = []
    for i, spec in enumerate(eval_specs):
        config = {
            "mode": "normal",
            "max_ticks": spec.max_ticks,
            "num_couriers": spec.num_couriers,
            "num_orders": spec.num_orders,
            "scenario_bucket": spec.scenario_bucket,
            "rolling_arrivals": spec.rolling_arrivals,
            "traffic_noise": spec.traffic_noise,
            "visible_prep": spec.visible_prep,
        }
        obs = client.reset(seed=spec.seed, config=config)
        total_reward = float(obs.reward)
        invalid = 0
        unparseable = 0
        valid_calls = 0

        messages = [
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": "Initial dashboard:\n" + _render_dashboard(obs)},
        ]
        for _step in range(max_tool_calls):
            if obs.done:
                break
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tok.eos_token_id,
                )
            completion = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": completion})

            parsed = _parse_tool_call(completion)
            if not parsed:
                unparseable += 1
                messages.append({"role": "user", "content": "Output not parseable. Reply with one <tool_call>{...}</tool_call> block."})
                continue
            tool_name, args = parsed
            action = _tool_call_to_action(tool_name, args)
            if action is None:
                unparseable += 1
                messages.append({"role": "user", "content": f"Unknown or malformed tool call: {tool_name}. Try a valid tool."})
                continue
            obs = client.step(action)
            total_reward += float(obs.reward)
            if obs.info.get("invalid_action"):
                invalid += 1
            else:
                valid_calls += 1
            messages.append({"role": "user", "content": _render_dashboard(obs)})

        summary = client.fetch_summary()
        metrics = summary.get("metrics", {})
        results.append({
            "name": spec.name,
            "difficulty": spec.difficulty,
            "skill_focus": list(spec.skill_focus),
            "total_reward": total_reward,
            "delivered": metrics.get("delivered", 0),
            "orders": metrics.get("orders", spec.num_orders),
            "success_rate": metrics.get("success_rate", 0.0),
            "on_time_rate": metrics.get("on_time_rate", 0.0),
            "expired_rate": metrics.get("expired_rate", 0.0),
            "mean_delivery_ticks": metrics.get("mean_delivery_ticks", 0.0),
            "mean_lateness": metrics.get("mean_lateness", 0.0),
            "invalid_rate": metrics.get("invalid_rate", 0.0),
            "invalid_count": invalid,
            "unparseable_count": unparseable,
            "valid_call_count": valid_calls,
            "verdict": summary.get("final_verdict", "unknown"),
            "ticks_taken": summary.get("ticks_taken", 0),
        })
        print(f"  [{i+1:>2}/{len(eval_specs)}] {spec.difficulty:>6} {spec.name[:40]:<40} "
              f"-> delivered={metrics.get('delivered',0)}/{metrics.get('orders',spec.num_orders)} "
              f"reward={total_reward:+.2f} invalid={invalid} unparseable={unparseable}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-difficulty eval over the catalog held-out split.")
    p.add_argument("--baseline",  action="store_true", help="Greedy heuristic (no model).")
    p.add_argument("--trained",   action="store_true", help="Base LLM + trained LoRA adapter.")
    p.add_argument("--untrained", action="store_true", help="Base LLM only (zero-shot, no adapter).")
    p.add_argument("--master-seed", type=int, default=0)
    p.add_argument("--eval-fraction", type=float, default=0.30)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to _eval_out/{baseline|trained|untrained}.json",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.baseline or args.trained or args.untrained):
        print("Pick one of --baseline / --trained / --untrained.")
        return 1

    mode_label = "heuristic_baseline" if args.baseline else ("trained_lora" if args.trained else "untrained_base")
    if args.out is None:
        args.out = Path(__file__).resolve().parents[1] / "scripts" / "_eval_out" / f"{mode_label}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    server, _thread = run_local_server_in_thread(port=0, max_concurrent_envs=16)
    host, port = server.server_address
    time.sleep(0.2)
    server_url = f"http://{host}:{port}"

    specs = load_catalog(CATALOG_PATH)
    _train_specs, eval_specs = stratified_split(
        specs, eval_fraction=args.eval_fraction, master_seed=args.master_seed
    )
    print(f"Eval split: {len(eval_specs)} scenarios  (mode={mode_label})")

    if args.baseline:
        print("Running greedy heuristic baseline over eval split...")
        results = evaluate_heuristic(eval_specs, server_url)
    else:
        tok, model = load_inference_model(use_adapter=args.trained)
        print(f"Running model eval over {len(eval_specs)} scenarios (this takes ~10-20 min)...")
        results = evaluate_with_model(eval_specs, server_url, tok, model)

    agg_diff = aggregate(results, "difficulty")
    agg_skill = aggregate(results, "skill_focus")
    payload = {
        "mode": mode_label,
        "n_scenarios": len(results),
        "by_difficulty": agg_diff,
        "by_skill_focus": agg_skill,
        "per_scenario": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== {mode_label.upper()} ({len(results)} scenarios) ===")
    for diff in ["easy", "medium", "hard"]:
        if diff in agg_diff:
            m = agg_diff[diff]
            print(
                f"  {diff:>6}  n={m['n']:>2}  "
                f"reward={m['total_reward']:>+6.2f}  "
                f"success={m['success_rate']:.2f}  "
                f"on_time={m['on_time_rate']:.2f}  "
                f"invalid_rate={m['invalid_rate']:.3f}"
            )
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
