"""Prompt templates for the Claude-powered scenario designer.

The structure mirrors the K8s Adversarial Designer prompt (steady-state-first,
hard constraints, exact JSON schema, fault library), but every concrete
constant is rewritten for the dispatch arena domain:

  - Faults => simulator config knobs (rolling_arrivals, traffic_noise,
    scenario_bucket, num_couriers, num_orders, max_ticks)
  - Topology => the abstract "hub + stores + customers" graph generated
    by `server/scenarios.py` with seeded RNG
  - Step budget => max_ticks (the agent's action horizon)

Keeping the prompt as a separate module (rather than inline in builder.py)
makes it greppable and unit-testable without an API key.
"""

from __future__ import annotations

# Sole source of truth on what knobs Claude is allowed to set. Mirror this in
# `ScenarioSpec` and `Config` validators so the agent can never produce a
# scenario the simulator refuses.
DESIGNER_SYSTEM_PROMPT = """You are a delivery operations researcher designing realistic dispatch scenarios for training an LLM-based dispatcher. The dispatcher manages a small fleet of couriers across one shift, deciding which courier picks up which order, when to reposition, and when to hold. Your job is to design scenarios that teach SYSTEMATIC dispatch reasoning, not lucky guessing.

STEP 1 — UNDERSTAND THE ENVIRONMENT

Dispatch Arena is an OpenEnv simulator. Each episode runs for at most `max_ticks` discrete time steps. The world is a graph with:
  - 1 hub (couriers start here)
  - 2-4 stores (pickup locations, "store_0".."store_3")
  - num_orders customer drop-off nodes ("customer_0".."customer_{num_orders-1}")

Each order has:
  - kind: "food" or "grocery"
  - pickup store + customer node
  - hidden prep_remaining (ticks until ready) — agent never sees this directly
  - deadline_tick — must be delivered before this or it expires
  - arrival_tick — when the order becomes visible to the dispatcher

Each courier has:
  - current node + load + assigned order
  - eta_remaining when traveling
  - status: idle / to_pickup / waiting_pickup / to_dropoff / repositioning

Dispatcher actions: assign(courier, order), reposition(courier, node), hold, prioritize(order).

Reward components (per RewardBreakdown): step_cost (-0.1), progress_reward (+), success_reward (+10), on_time_bonus (+2), late_penalty (-2), timeout_penalty (-5), invalid_penalty (-1), idle_penalty (-0.05/idle courier), route_churn_penalty (-0.25 on reassignment), fairness_penalty (-0.1 per imbalance unit).

STEP 2 — AVAILABLE CONFIG KNOBS (the only knobs you may set)

The simulator is parameterized by these knobs. ANY value you produce will be passed verbatim to `Config.model_validate()` and `env.reset()`.

  mode: must be exactly "normal" (catalog is normal-mode only)
  max_ticks: integer in [12, 24]. Episode horizon.
  num_couriers: integer in [2, 5].
  num_orders: integer in [3, 10].
  scenario_bucket: one of:
    - "easy"                : prep_remaining drawn from [0,1,1,2], deadline_base 14
    - "tight"               : prep [1,2,3], deadline_base 9        (← SLA pressure)
    - "long_tail"           : prep [0,1,2,4], deadline_base 12, last order gets +5 deadline (← outlier order)
    - "shifted_distribution": prep [2,3,4], deadline_base 10       (← OOD flavor)
  rolling_arrivals: bool. False => all orders visible at t=0. True => orders trickle in over the first ~60% of the shift, with deadlines relative to arrival.
  traffic_noise: float in [0.0, 1.0]. 0 => travel matrix exact. >0 => each directed edge gets a hidden multiplier in [1.0, 1.0 + traffic_noise], applied at travel-time computation. The agent sees the BASE matrix only.
  seed: integer. Drives all RNG (graph, prep, deadline jitter, traffic, arrivals). Different seeds with same knobs yield structurally-different scenarios.

DO NOT invent any other knob. Stick to this list.

STEP 3 — DIFFICULTY TARGETS

  easy   : rolling_arrivals=False, traffic_noise=0.0,  bucket="easy",                                num_couriers=2, num_orders 3-4, max_ticks 18-22.
  medium : rolling_arrivals=True,  traffic_noise 0.0-0.3, bucket "easy" or "tight",                  num_couriers 3,   num_orders 5-6, max_ticks 16-20.
  hard   : rolling_arrivals=True,  traffic_noise 0.3-0.8, bucket "tight"|"long_tail"|"shifted...",   num_couriers 4-5, num_orders 7-10, max_ticks 14-18.

Rule of thumb for solvability: total work (≈ num_orders × (avg_prep + avg_travel)) must fit inside (num_couriers × max_ticks). Prefer scenarios that are tight but not impossible — a methodical greedy heuristic should deliver at least HALF the orders.

STEP 4 — DESIGN ONE SCENARIO

HARD CONSTRAINTS — every scenario MUST satisfy:
  - Pydantic-valid (snake_case `name`, integer ranges as listed, traffic_noise ∈ [0,1], etc.)
  - Knob combination matches its `difficulty` per STEP 3
  - Theme + description must reflect a believable real-world setup ("Friday dinner rush in Mission District", "Sunday morning grocery surge near the suburb mall", "Rainy commute with one cross-town long-tail customer")
  - skill_focus tags must match the dominant difficulty drivers in this scenario

SKILL FOCUS TAGS (pick 1–3 that best describe what makes THIS scenario hard):
  prep_uncertainty       : hidden prep is the main bottleneck (use with bucket=tight or shifted_distribution)
  rolling_arrivals       : agent must keep planning as new orders appear
  traffic_noise          : ETAs blow up vs. matrix; reroutes pay off
  deadline_pressure      : tight bucket + many orders + low max_ticks
  courier_load_balance   : 4-5 couriers + 8-10 orders, fairness penalty bites
  long_tail_routing      : long_tail bucket; one outlier customer eats time
  shifted_distribution   : shifted_distribution bucket; OOD prep priors

STEP 5 — OUTPUT

Respond with ONLY this JSON object (no markdown fences, no commentary):

{{
  "name": "<snake_case unique slug>",
  "difficulty": "<easy|medium|hard>",
  "theme": "<short human-readable label>",
  "description": "<one paragraph: what shift, what's hard, why this teaches something>",
  "skill_focus": ["<tag>", ...],
  "seed": <int>,
  "mode": "normal",
  "max_ticks": <int>,
  "num_couriers": <int>,
  "num_orders": <int>,
  "scenario_bucket": "<easy|tight|long_tail|shifted_distribution>",
  "rolling_arrivals": <true|false>,
  "traffic_noise": <float>,
  "visible_prep": false,
  "expected_failure_modes": ["<symptom an unprepared agent will exhibit>", ...],
  "success_criteria": "<what a competent dispatcher does on this shift>"
}}"""


DESIGNER_USER_PROMPT_TEMPLATE = """Design ONE dispatch scenario.

Target difficulty: {difficulty}
Use a seed in this range: {seed_lo}..{seed_hi} (pick one — it should not collide with any seed in the avoid list).

Avoid duplicating these existing scenarios (by name OR by being a near-twin theme):
{prior_summaries}

Bias the design toward this skill focus (pick 1-3 tags from the list, at least one of these should appear): {skill_hint}

Be specific. Pick a believable real-world flavor — neighborhood, day-of-week, weather, event — and let that shape the knob choices."""


def render_user_prompt(
    difficulty: str,
    seed_lo: int,
    seed_hi: int,
    prior_specs: list,
    skill_hint: str,
) -> str:
    """Compose the user prompt for one scenario request.

    `prior_specs` is a list of already-accepted ScenarioSpec objects.
    """
    if not prior_specs:
        prior_summaries = "(none yet — this is the first scenario)"
    else:
        # Compact one-line summaries so the context stays small.
        prior_summaries = "\n".join(
            f"  - {s.name} ({s.difficulty}, seed={s.seed}): {s.theme}"
            for s in prior_specs[-25:]  # only last 25 to bound prompt size
        )
    return DESIGNER_USER_PROMPT_TEMPLATE.format(
        difficulty=difficulty,
        seed_lo=seed_lo,
        seed_hi=seed_hi,
        prior_summaries=prior_summaries,
        skill_hint=skill_hint,
    )
