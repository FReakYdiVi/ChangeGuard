"""Hand-curated anchor themes for offline (no-API) catalog generation.

Each anchor describes a narrative archetype plus knob ranges. The
`AnchorScenarioBuilder` samples N variations per anchor (different seed,
jittered knobs within the declared range) so the catalog has ~12-14 themes
with multiple seeds each. This is the deterministic fallback when
ANTHROPIC_API_KEY is unavailable — the Claude builder is preferred when it
is, since the LLM produces richer descriptions.

Knob ranges respect the difficulty contract from prompts.py:
  easy   : rolling=False, traffic=0.0,    bucket=easy
  medium : rolling=True,  traffic 0-0.3,  bucket easy or tight
  hard   : rolling=True,  traffic 0.3-0.8, bucket tight|long_tail|shifted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from dispatch_arena.catalog.spec import Difficulty, SkillTag


@dataclass(frozen=True)
class Anchor:
    slug: str  # used to build scenario names: f"{slug}_seed{seed}"
    difficulty: Difficulty
    theme: str
    description: str
    skill_focus: List[SkillTag]
    expected_failure_modes: List[str]
    success_criteria: str

    # Knob ranges. AnchorScenarioBuilder will sample within these.
    max_ticks_range: Tuple[int, int]
    num_couriers_range: Tuple[int, int]
    num_orders_range: Tuple[int, int]
    scenario_buckets: List[str]  # one will be picked per variation
    rolling_arrivals: bool
    traffic_noise_range: Tuple[float, float]


# ---- EASY tier (4 anchors) -------------------------------------------------
EASY_ANCHORS: List[Anchor] = [
    Anchor(
        slug="weekday_lunch_simple",
        difficulty="easy",
        theme="Weekday lunch in a single neighborhood",
        description=(
            "A handful of food orders during a quiet weekday lunch. Restaurants are "
            "responsive, deadlines are loose, no traffic surprises. The dispatcher "
            "should learn the basic assign-then-deliver loop here."
        ),
        skill_focus=["prep_uncertainty"],
        expected_failure_modes=[
            "Dispatcher idles couriers instead of pre-positioning at the store",
            "Repeated invalid assigns when the order is already taken",
        ],
        success_criteria="Deliver every order on time with zero invalid actions.",
        max_ticks_range=(18, 22),
        num_couriers_range=(2, 2),
        num_orders_range=(3, 4),
        scenario_buckets=["easy"],
        rolling_arrivals=False,
        traffic_noise_range=(0.0, 0.0),
    ),
    Anchor(
        slug="campus_quiet_afternoon",
        difficulty="easy",
        theme="Campus afternoon, light load",
        description=(
            "Late-afternoon orders around a small campus. Two couriers, three orders, "
            "everyone visible from t=0. The teaching value here is courier load balance: "
            "one courier should not take all three orders sequentially."
        ),
        skill_focus=["courier_load_balance"],
        expected_failure_modes=[
            "Single courier hoards all assignments while the other idles",
            "Fairness penalty accrues unnoticed",
        ],
        success_criteria="Both couriers deliver at least one order; reward exceeds 8.",
        max_ticks_range=(18, 20),
        num_couriers_range=(2, 2),
        num_orders_range=(3, 3),
        scenario_buckets=["easy"],
        rolling_arrivals=False,
        traffic_noise_range=(0.0, 0.0),
    ),
    Anchor(
        slug="suburban_steady_state",
        difficulty="easy",
        theme="Suburban steady state",
        description=(
            "Suburban afternoon with light, predictable load. Mostly grocery orders. "
            "Tests the agent's ability to read a clean dashboard and act decisively "
            "without overthinking."
        ),
        skill_focus=["prep_uncertainty"],
        expected_failure_modes=[
            "Excessive 'hold' actions while waiting for prep to complete",
        ],
        success_criteria="All orders delivered within 16 ticks.",
        max_ticks_range=(20, 22),
        num_couriers_range=(2, 2),
        num_orders_range=(3, 4),
        scenario_buckets=["easy"],
        rolling_arrivals=False,
        traffic_noise_range=(0.0, 0.0),
    ),
    Anchor(
        slug="office_park_predictable",
        difficulty="easy",
        theme="Office-park predictable cluster",
        description=(
            "A tight cluster of customers near one store. Travel is short, prep is "
            "the only meaningful uncertainty. Useful as the early curriculum rung "
            "before introducing rolling arrivals or traffic."
        ),
        skill_focus=["prep_uncertainty", "courier_load_balance"],
        expected_failure_modes=[
            "Courier sent to pickup before order is ready, then idles",
        ],
        success_criteria="100% delivery, on-time bonus on every order.",
        max_ticks_range=(18, 20),
        num_couriers_range=(2, 2),
        num_orders_range=(3, 4),
        scenario_buckets=["easy"],
        rolling_arrivals=False,
        traffic_noise_range=(0.0, 0.0),
    ),
]

# ---- MEDIUM tier (4 anchors) -----------------------------------------------
MEDIUM_ANCHORS: List[Anchor] = [
    Anchor(
        slug="downtown_dinner_kickoff",
        difficulty="medium",
        theme="Downtown dinner kickoff with rolling arrivals",
        description=(
            "Dinner service is starting and orders trickle in over the first half of "
            "the shift. Light traffic noise. Tests whether the agent can keep "
            "planning as new orders appear instead of locking in a route at t=0."
        ),
        skill_focus=["rolling_arrivals", "prep_uncertainty"],
        expected_failure_modes=[
            "Agent assigns all current orders early and idles when new ones arrive",
            "Late deliveries on orders that arrived mid-shift",
        ],
        success_criteria="At least 80% on-time deliveries despite rolling arrivals.",
        max_ticks_range=(16, 20),
        num_couriers_range=(3, 3),
        num_orders_range=(5, 6),
        scenario_buckets=["easy", "tight"],
        rolling_arrivals=True,
        traffic_noise_range=(0.0, 0.2),
    ),
    Anchor(
        slug="weekend_grocery_morning",
        difficulty="medium",
        theme="Weekend grocery morning surge",
        description=(
            "Saturday morning grocery surge: orders trickle in over the shift, "
            "deadlines are loose-ish but prep takes longer than expected because "
            "stores are batching. Light traffic only on a couple of edges."
        ),
        skill_focus=["rolling_arrivals", "prep_uncertainty"],
        expected_failure_modes=[
            "Assigns courier to a queued order whose prep timer is still high",
            "Underuses 'reposition' to pre-stage couriers near busy stores",
        ],
        success_criteria="At least 4 of 5 orders delivered, fairness within 1.",
        max_ticks_range=(18, 20),
        num_couriers_range=(3, 3),
        num_orders_range=(5, 6),
        scenario_buckets=["easy"],
        rolling_arrivals=True,
        traffic_noise_range=(0.0, 0.15),
    ),
    Anchor(
        slug="friday_evening_pickup",
        difficulty="medium",
        theme="Friday evening tight prep",
        description=(
            "Friday evening: stores are slow, deadlines are tight, but traffic is "
            "manageable. Orders arrive in two waves. Tests prep-aware scheduling: "
            "when to push 'hold' vs. when to send a courier early."
        ),
        skill_focus=["prep_uncertainty", "deadline_pressure", "rolling_arrivals"],
        expected_failure_modes=[
            "Deadlines slip because the agent waits too long to dispatch",
            "Idle penalty accrues from over-cautious holds",
        ],
        success_criteria="Total reward > 6; at most 1 expired order.",
        max_ticks_range=(16, 18),
        num_couriers_range=(3, 3),
        num_orders_range=(5, 6),
        scenario_buckets=["tight"],
        rolling_arrivals=True,
        traffic_noise_range=(0.0, 0.2),
    ),
    Anchor(
        slug="brunch_clustered_arrivals",
        difficulty="medium",
        theme="Sunday brunch with bursty arrivals",
        description=(
            "Sunday brunch: orders cluster around two narrow time windows rather "
            "than spreading evenly. Mostly easy travel, mild traffic. Trains the "
            "agent to handle bursts without panic-assigning everything at once."
        ),
        skill_focus=["rolling_arrivals", "courier_load_balance"],
        expected_failure_modes=[
            "Burst-assigns 4 orders to one courier while others idle",
            "Route churn from late re-balancing",
        ],
        success_criteria="Fairness imbalance ≤ 1; route churn penalty ≤ 0.5 total.",
        max_ticks_range=(18, 20),
        num_couriers_range=(3, 3),
        num_orders_range=(5, 6),
        scenario_buckets=["easy"],
        rolling_arrivals=True,
        traffic_noise_range=(0.0, 0.25),
    ),
]

# ---- HARD tier (6 anchors) -------------------------------------------------
HARD_ANCHORS: List[Anchor] = [
    Anchor(
        slug="rainy_rush_hour",
        difficulty="hard",
        theme="Rainy rush hour, tight SLAs",
        description=(
            "Rainy rush hour. Deadlines are tight, traffic is unpredictable across "
            "every cross-town edge, and orders keep arriving. The dispatcher must "
            "absorb traffic blow-ups without spiraling into reroute churn."
        ),
        skill_focus=["traffic_noise", "deadline_pressure", "rolling_arrivals"],
        expected_failure_modes=[
            "Repeated reposition+reassign cycles as ETAs slip",
            "Cascading expirations once one courier gets stuck",
        ],
        success_criteria="At least 60% on-time despite traffic; route churn ≤ 1.0.",
        max_ticks_range=(14, 16),
        num_couriers_range=(4, 4),
        num_orders_range=(7, 8),
        scenario_buckets=["tight"],
        rolling_arrivals=True,
        traffic_noise_range=(0.4, 0.7),
    ),
    Anchor(
        slug="marathon_route_diversion",
        difficulty="hard",
        theme="Marathon-day cross-town diversion",
        description=(
            "A road closure scrambles travel times on most cross-town edges, and "
            "one customer is far from every store (long-tail). The agent must "
            "decide whether to dedicate a courier to that outlier or skip it."
        ),
        skill_focus=["traffic_noise", "long_tail_routing", "rolling_arrivals"],
        expected_failure_modes=[
            "Wastes the only fast courier on the long-tail customer",
            "Long-tail order expires while agent over-optimizes the cluster",
        ],
        success_criteria="Long-tail order delivered OR explicitly deprioritized; cluster orders all delivered.",
        max_ticks_range=(15, 17),
        num_couriers_range=(4, 4),
        num_orders_range=(7, 8),
        scenario_buckets=["long_tail"],
        rolling_arrivals=True,
        traffic_noise_range=(0.4, 0.6),
    ),
    Anchor(
        slug="stadium_post_game_spike",
        difficulty="hard",
        theme="Stadium post-game spike",
        description=(
            "Heavy load: 9-10 orders, 5 couriers, shifted prep distribution (everything "
            "takes 2-4 ticks). Traffic is moderate. Trains the agent to coordinate "
            "5 couriers without piling assignments on the closest one."
        ),
        skill_focus=["courier_load_balance", "shifted_distribution", "rolling_arrivals"],
        expected_failure_modes=[
            "Two couriers get 4 orders each while two idle",
            "Fairness penalty dominates the reward",
        ],
        success_criteria="Each courier delivers ≥ 1 order; fairness imbalance ≤ 2.",
        max_ticks_range=(16, 18),
        num_couriers_range=(5, 5),
        num_orders_range=(9, 10),
        scenario_buckets=["shifted_distribution"],
        rolling_arrivals=True,
        traffic_noise_range=(0.2, 0.4),
    ),
    Anchor(
        slug="holiday_eve_chaos",
        difficulty="hard",
        theme="Holiday eve chaos: long-tail + heavy traffic",
        description=(
            "Holiday eve everything-at-once: long-tail customer mix, heavy traffic, "
            "rolling arrivals across the whole shift, max courier count. The "
            "hardest anchor — only consistent strong dispatchers should solve it."
        ),
        skill_focus=["traffic_noise", "long_tail_routing", "rolling_arrivals", "courier_load_balance"],
        expected_failure_modes=[
            "Multiple expired orders due to late assignment under traffic shocks",
            "Route churn explodes as the agent reassigns under uncertainty",
        ],
        success_criteria="At least 50% delivered; non-trivial reward; no more than 2 expired.",
        max_ticks_range=(15, 17),
        num_couriers_range=(5, 5),
        num_orders_range=(8, 10),
        scenario_buckets=["long_tail"],
        rolling_arrivals=True,
        traffic_noise_range=(0.5, 0.8),
    ),
    Anchor(
        slug="fog_morning_delays",
        difficulty="hard",
        theme="Foggy morning, shifted prep priors",
        description=(
            "Foggy morning: prep distributions are shifted (everything starts at 2-4 "
            "ticks), traffic is moderate, orders arrive over the shift. OOD-flavor "
            "scenario — the agent's prior on prep is wrong and must adapt."
        ),
        skill_focus=["shifted_distribution", "prep_uncertainty", "traffic_noise"],
        expected_failure_modes=[
            "Sends couriers to pickup too early; they idle waiting for prep",
            "Misses deadlines because prep is longer than expected",
        ],
        success_criteria="At least 60% delivered on time; idle penalty bounded.",
        max_ticks_range=(15, 17),
        num_couriers_range=(4, 4),
        num_orders_range=(7, 8),
        scenario_buckets=["shifted_distribution"],
        rolling_arrivals=True,
        traffic_noise_range=(0.3, 0.5),
    ),
    Anchor(
        slug="blackout_recovery",
        difficulty="hard",
        theme="Post-blackout recovery, tight everything",
        description=(
            "Recovering from an outage: tight deadlines, traffic is heavy because "
            "signals are out, orders backlogged so they arrive in two big waves. "
            "Tests whether the dispatcher can prioritize under cascade pressure."
        ),
        skill_focus=["deadline_pressure", "traffic_noise", "rolling_arrivals"],
        expected_failure_modes=[
            "Tries to deliver everything, ends up delivering nothing on time",
            "No 'prioritize' signal sent; dispatcher just reacts",
        ],
        success_criteria="At least 60% on-time; clear prioritization pattern in trace.",
        max_ticks_range=(14, 16),
        num_couriers_range=(4, 5),
        num_orders_range=(8, 9),
        scenario_buckets=["tight"],
        rolling_arrivals=True,
        traffic_noise_range=(0.5, 0.7),
    ),
]


ALL_ANCHORS: List[Anchor] = EASY_ANCHORS + MEDIUM_ANCHORS + HARD_ANCHORS


def anchors_by_difficulty(difficulty: Difficulty) -> List[Anchor]:
    return [a for a in ALL_ANCHORS if a.difficulty == difficulty]
