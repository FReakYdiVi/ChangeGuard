"""Catalog builders.

Two backends, same output schema:

  AnchorScenarioBuilder: deterministic, samples N variations per anchor with
                         seeded jitter on knobs. No external dependencies.

  LLMScenarioBuilder:    calls Claude via the anthropic SDK. Produces richer
                         descriptions and more diverse themes. Falls back to
                         the anchor builder if the SDK / API key is missing.

Every produced ScenarioSpec is round-trip validated against the simulator
before being added to the catalog: we actually call `env.reset()` with the
spec's Config + seed and ensure no exception. Specs that fail validation are
discarded and replaced.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Iterable, List, Optional, Sequence

from pydantic import ValidationError

from dispatch_arena.catalog.anchors import (
    ALL_ANCHORS,
    Anchor,
    anchors_by_difficulty,
)
from dispatch_arena.catalog.prompts import (
    DESIGNER_SYSTEM_PROMPT,
    render_user_prompt,
)
from dispatch_arena.catalog.spec import Difficulty, ScenarioSpec, SkillTag
from dispatch_arena.models import OrderStatus
from dispatch_arena.server.env import DispatchArenaEnvironment

logger = logging.getLogger(__name__)


# --- Shared validation -------------------------------------------------------


def round_trip_validate(spec: ScenarioSpec) -> None:
    """Make sure the simulator can actually run this scenario.

    Calls reset() with the spec's Config + seed and verifies the resulting
    state has at least one courier, at least one order (visible or pending),
    and the episode is not already done. Raises on failure.
    """
    env = DispatchArenaEnvironment(config=spec.to_config())
    obs = env.reset(seed=spec.seed)
    if obs.done:
        raise ValueError(f"scenario {spec.name!r} terminates immediately on reset")
    if not obs.state.couriers:
        raise ValueError(f"scenario {spec.name!r} has no couriers after reset")
    total_visible = len(obs.state.orders)
    total_pending = len(env._pending_arrivals)
    if total_visible + total_pending != spec.num_orders:
        raise ValueError(
            f"scenario {spec.name!r}: expected {spec.num_orders} orders, "
            f"got {total_visible} visible + {total_pending} pending"
        )


def heuristic_solvable(spec: ScenarioSpec, min_deliveries: int = 1) -> bool:
    """Lightweight smoke check: does a greedy heuristic deliver ≥ N orders?

    Filters out unsolvable scenarios where even the easy heuristic times out
    with zero deliveries — those would just inject noise into training.
    """
    env = DispatchArenaEnvironment(config=spec.to_config())
    obs = env.reset(seed=spec.seed)
    safety_cap = spec.max_ticks * 3
    steps = 0
    while not obs.done and steps < safety_cap:
        steps += 1
        # Greedy: assign the first idle courier to the first unassigned order.
        courier = next(
            (c for c in obs.state.couriers if c.status.value == "idle" and c.load is None),
            None,
        )
        order = next(
            (
                o
                for o in obs.state.orders
                if o.status.value in {"queued", "ready"} and o.assigned_courier_id is None
            ),
            None,
        )
        if courier and order:
            action = {"action_type": "assign", "courier_id": courier.id, "order_id": order.id}
        else:
            action = {"action_type": "hold"}
        obs = env.step(action)
    delivered = sum(1 for o in obs.state.orders if o.status == OrderStatus.DELIVERED)
    return delivered >= min_deliveries


# --- Anchor (deterministic) builder ------------------------------------------


class AnchorScenarioBuilder:
    """Sample N variations per anchor with seeded jitter on knobs.

    Determinism: same `master_seed` => same catalog, every time.
    """

    def __init__(self, master_seed: int = 0) -> None:
        self._rng = random.Random(master_seed)

    def build_batch(self, count_per_difficulty: dict) -> List[ScenarioSpec]:
        accepted: List[ScenarioSpec] = []
        used_names: set = set()
        used_seeds: set = set()
        for difficulty, count in count_per_difficulty.items():
            anchors = anchors_by_difficulty(difficulty)
            if not anchors:
                raise ValueError(f"no anchors for difficulty {difficulty!r}")
            attempts = 0
            target = count
            while sum(1 for s in accepted if s.difficulty == difficulty) < target:
                attempts += 1
                if attempts > target * 10:
                    raise RuntimeError(
                        f"too many failed attempts producing {difficulty} scenarios"
                    )
                anchor = self._rng.choice(anchors)
                spec = self._instantiate(anchor, used_names, used_seeds)
                if spec is None:
                    continue
                try:
                    round_trip_validate(spec)
                except (ValueError, ValidationError) as exc:
                    logger.warning("dropping %s: %s", spec.name, exc)
                    continue
                if not heuristic_solvable(spec):
                    logger.warning("dropping %s: heuristic delivered 0 orders", spec.name)
                    continue
                accepted.append(spec)
                used_names.add(spec.name)
                used_seeds.add(spec.seed)
        return accepted

    def _instantiate(
        self,
        anchor: Anchor,
        used_names: set,
        used_seeds: set,
    ) -> Optional[ScenarioSpec]:
        # Pick a unique seed in [1, 99_999] not used yet.
        for _ in range(64):
            seed = self._rng.randint(1, 99_999)
            if seed not in used_seeds:
                break
        else:
            return None

        name = f"{anchor.slug}_seed{seed}"
        if name in used_names:
            return None

        max_ticks = self._rng.randint(*anchor.max_ticks_range)
        num_couriers = self._rng.randint(*anchor.num_couriers_range)
        num_orders = self._rng.randint(*anchor.num_orders_range)
        scenario_bucket = self._rng.choice(anchor.scenario_buckets)
        traffic_lo, traffic_hi = anchor.traffic_noise_range
        traffic_noise = round(
            traffic_lo + self._rng.random() * (traffic_hi - traffic_lo), 2
        )

        try:
            spec = ScenarioSpec(
                name=name,
                difficulty=anchor.difficulty,
                theme=anchor.theme,
                description=anchor.description,
                skill_focus=list(anchor.skill_focus),
                seed=seed,
                mode="normal",
                max_ticks=max_ticks,
                num_couriers=num_couriers,
                num_orders=num_orders,
                scenario_bucket=scenario_bucket,
                rolling_arrivals=anchor.rolling_arrivals,
                traffic_noise=traffic_noise,
                visible_prep=False,
                expected_failure_modes=list(anchor.expected_failure_modes),
                success_criteria=anchor.success_criteria,
            )
        except ValidationError as exc:
            logger.warning("schema rejection for %s: %s", name, exc)
            return None
        return spec


# --- LLM builder (Claude) ----------------------------------------------------


class LLMScenarioBuilder:
    """Use Claude to design scenarios. Falls back to anchors on any failure.

    Uses the anthropic SDK if available + ANTHROPIC_API_KEY is set. The LLM
    output is validated against the same Pydantic schema and simulator
    round-trip as the anchor builder, so a misbehaving model can never inject
    invalid data into the catalog.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        master_seed: int = 0,
        max_retries_per_scenario: int = 3,
    ) -> None:
        self._rng = random.Random(master_seed)
        self._model = model
        self._max_retries = max_retries_per_scenario
        self._client = self._maybe_load_client()
        self._anchor_fallback = AnchorScenarioBuilder(master_seed=master_seed + 1)

    @staticmethod
    def _maybe_load_client():
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("ANTHROPIC_API_KEY not set; LLM builder will fall back to anchors")
            return None
        try:
            import anthropic  # type: ignore
        except ImportError:
            logger.info("anthropic SDK not installed; LLM builder will fall back to anchors")
            return None
        return anthropic.Anthropic()

    def build_batch(self, count_per_difficulty: dict) -> List[ScenarioSpec]:
        if self._client is None:
            logger.warning(
                "LLM builder unavailable, delegating entire build to anchor fallback"
            )
            return self._anchor_fallback.build_batch(count_per_difficulty)

        accepted: List[ScenarioSpec] = []
        for difficulty, count in count_per_difficulty.items():
            target = count
            while sum(1 for s in accepted if s.difficulty == difficulty) < target:
                spec = self._design_one(difficulty, accepted)
                if spec is None:
                    # Bail to anchor for this slot rather than infinite-loop.
                    fill = self._anchor_fallback.build_batch({difficulty: 1})
                    accepted.extend(fill)
                    continue
                accepted.append(spec)
        return accepted

    def _design_one(
        self,
        difficulty: Difficulty,
        prior: Sequence[ScenarioSpec],
    ) -> Optional[ScenarioSpec]:
        skill_hint = self._pick_skill_hint(difficulty)
        used_seeds = {s.seed for s in prior}
        for attempt in range(self._max_retries):
            seed_lo = attempt * 10_000 + 1
            seed_hi = seed_lo + 9_998
            user_prompt = render_user_prompt(
                difficulty=difficulty,
                seed_lo=seed_lo,
                seed_hi=seed_hi,
                prior_specs=list(prior),
                skill_hint=skill_hint,
            )
            try:
                raw = self._chat_json(DESIGNER_SYSTEM_PROMPT, user_prompt)
            except Exception as exc:  # API errors, JSON errors, anything
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                continue
            try:
                spec = ScenarioSpec.model_validate(raw)
            except ValidationError as exc:
                logger.warning("schema rejection: %s", exc)
                continue
            if spec.seed in used_seeds or any(s.name == spec.name for s in prior):
                logger.warning("duplicate seed/name from LLM, retrying")
                continue
            try:
                round_trip_validate(spec)
            except ValueError as exc:
                logger.warning("simulator rejection: %s", exc)
                continue
            if not heuristic_solvable(spec):
                logger.warning("dropping %s: heuristic delivers 0 orders", spec.name)
                continue
            return spec
        return None

    def _chat_json(self, system: str, user: str) -> dict:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=0.9,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        # Tolerate accidental markdown fences just in case.
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def _pick_skill_hint(self, difficulty: Difficulty) -> str:
        # Bias toward skill tags appropriate for the difficulty.
        easy = ["prep_uncertainty", "courier_load_balance"]
        medium = ["rolling_arrivals", "prep_uncertainty", "deadline_pressure", "courier_load_balance"]
        hard = [
            "traffic_noise",
            "rolling_arrivals",
            "long_tail_routing",
            "shifted_distribution",
            "deadline_pressure",
        ]
        pool = {"easy": easy, "medium": medium, "hard": hard}[difficulty]
        return self._rng.choice(pool)


# --- Catalog I/O -------------------------------------------------------------


def save_catalog(specs: Iterable[ScenarioSpec], path) -> None:
    payload = [s.model_dump(mode="json") for s in specs]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_catalog(path) -> List[ScenarioSpec]:
    with open(path) as f:
        payload = json.load(f)
    return [ScenarioSpec.model_validate(item) for item in payload]
