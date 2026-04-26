"""Seeded scenario generation for Dispatch Arena."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dispatch_arena.models import Config, Courier, Mode, Node, Order, OrderStatus


@dataclass(frozen=True)
class Scenario:
    config: Config
    nodes: List[Node]
    travel_time_matrix: Dict[str, Dict[str, int]]
    couriers: List[Courier]
    orders: List[Order]
    # Hidden multiplier per directed edge applied at travel-time computation.
    # Empty dict => no traffic noise. Never serialized into public state.
    traffic_multipliers: Dict[Tuple[str, str], float] = field(default_factory=dict)


def generate_scenario(config: Config, seed: int | None = None) -> Scenario:
    if config.mode == Mode.NORMAL:
        return _normal_scenario(config, seed)
    return _mini_scenario(config, seed)


def scenario_catalog() -> List[dict]:
    return [
        {"mode": "mini", "bucket": "easy", "description": "One courier, one order, short travel."},
        {"mode": "normal", "bucket": "easy", "description": "Loose deadlines and short routes."},
        {"mode": "normal", "bucket": "tight", "description": "Tighter SLAs and more pickup contention."},
        {"mode": "normal", "bucket": "long_tail", "description": "Some long routes dominate dispatch quality."},
        {"mode": "normal", "bucket": "shifted_distribution", "description": "Changed prep and deadline mix."},
    ]


def _mini_scenario(config: Config, seed: int | None) -> Scenario:
    rng = random.Random(seed)
    nodes = [
        Node(id="hub", kind="hub", label="Hub"),
        Node(id="pickup", kind="pickup", label="Pickup"),
        Node(id="dropoff", kind="dropoff", label="Dropoff"),
    ]
    travel = {
        "hub": {"hub": 0, "pickup": 1, "dropoff": 2},
        "pickup": {"hub": 1, "pickup": 0, "dropoff": 1},
        "dropoff": {"hub": 2, "pickup": 1, "dropoff": 0},
    }
    prep_remaining = rng.choice([1, 2, 3])
    order = Order(
        id="order_0",
        kind="food",
        pickup_node_id="pickup",
        dropoff_node_id="dropoff",
        created_tick=0,
        prep_remaining=prep_remaining,
        deadline_tick=min(config.max_ticks, 10),
        status=OrderStatus.READY if prep_remaining == 0 else OrderStatus.QUEUED,
    )
    return Scenario(
        config=config,
        nodes=nodes,
        travel_time_matrix=travel,
        couriers=[Courier(id="courier_0", node_id="hub")],
        orders=[order],
    )


def _normal_scenario(config: Config, seed: int | None) -> Scenario:
    rng = random.Random(seed)
    pickup_count = min(4, max(2, config.num_orders // 2))
    nodes = [Node(id="hub", kind="hub", label="Hub")]
    nodes.extend(Node(id=f"store_{i}", kind="pickup", label=f"Store {i}") for i in range(pickup_count))
    nodes.extend(Node(id=f"customer_{i}", kind="dropoff", label=f"Customer {i}") for i in range(config.num_orders))

    node_ids = [node.id for node in nodes]
    travel: Dict[str, Dict[str, int]] = {}
    for src in node_ids:
        travel[src] = {}
        for dst in node_ids:
            if src == dst:
                travel[src][dst] = 0
            elif src == "hub" or dst == "hub":
                travel[src][dst] = rng.randint(1, 3)
            else:
                travel[src][dst] = rng.randint(1, 4)

    deadline_base = {
        "easy": 14,
        "tight": 9,
        "long_tail": 12,
        "shifted_distribution": 10,
    }.get(config.scenario_bucket, 12)
    prep_choices = {
        "easy": [0, 1, 1, 2],
        "tight": [1, 2, 3],
        "long_tail": [0, 1, 2, 4],
        "shifted_distribution": [2, 3, 4],
    }.get(config.scenario_bucket, [1, 2, 3])

    couriers = [Courier(id=f"courier_{i}", node_id="hub") for i in range(config.num_couriers)]

    # When rolling arrivals are enabled, spread orders across the first ~75% of
    # the shift so the agent sees genuine in-shift variability (and the last
    # arrivals still have a fighting chance against max_ticks).
    arrival_ticks = _arrival_schedule(config, rng)

    orders: List[Order] = []
    for i in range(config.num_orders):
        prep = rng.choice(prep_choices)
        extra_deadline = rng.randint(0, 5)
        if config.scenario_bucket == "long_tail" and i == config.num_orders - 1:
            extra_deadline += 5
        arrival = arrival_ticks[i]
        # Deadline is relative to arrival when rolling, otherwise absolute.
        sla_window = deadline_base + extra_deadline
        deadline = min(config.max_ticks, arrival + sla_window) if config.rolling_arrivals else min(
            config.max_ticks, sla_window
        )
        orders.append(
            Order(
                id=f"order_{i}",
                kind="food" if i % 2 == 0 else "grocery",
                pickup_node_id=f"store_{i % pickup_count}",
                dropoff_node_id=f"customer_{i}",
                created_tick=arrival,
                arrival_tick=arrival,
                prep_remaining=prep,
                deadline_tick=deadline,
                status=OrderStatus.READY if prep == 0 else OrderStatus.QUEUED,
            )
        )

    traffic_multipliers = _traffic_multipliers(config, node_ids, rng)

    return Scenario(
        config=config,
        nodes=nodes,
        travel_time_matrix=travel,
        couriers=couriers,
        orders=orders,
        traffic_multipliers=traffic_multipliers,
    )


def _arrival_schedule(config: Config, rng: random.Random) -> List[int]:
    if not config.rolling_arrivals:
        return [0] * config.num_orders
    # Spread arrivals across the front portion of the shift with light jitter.
    horizon = max(1, int(config.max_ticks * 0.6))
    base_gap = max(1, horizon // max(1, config.num_orders))
    schedule = []
    for i in range(config.num_orders):
        nominal = i * base_gap
        jitter = rng.randint(0, 1)
        schedule.append(min(horizon, nominal + jitter))
    return schedule


def _traffic_multipliers(
    config: Config, node_ids: List[str], rng: random.Random
) -> Dict[Tuple[str, str], float]:
    if config.traffic_noise <= 0.0:
        return {}
    multipliers: Dict[Tuple[str, str], float] = {}
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            multipliers[(src, dst)] = 1.0 + rng.random() * config.traffic_noise
    return multipliers
