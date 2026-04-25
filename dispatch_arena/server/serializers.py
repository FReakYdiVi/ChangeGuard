"""Serialization helpers that keep hidden arena state private."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from dispatch_arena.models import Config, CourierStatus, Mode, Observation, Order, OrderStatus, State


def public_state(state: State, config: Config) -> State:
    """Return a copy of state safe for observations, replay, and summaries."""

    sanitized = deepcopy(state)
    sanitized.orders = [_public_order(order, config.visible_prep) for order in sanitized.orders]
    sanitized.backlog = sum(1 for order in sanitized.orders if order.status in {OrderStatus.QUEUED, OrderStatus.READY})
    sanitized.sla_pressure = _sla_pressure(sanitized)
    return sanitized


def observation_summary(state: State) -> str:
    delivered = sum(1 for order in state.orders if order.status == OrderStatus.DELIVERED)
    active = sum(1 for order in state.orders if order.status in {OrderStatus.QUEUED, OrderStatus.READY, OrderStatus.PICKED})
    courier_bits = ", ".join(
        f"{courier.id}@{courier.node_id}:{courier.status.value}" for courier in state.couriers
    )
    return (
        f"mode={state.mode.value}; tick={state.tick}/{state.max_ticks}; "
        f"active_orders={active}; delivered={delivered}; couriers=[{courier_bits}]"
    )


def make_observation(
    state: State,
    config: Config,
    legal_actions: Iterable[str],
    action_mask: list[int],
    info: dict | None = None,
) -> Observation:
    public = public_state(state, config)
    return Observation(
        state=public,
        reward=public.reward_breakdown.total_reward,
        done=public.done,
        truncated=public.truncated,
        verifier_status=public.verifier_status,
        reward_breakdown=public.reward_breakdown,
        legal_actions=list(legal_actions),
        action_mask=action_mask,
        summary_text=observation_summary(public),
        info=info or {},
    )


def _public_order(order: Order, visible_prep: bool) -> Order:
    public = order.model_copy(deep=True)
    if visible_prep:
        public.ready_now = public.status in {OrderStatus.READY, OrderStatus.PICKED, OrderStatus.DELIVERED}
    else:
        public.prep_remaining = None
        public.ready_now = None
    return public


def _sla_pressure(state: State) -> float:
    active_deadlines = [
        order.deadline_tick - state.tick
        for order in state.orders
        if order.status in {OrderStatus.QUEUED, OrderStatus.READY, OrderStatus.PICKED}
    ]
    if not active_deadlines:
        return 0.0
    urgent = sum(1 for remaining in active_deadlines if remaining <= 3)
    return urgent / len(active_deadlines)


def terminal_verdict(state: State) -> str:
    delivered = sum(1 for order in state.orders if order.status == OrderStatus.DELIVERED)
    if delivered == len(state.orders):
        return "delivered_successfully"
    if state.truncated:
        if state.mode == Mode.NORMAL and delivered > 0:
            return "partial_success"
        return "timeout_failure"
    return "in_progress"


def idle_courier_count(state: State) -> int:
    return sum(1 for courier in state.couriers if courier.status == CourierStatus.IDLE and courier.load is None)
