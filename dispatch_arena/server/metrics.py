"""Episode metrics for Dispatch Arena."""

from __future__ import annotations

from typing import Dict

from dispatch_arena.models import OrderStatus, State


def episode_metrics(state: State) -> Dict[str, float | int]:
    delivered = sum(1 for order in state.orders if order.status == OrderStatus.DELIVERED)
    expired = sum(1 for order in state.orders if order.status == OrderStatus.EXPIRED)
    late = sum(
        1
        for order in state.orders
        if order.status == OrderStatus.DELIVERED and state.tick > order.deadline_tick
    )
    return {
        "orders": len(state.orders),
        "delivered": delivered,
        "expired": expired,
        "late": late,
        "success_rate": delivered / len(state.orders) if state.orders else 0.0,
        "invalid_actions": state.invalid_actions,
        "total_reward": state.total_reward,
        "sla_pressure": state.sla_pressure,
    }
