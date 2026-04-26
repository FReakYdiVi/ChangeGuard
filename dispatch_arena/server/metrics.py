"""Episode metrics for Dispatch Arena."""

from __future__ import annotations

from typing import Dict

from dispatch_arena.models import OrderStatus, State


def episode_metrics(state: State) -> Dict[str, float | int]:
    delivered_orders = [o for o in state.orders if o.status == OrderStatus.DELIVERED]
    delivered = len(delivered_orders)
    expired = sum(1 for o in state.orders if o.status == OrderStatus.EXPIRED)

    # Use the order's delivered_tick (set at delivery time) instead of the
    # current state.tick, otherwise an episode that runs past a deadline marks
    # all delivered orders as late even when they were actually on time.
    late_orders = [
        o for o in delivered_orders
        if o.delivered_tick is not None and o.delivered_tick > o.deadline_tick
    ]
    late = len(late_orders)

    delivery_ticks = [
        o.delivered_tick - o.created_tick
        for o in delivered_orders
        if o.delivered_tick is not None
    ]
    lateness = [
        max(0, o.delivered_tick - o.deadline_tick)
        for o in delivered_orders
        if o.delivered_tick is not None
    ]

    total_orders = len(state.orders)
    return {
        "orders": total_orders,
        "delivered": delivered,
        "expired": expired,
        "late": late,
        "success_rate": delivered / total_orders if total_orders else 0.0,
        "on_time_rate": (delivered - late) / delivered if delivered else 0.0,
        "expired_rate": expired / total_orders if total_orders else 0.0,
        "mean_delivery_ticks": sum(delivery_ticks) / len(delivery_ticks) if delivery_ticks else 0.0,
        "mean_lateness": sum(lateness) / len(lateness) if lateness else 0.0,
        "invalid_actions": state.invalid_actions,
        "invalid_rate": state.invalid_actions / state.tick if state.tick else 0.0,
        "total_reward": state.total_reward,
        "sla_pressure": state.sla_pressure,
    }
