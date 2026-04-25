"""Authoritative Dispatch Arena simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from dispatch_arena.models import (
    Action,
    Config,
    Courier,
    CourierStatus,
    EpisodeSummary,
    MiniActionType,
    Mode,
    NormalActionType,
    Observation,
    Order,
    OrderStatus,
    State,
    VerifierVerdict,
)
from dispatch_arena.server.rewards import RewardModel
from dispatch_arena.server.scenarios import generate_scenario
from dispatch_arena.server.serializers import idle_courier_count, make_observation, public_state

DEFAULT_MAX_TICKS = 12
MINI_ACTION_ORDER = [action.value for action in MiniActionType]
NORMAL_ACTION_ORDER = [action.value for action in NormalActionType]


@dataclass
class DispatchArenaEnvironment:
    """Native OpenEnv-style dispatch simulation for mini and normal modes."""

    config: Config = field(default_factory=Config)
    _rng: random.Random = field(default_factory=random.Random)
    _state: Optional[State] = None
    _reward_model: RewardModel = field(default_factory=RewardModel)
    _action_trace: List[Action] = field(default_factory=list)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        config: Optional[Config | Mapping[str, Any]] = None,
    ) -> Observation:
        if config is not None:
            self.config = config if isinstance(config, Config) else Config.model_validate(dict(config))
        self._rng.seed(seed)
        scenario = generate_scenario(self.config, seed)
        self._state = State(
            episode_id=episode_id,
            tick=0,
            max_ticks=self.config.max_ticks,
            seed=seed,
            mode=self.config.mode,
            nodes=scenario.nodes,
            travel_time_matrix=scenario.travel_time_matrix,
            couriers=scenario.couriers,
            orders=scenario.orders,
        )
        self._action_trace = []
        self._refresh_derived()
        return self._observation(info={"reset": True})

    def step(self, action: Action | str | Mapping[str, Any]) -> Observation:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode already finished. Call reset() before stepping again.")

        parsed_action = self._coerce_action(action)
        reward = self._reward_model.base()
        info: Dict[str, Any] = {"invalid_action": False, "invalid_reason": None, "events": []}
        legal_actions = self.legal_actions()

        state.tick += 1
        state.last_action = parsed_action
        self._action_trace.append(parsed_action)

        if self.config.mode == Mode.MINI:
            valid = parsed_action.action_type in legal_actions
            if valid:
                self._progress_prep()
                self._apply_mini_action(parsed_action, reward, info)
            else:
                self._mark_invalid(parsed_action, reward, info)
        else:
            valid = self._is_valid_normal_action(parsed_action)
            if valid:
                self._progress_prep()
                self._apply_normal_action(parsed_action, reward, info)
                self._advance_normal_couriers(reward, info)
                self._expire_orders(info)
            else:
                self._mark_invalid(parsed_action, reward, info)
            self._reward_model.idle(reward, idle_courier_count(state))
            self._reward_model.fairness(reward, self._delivery_imbalance())

        if not state.done and state.tick >= state.max_ticks:
            state.done = True
            state.truncated = True
            self._reward_model.timeout(reward)
        else:
            self._reward_model.finalize(reward)

        state.reward_breakdown = reward
        state.total_reward += reward.total_reward
        state.event_log.extend(info["events"])
        self._refresh_derived()
        return self._observation(info=info)

    @property
    def state(self) -> State:
        return public_state(self._require_state(), self.config)

    def legal_actions(self) -> List[str]:
        state = self._require_state()
        if state.done:
            return []
        if state.mode == Mode.NORMAL:
            actions = [NormalActionType.HOLD.value]
            if any(c.status == CourierStatus.IDLE and c.load is None for c in state.couriers) and any(
                o.status in {OrderStatus.QUEUED, OrderStatus.READY} and o.assigned_courier_id is None
                for o in state.orders
            ):
                actions.append(NormalActionType.ASSIGN.value)
            if any(c.status == CourierStatus.IDLE and c.load is None for c in state.couriers):
                actions.append(NormalActionType.REPOSITION.value)
            if any(o.status in {OrderStatus.QUEUED, OrderStatus.READY} for o in state.orders):
                actions.append(NormalActionType.PRIORITIZE.value)
            return actions

        courier = state.couriers[0]
        order = state.orders[0]
        actions = []
        if courier.load is None and order.status in {OrderStatus.QUEUED, OrderStatus.READY} and courier.node_id != order.pickup_node_id:
            actions.append(MiniActionType.GO_PICKUP.value)
        if courier.node_id == order.pickup_node_id and courier.load is None and order.status == OrderStatus.READY:
            actions.append(MiniActionType.PICKUP.value)
        if courier.load == order.id and courier.node_id != order.dropoff_node_id:
            actions.append(MiniActionType.GO_DROPOFF.value)
        if courier.load == order.id and courier.node_id == order.dropoff_node_id:
            actions.append(MiniActionType.DROPOFF.value)
        actions.append(MiniActionType.WAIT.value)
        return actions

    def action_mask(self) -> List[int]:
        legal = set(self.legal_actions())
        order = NORMAL_ACTION_ORDER if self.config.mode == Mode.NORMAL else MINI_ACTION_ORDER
        return [1 if action in legal else 0 for action in order]

    def get_episode_summary(self) -> Dict[str, Any]:
        state = self._require_state()
        summary = EpisodeSummary(
            episode_id=state.episode_id,
            seed=state.seed,
            mode=state.mode,
            max_ticks=state.max_ticks,
            ticks_taken=state.tick,
            invalid_actions=state.invalid_actions,
            total_reward=state.total_reward,
            final_verdict=state.verifier_status,
            action_trace=self._action_trace,
            delivered_orders=sum(1 for order in state.orders if order.status == OrderStatus.DELIVERED),
            expired_orders=sum(1 for order in state.orders if order.status == OrderStatus.EXPIRED),
        )
        return summary.to_dict()

    def _observation(self, info: Optional[dict] = None) -> Observation:
        return make_observation(self._require_state(), self.config, self.legal_actions(), self.action_mask(), info=info)

    def _require_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _coerce_action(self, action: Action | str | Mapping[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, str):
            return Action(action_type=action)
        if isinstance(action, Mapping):
            return Action.model_validate(dict(action))
        raise TypeError("action must be Action, string, or mapping")

    def _mark_invalid(self, action: Action, reward, info: Dict[str, Any]) -> None:
        state = self._require_state()
        state.invalid_actions += 1
        info["invalid_action"] = True
        info["invalid_reason"] = f"{action.action_type} is not legal from the current state"
        info["events"].append(info["invalid_reason"])
        self._reward_model.invalid(reward)

    def _apply_mini_action(self, action: Action, reward, info: Dict[str, Any]) -> None:
        state = self._require_state()
        courier = state.couriers[0]
        order = state.orders[0]
        action_type = action.action_type

        if action_type == MiniActionType.WAIT.value:
            info["events"].append("courier waited")
        elif action_type == MiniActionType.GO_PICKUP.value:
            courier.node_id = order.pickup_node_id
            courier.status = CourierStatus.IDLE
            info["events"].append(f"{courier.id} moved to pickup")
        elif action_type == MiniActionType.PICKUP.value:
            order.status = OrderStatus.PICKED
            order.assigned_courier_id = courier.id
            courier.load = order.id
            courier.assigned_order_id = order.id
            courier.status = CourierStatus.IDLE
            info["events"].append(f"{courier.id} picked {order.id}")
        elif action_type == MiniActionType.GO_DROPOFF.value:
            courier.node_id = order.dropoff_node_id
            courier.status = CourierStatus.IDLE
            info["events"].append(f"{courier.id} moved to dropoff")
        elif action_type == MiniActionType.DROPOFF.value:
            order.status = OrderStatus.DELIVERED
            courier.load = None
            courier.assigned_order_id = None
            state.done = True
            info["events"].append(f"{order.id} delivered")
        self._reward_model.mini_progress(reward, action_type)

    def _apply_normal_action(self, action: Action, reward, info: Dict[str, Any]) -> None:
        state = self._require_state()
        action_type = action.action_type
        if action_type == NormalActionType.HOLD.value:
            info["events"].append(f"{action.courier_id or 'dispatcher'} held")
        elif action_type == NormalActionType.PRIORITIZE.value:
            info["events"].append(f"{action.order_id or 'backlog'} prioritized")
        elif action_type == NormalActionType.REPOSITION.value:
            courier = self._courier(action.courier_id)
            if courier.node_id != action.node_id:
                courier.status = CourierStatus.REPOSITIONING
                courier.target_node_id = action.node_id
                courier.eta_remaining = self._travel_time(courier.node_id, action.node_id)
                info["events"].append(f"{courier.id} repositioning to {action.node_id}")
        elif action_type == NormalActionType.ASSIGN.value:
            courier = self._courier(action.courier_id)
            order = self._order(action.order_id)
            if courier.assigned_order_id and courier.assigned_order_id != order.id:
                self._reward_model.churn(reward)
            courier.assigned_order_id = order.id
            courier.status = CourierStatus.TO_PICKUP
            courier.target_node_id = order.pickup_node_id
            courier.eta_remaining = self._travel_time(courier.node_id, order.pickup_node_id)
            order.assigned_courier_id = courier.id
            info["events"].append(f"{courier.id} assigned {order.id}")
        self._reward_model.normal_action_progress(reward, action_type)

    def _is_valid_normal_action(self, action: Action) -> bool:
        state = self._require_state()
        action_type = action.action_type
        if action_type == NormalActionType.HOLD.value:
            return action.courier_id is None or any(c.id == action.courier_id for c in state.couriers)
        if action_type == NormalActionType.PRIORITIZE.value:
            return action.order_id is None or any(o.id == action.order_id and o.status in {OrderStatus.QUEUED, OrderStatus.READY} for o in state.orders)
        if action_type == NormalActionType.REPOSITION.value:
            if not action.courier_id or not action.node_id:
                return False
            courier = self._maybe_courier(action.courier_id)
            return courier is not None and courier.status == CourierStatus.IDLE and courier.load is None and action.node_id in self._node_ids()
        if action_type == NormalActionType.ASSIGN.value:
            if not action.courier_id or not action.order_id:
                return False
            courier = self._maybe_courier(action.courier_id)
            order = self._maybe_order(action.order_id)
            return (
                courier is not None
                and order is not None
                and courier.status == CourierStatus.IDLE
                and courier.load is None
                and order.status in {OrderStatus.QUEUED, OrderStatus.READY}
                and order.assigned_courier_id is None
            )
        return False

    def _advance_normal_couriers(self, reward, info: Dict[str, Any]) -> None:
        state = self._require_state()
        for courier in state.couriers:
            if courier.eta_remaining > 0:
                courier.eta_remaining -= 1
                if courier.eta_remaining > 0:
                    continue
                if courier.target_node_id:
                    courier.node_id = courier.target_node_id
            if courier.status == CourierStatus.REPOSITIONING and courier.eta_remaining == 0:
                courier.status = CourierStatus.IDLE
                courier.target_node_id = None
                info["events"].append(f"{courier.id} finished reposition")
            elif courier.status == CourierStatus.TO_PICKUP and courier.eta_remaining == 0:
                order = self._order(courier.assigned_order_id)
                if order.status == OrderStatus.READY:
                    self._auto_pickup(courier, order, reward, info)
                else:
                    courier.status = CourierStatus.WAITING_PICKUP
                    info["events"].append(f"{courier.id} waiting for {order.id}")
            elif courier.status == CourierStatus.WAITING_PICKUP:
                order = self._order(courier.assigned_order_id)
                if order.status == OrderStatus.READY:
                    self._auto_pickup(courier, order, reward, info)
            elif courier.status == CourierStatus.TO_DROPOFF and courier.eta_remaining == 0:
                order = self._order(courier.load)
                order.status = OrderStatus.DELIVERED
                on_time = state.tick <= order.deadline_tick
                courier.load = None
                courier.assigned_order_id = None
                courier.target_node_id = None
                courier.status = CourierStatus.IDLE
                self._reward_model.delivered(reward, on_time=on_time)
                info["events"].append(f"{order.id} delivered by {courier.id}")

    def _auto_pickup(self, courier: Courier, order: Order, reward, info: Dict[str, Any]) -> None:
        order.status = OrderStatus.PICKED
        courier.load = order.id
        courier.status = CourierStatus.TO_DROPOFF
        courier.target_node_id = order.dropoff_node_id
        courier.eta_remaining = self._travel_time(courier.node_id, order.dropoff_node_id)
        reward.progress_reward += self._reward_model.config.pickup_progress_bonus
        info["events"].append(f"{courier.id} picked {order.id}")

    def _progress_prep(self) -> None:
        state = self._require_state()
        for order in state.orders:
            if order.status not in {OrderStatus.QUEUED, OrderStatus.READY}:
                continue
            if order.prep_remaining is None:
                continue
            if order.prep_remaining > 0:
                order.prep_remaining -= 1
            if order.prep_remaining == 0:
                order.status = OrderStatus.READY

    def _expire_orders(self, info: Dict[str, Any]) -> None:
        state = self._require_state()
        for order in state.orders:
            if order.status in {OrderStatus.QUEUED, OrderStatus.READY} and state.tick > order.deadline_tick:
                order.status = OrderStatus.EXPIRED
                info["events"].append(f"{order.id} expired")

    def _refresh_derived(self) -> None:
        state = self._require_state()
        delivered = sum(1 for order in state.orders if order.status == OrderStatus.DELIVERED)
        active = [order for order in state.orders if order.status in {OrderStatus.QUEUED, OrderStatus.READY, OrderStatus.PICKED}]
        state.backlog = len(active)
        state.sla_pressure = 0.0 if not active else sum(1 for order in active if order.deadline_tick - state.tick <= 3) / len(active)
        if delivered == len(state.orders):
            state.done = True
            state.verifier_status = VerifierVerdict.DELIVERED_SUCCESSFULLY
        elif state.truncated:
            state.verifier_status = VerifierVerdict.PARTIAL_SUCCESS if delivered > 0 and state.mode == Mode.NORMAL else VerifierVerdict.TIMEOUT_FAILURE
        else:
            state.verifier_status = VerifierVerdict.IN_PROGRESS

    def _travel_time(self, src: str, dst: Optional[str]) -> int:
        if dst is None:
            return 0
        return max(1, self._require_state().travel_time_matrix.get(src, {}).get(dst, 1))

    def _delivery_imbalance(self) -> int:
        state = self._require_state()
        delivered_by: Dict[str, int] = {courier.id: 0 for courier in state.couriers}
        for order in state.orders:
            if order.status == OrderStatus.DELIVERED and order.assigned_courier_id:
                delivered_by[order.assigned_courier_id] = delivered_by.get(order.assigned_courier_id, 0) + 1
        return max(delivered_by.values(), default=0) - min(delivered_by.values(), default=0)

    def _courier(self, courier_id: Optional[str]) -> Courier:
        courier = self._maybe_courier(courier_id)
        if courier is None:
            raise ValueError(f"Unknown courier_id: {courier_id}")
        return courier

    def _maybe_courier(self, courier_id: Optional[str]) -> Optional[Courier]:
        return next((courier for courier in self._require_state().couriers if courier.id == courier_id), None)

    def _order(self, order_id: Optional[str]) -> Order:
        order = self._maybe_order(order_id)
        if order is None:
            raise ValueError(f"Unknown order_id: {order_id}")
        return order

    def _maybe_order(self, order_id: Optional[str]) -> Optional[Order]:
        return next((order for order in self._require_state().orders if order.id == order_id), None)

    def _node_ids(self) -> set[str]:
        return {node.id for node in self._require_state().nodes}


Environment = DispatchArenaEnvironment
