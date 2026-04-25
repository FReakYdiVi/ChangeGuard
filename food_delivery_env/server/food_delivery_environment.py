"""Authoritative food delivery environment engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from food_delivery_env.models import (
    Action,
    ActionType,
    FoodDeliveryObservation,
    FoodDeliveryState,
    Location,
    Observation,
    OrderStatus,
    RewardBreakdown,
    State,
    VerifierVerdict,
)


DEFAULT_MAX_STEPS = 8


@dataclass
class WorldState:
    """Internal mutable state, including hidden cooking progress."""

    episode_id: Optional[str] = None
    seed: Optional[int] = None
    max_steps: int = DEFAULT_MAX_STEPS
    courier_id: str = "courier_0"
    order_id: str = "order_0"
    restaurant_id: str = "restaurant_0"
    customer_id: str = "customer_0"
    courier_location: Location = Location.DISPATCH
    courier_assigned_order_id: Optional[str] = None
    order_status: OrderStatus = OrderStatus.UNASSIGNED
    food_ready: bool = False
    prep_delay_steps: int = 1
    steps_taken: int = 0
    invalid_actions: int = 0
    total_reward: float = 0.0
    done: bool = False
    truncated: bool = False
    verifier_status: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    last_action: Optional[ActionType] = None
    action_trace: List[ActionType] = field(default_factory=list)

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.steps_taken)


class LegalActionPolicy:
    """Computes legal actions and masks from authoritative state."""

    def legal_actions(self, world: WorldState) -> List[ActionType]:
        if world.done:
            return []
        if world.order_status == OrderStatus.UNASSIGNED:
            return [ActionType.ASSIGN_ORDER]
        if world.order_status in {OrderStatus.ASSIGNED, OrderStatus.READY_AT_RESTAURANT}:
            if world.courier_location == Location.DISPATCH:
                return [ActionType.MOVE_TO_RESTAURANT]
            if world.courier_location == Location.RESTAURANT:
                if world.food_ready:
                    return [ActionType.PICKUP_ORDER]
                return [ActionType.WAIT_AT_RESTAURANT]
        if world.order_status == OrderStatus.PICKED_UP:
            if world.courier_location == Location.RESTAURANT:
                return [ActionType.MOVE_TO_CUSTOMER]
            if world.courier_location == Location.CUSTOMER:
                return [ActionType.DELIVER_ORDER]
        return []

    def action_mask(self, world: WorldState) -> List[int]:
        legal = set(self.legal_actions(world))
        return [1 if action_type in legal else 0 for action_type in ActionType]


class RewardModel:
    """Small dense reward model for the finite-state benchmark."""

    step_penalty = -0.1
    progress_rewards: Dict[ActionType, float] = {
        ActionType.ASSIGN_ORDER: 0.5,
        ActionType.MOVE_TO_RESTAURANT: 0.2,
        ActionType.PICKUP_ORDER: 1.0,
        ActionType.MOVE_TO_CUSTOMER: 0.2,
    }
    invalid_action_penalty = -1.0
    delivered_reward = 10.0
    timeout_penalty = -5.0

    def base(self) -> RewardBreakdown:
        return RewardBreakdown(step_penalty=self.step_penalty)

    def finalize(self, reward: RewardBreakdown) -> RewardBreakdown:
        reward.total_reward = (
            reward.step_penalty
            + reward.progress_reward
            + reward.invalid_action_penalty
            + reward.terminal_reward
        )
        return reward


class Verifier:
    """Maps terminal world state into public verdicts."""

    def verdict(self, world: WorldState) -> VerifierVerdict:
        if world.order_status == OrderStatus.DELIVERED:
            return VerifierVerdict.DELIVERED_SUCCESSFULLY
        if world.truncated:
            return VerifierVerdict.TIMEOUT_FAILURE
        return VerifierVerdict.IN_PROGRESS


class TransitionEngine:
    """Applies action semantics after time has advanced."""

    def apply(self, world: WorldState, action_type: ActionType) -> None:
        if action_type == ActionType.ASSIGN_ORDER:
            world.courier_assigned_order_id = world.order_id
            world.order_status = OrderStatus.ASSIGNED
        elif action_type == ActionType.MOVE_TO_RESTAURANT:
            world.courier_location = Location.RESTAURANT
        elif action_type == ActionType.WAIT_AT_RESTAURANT:
            pass
        elif action_type == ActionType.PICKUP_ORDER:
            world.order_status = OrderStatus.PICKED_UP
        elif action_type == ActionType.MOVE_TO_CUSTOMER:
            world.courier_location = Location.CUSTOMER
        elif action_type == ActionType.DELIVER_ORDER:
            world.order_status = OrderStatus.DELIVERED
            world.done = True


class ObservationBuilder:
    """Converts hidden world state into public state and observation models."""

    def __init__(self, legal_policy: LegalActionPolicy) -> None:
        self._legal_policy = legal_policy

    def public_state(self, world: WorldState) -> State:
        return FoodDeliveryState(
            episode_id=world.episode_id,
            courier_id=world.courier_id,
            order_id=world.order_id,
            restaurant_id=world.restaurant_id,
            customer_id=world.customer_id,
            courier_location=world.courier_location,
            courier_assigned_order_id=world.courier_assigned_order_id,
            order_status=world.order_status,
            food_ready=world.food_ready,
            steps_taken=world.steps_taken,
            steps_remaining=world.steps_remaining,
            last_action=world.last_action,
            verifier_status=world.verifier_status,
        )

    def observation(
        self,
        world: WorldState,
        reward: RewardBreakdown,
        info: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        return FoodDeliveryObservation(
            state=self.public_state(world),
            reward=reward.total_reward,
            done=world.done,
            truncated=world.truncated,
            verifier_status=world.verifier_status,
            reward_breakdown=reward,
            legal_actions=self._legal_policy.legal_actions(world),
            action_mask=self._legal_policy.action_mask(world),
            summary_text=self.summary_text(world),
            info=info or {},
        )

    def summary_text(self, world: WorldState) -> str:
        assigned = world.courier_assigned_order_id or "none"
        return (
            f"Courier={world.courier_id} at {world.courier_location.value}; "
            f"order={world.order_status.value}; assigned_order={assigned}; "
            f"food_ready={str(world.food_ready).lower()}; "
            f"steps_remaining={world.steps_remaining}."
        )


@dataclass
class FoodDeliveryEnvironment:
    """Native OpenEnv-style food delivery simulation."""

    max_steps: int = DEFAULT_MAX_STEPS
    _rng: random.Random = field(default_factory=random.Random)
    _world: Optional[WorldState] = None
    _legal_policy: LegalActionPolicy = field(default_factory=LegalActionPolicy)
    _transition_engine: TransitionEngine = field(default_factory=TransitionEngine)
    _reward_model: RewardModel = field(default_factory=RewardModel)
    _verifier: Verifier = field(default_factory=Verifier)

    def __post_init__(self) -> None:
        self._observation_builder = ObservationBuilder(self._legal_policy)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None) -> Observation:
        self._rng.seed(seed)
        self._world = WorldState(
            episode_id=episode_id,
            seed=seed,
            max_steps=self.max_steps,
            prep_delay_steps=self._rng.choice([1, 2, 3]),
        )
        return self._observation_builder.observation(
            self._world,
            RewardBreakdown(),
            info={"reset": True},
        )

    def step(self, action: Action | ActionType | str | Mapping[str, Any]) -> Observation:
        world = self._require_world()
        if world.done:
            raise RuntimeError("Episode already finished. Call reset() before stepping again.")

        action_type = self._coerce_action_type(action)
        reward = self._reward_model.base()
        info: Dict[str, Any] = {"invalid_action": False, "invalid_reason": None}

        legal_actions_at_decision = self._legal_policy.legal_actions(world)

        world.steps_taken += 1
        world.last_action = action_type
        world.action_trace.append(action_type)

        self._progress_prep_timer(world)

        if action_type not in legal_actions_at_decision:
            world.invalid_actions += 1
            reward.invalid_action_penalty = self._reward_model.invalid_action_penalty
            info["invalid_action"] = True
            info["invalid_reason"] = f"{action_type.value} is not legal from the current state"
        else:
            self._transition_engine.apply(world, action_type)
            reward.progress_reward = self._reward_model.progress_rewards.get(action_type, 0.0)
            if action_type == ActionType.DELIVER_ORDER:
                reward.terminal_reward = self._reward_model.delivered_reward

        if not world.done and world.steps_remaining == 0:
            world.done = True
            world.truncated = True
            reward.terminal_reward = self._reward_model.timeout_penalty

        world.verifier_status = self._verifier.verdict(world)
        self._reward_model.finalize(reward)
        world.total_reward += reward.total_reward
        return self._observation_builder.observation(world, reward, info=info)

    @property
    def state(self) -> State:
        return self._observation_builder.public_state(self._require_world())

    def get_episode_summary(self) -> Dict[str, Any]:
        world = self._require_world()
        return {
            "episode_id": world.episode_id,
            "seed": world.seed,
            "max_steps": world.max_steps,
            "steps_taken": world.steps_taken,
            "invalid_actions": world.invalid_actions,
            "total_reward": world.total_reward,
            "final_verdict": world.verifier_status.value,
            "action_trace": [action.value for action in world.action_trace],
        }

    def _require_world(self) -> WorldState:
        if self._world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._world

    def _coerce_action_type(self, action: Action | ActionType | str | Mapping[str, Any]) -> ActionType:
        if isinstance(action, Action):
            return action.action_type
        if isinstance(action, ActionType):
            return action
        if isinstance(action, str):
            return ActionType(action)
        if isinstance(action, Mapping):
            return Action.model_validate(dict(action)).action_type
        raise TypeError("action must be FoodDeliveryAction, ActionType, string, or mapping")

    def _progress_prep_timer(self, world: WorldState) -> None:
        if world.courier_assigned_order_id is None or world.food_ready:
            return
        if world.order_status not in {OrderStatus.ASSIGNED, OrderStatus.READY_AT_RESTAURANT}:
            return
        world.prep_delay_steps = max(0, world.prep_delay_steps - 1)
        if world.prep_delay_steps == 0:
            world.food_ready = True
            if world.order_status == OrderStatus.ASSIGNED:
                world.order_status = OrderStatus.READY_AT_RESTAURANT
