"""Typed public models for the food delivery OpenEnv simulation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    ASSIGN_ORDER = "assign_order"
    MOVE_TO_RESTAURANT = "move_to_restaurant"
    WAIT_AT_RESTAURANT = "wait_at_restaurant"
    PICKUP_ORDER = "pickup_order"
    MOVE_TO_CUSTOMER = "move_to_customer"
    DELIVER_ORDER = "deliver_order"


class Location(str, Enum):
    DISPATCH = "dispatch"
    RESTAURANT = "restaurant"
    CUSTOMER = "customer"


class OrderStatus(str, Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    READY_AT_RESTAURANT = "ready_at_restaurant"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"


class VerifierVerdict(str, Enum):
    IN_PROGRESS = "in_progress"
    DELIVERED_SUCCESSFULLY = "delivered_successfully"
    TIMEOUT_FAILURE = "timeout_failure"


class FoodDeliveryModel(BaseModel):
    """Base model with JSON-friendly helpers used by server and client."""

    model_config = ConfigDict(use_enum_values=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class FoodDeliveryAction(FoodDeliveryModel):
    """OpenEnv-facing action payload."""

    action_type: ActionType

    @property
    def name(self) -> str:
        return self.action_type.value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FoodDeliveryAction":
        return cls.model_validate(dict(data))


class FoodDeliveryState(FoodDeliveryModel):
    """Sanitized public environment state.

    Hidden cooking timer values are intentionally excluded.
    """

    episode_id: Optional[str] = None
    courier_id: str = "courier_0"
    order_id: str = "order_0"
    restaurant_id: str = "restaurant_0"
    customer_id: str = "customer_0"
    courier_location: Location = Location.DISPATCH
    courier_assigned_order_id: Optional[str] = None
    order_status: OrderStatus = OrderStatus.UNASSIGNED
    food_ready: bool = False
    steps_taken: int = 0
    steps_remaining: int = 8
    last_action: Optional[ActionType] = None
    verifier_status: VerifierVerdict = VerifierVerdict.IN_PROGRESS

    @model_validator(mode="after")
    def _validate_steps(self) -> "FoodDeliveryState":
        if self.steps_taken < 0:
            raise ValueError("steps_taken must be >= 0")
        if self.steps_remaining < 0:
            raise ValueError("steps_remaining must be >= 0")
        return self


class RewardBreakdown(FoodDeliveryModel):
    """Machine-readable reward decomposition for a transition."""

    step_penalty: float = 0.0
    progress_reward: float = 0.0
    invalid_action_penalty: float = 0.0
    terminal_reward: float = 0.0
    total_reward: float = 0.0


class FoodDeliveryObservation(FoodDeliveryModel):
    """Public observation returned by `reset()` and `step()`."""

    state: FoodDeliveryState = Field(default_factory=FoodDeliveryState)
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    verifier_status: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    legal_actions: List[ActionType] = Field(default_factory=list)
    action_mask: List[int] = Field(default_factory=lambda: [0] * len(ActionType))
    summary_text: str = "Awaiting assignment."
    info: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_action_mask(self) -> "FoodDeliveryObservation":
        if len(self.action_mask) != len(ActionType):
            raise ValueError("action_mask must have one entry per ActionType")
        legal = set(self.legal_actions)
        for index, action_type in enumerate(ActionType):
            expected = 1 if action_type in legal else 0
            if self.action_mask[index] != expected:
                raise ValueError("action_mask must match legal_actions in ActionType order")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FoodDeliveryObservation":
        return cls.model_validate(dict(data))


class EpisodeSummary(FoodDeliveryModel):
    """Episode-level summary exposed by the local HTTP app."""

    episode_id: Optional[str] = None
    seed: Optional[int] = None
    max_steps: int = 8
    steps_taken: int = 0
    invalid_actions: int = 0
    total_reward: float = 0.0
    final_verdict: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    action_trace: List[ActionType] = Field(default_factory=list)


Action = FoodDeliveryAction
Observation = FoodDeliveryObservation
State = FoodDeliveryState
