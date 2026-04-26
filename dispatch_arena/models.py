"""Public models for the Dispatch Arena environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DispatchArenaModel(BaseModel):
    """Base model with JSON-friendly helpers used by server and client."""

    model_config = ConfigDict(use_enum_values=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class Mode(str, Enum):
    MINI = "mini"
    NORMAL = "normal"


class CourierStatus(str, Enum):
    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    WAITING_PICKUP = "waiting_pickup"
    TO_DROPOFF = "to_dropoff"
    REPOSITIONING = "repositioning"


class OrderStatus(str, Enum):
    QUEUED = "queued"
    READY = "ready"
    PICKED = "picked"
    DELIVERED = "delivered"
    EXPIRED = "expired"


class VerifierVerdict(str, Enum):
    IN_PROGRESS = "in_progress"
    DELIVERED_SUCCESSFULLY = "delivered_successfully"
    TIMEOUT_FAILURE = "timeout_failure"
    PARTIAL_SUCCESS = "partial_success"


class MiniActionType(str, Enum):
    WAIT = "wait"
    GO_PICKUP = "go_pickup"
    GO_DROPOFF = "go_dropoff"
    PICKUP = "pickup"
    DROPOFF = "dropoff"


class NormalActionType(str, Enum):
    ASSIGN = "assign"
    REPOSITION = "reposition"
    HOLD = "hold"
    PRIORITIZE = "prioritize"


class Action(DispatchArenaModel):
    """OpenEnv-facing action payload."""

    action_type: str
    courier_id: Optional[str] = None
    order_id: Optional[str] = None
    node_id: Optional[str] = None

    @property
    def name(self) -> str:
        return self.action_type

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Action":
        return cls.model_validate(dict(data))


class Node(DispatchArenaModel):
    id: str
    kind: str
    label: str


class Courier(DispatchArenaModel):
    id: str
    node_id: str
    status: CourierStatus = CourierStatus.IDLE
    eta_remaining: int = 0
    assigned_order_id: Optional[str] = None
    load: Optional[str] = None
    target_node_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_eta(self) -> "Courier":
        if self.eta_remaining < 0:
            raise ValueError("eta_remaining must be >= 0")
        return self


class Order(DispatchArenaModel):
    id: str
    kind: str = "food"
    pickup_node_id: str
    dropoff_node_id: str
    created_tick: int = 0
    arrival_tick: int = 0
    prep_remaining: Optional[int] = None
    deadline_tick: int = 20
    status: OrderStatus = OrderStatus.QUEUED
    assigned_courier_id: Optional[str] = None
    ready_now: Optional[bool] = None
    delivered_tick: Optional[int] = None


class Config(DispatchArenaModel):
    mode: Mode = Mode.MINI
    max_ticks: int = 12
    visible_prep: bool = False
    num_couriers: int = 1
    num_orders: int = 1
    scenario_bucket: str = "easy"
    progress_shaping: bool = True
    rolling_arrivals: bool = False
    traffic_noise: float = 0.0

    @model_validator(mode="after")
    def _validate_config(self) -> "Config":
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be > 0")
        if self.traffic_noise < 0.0 or self.traffic_noise > 2.0:
            raise ValueError("traffic_noise must be in [0.0, 2.0]")
        if self.mode == Mode.MINI:
            self.num_couriers = 1
            self.num_orders = 1
        if self.mode == Mode.NORMAL:
            self.num_couriers = min(max(self.num_couriers, 2), 5)
            self.num_orders = min(max(self.num_orders, 3), 10)
        return self


class RewardBreakdown(DispatchArenaModel):
    """Machine-readable reward decomposition for a transition."""

    step_cost: float = 0.0
    progress_reward: float = 0.0
    invalid_penalty: float = 0.0
    success_reward: float = 0.0
    timeout_penalty: float = 0.0
    on_time_bonus: float = 0.0
    late_penalty: float = 0.0
    idle_penalty: float = 0.0
    route_churn_penalty: float = 0.0
    fairness_penalty: float = 0.0
    total_reward: float = 0.0


class State(DispatchArenaModel):
    """Sanitized public environment state.

    Hidden `prep_remaining` values are excluded unless visible prep mode is
    explicitly enabled by the scenario config.
    """

    episode_id: Optional[str] = None
    tick: int = 0
    max_ticks: int = 12
    seed: Optional[int] = None
    mode: Mode = Mode.MINI
    nodes: List[Node] = Field(default_factory=list)
    travel_time_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    couriers: List[Courier] = Field(default_factory=list)
    orders: List[Order] = Field(default_factory=list)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    done: bool = False
    truncated: bool = False
    verifier_status: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    last_action: Optional[Action] = None
    event_log: List[str] = Field(default_factory=list)
    invalid_actions: int = 0
    total_reward: float = 0.0
    backlog: int = 0
    sla_pressure: float = 0.0

    @model_validator(mode="after")
    def _validate_ticks(self) -> "State":
        if self.tick < 0:
            raise ValueError("tick must be >= 0")
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be > 0")
        return self


class Observation(DispatchArenaModel):
    """Public observation returned by reset and step."""

    state: State = Field(default_factory=State)
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    verifier_status: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    legal_actions: List[str] = Field(default_factory=list)
    action_mask: List[int] = Field(default_factory=list)
    summary_text: str = "Awaiting dispatch."
    info: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        return cls.model_validate(dict(data))


class EpisodeSummary(DispatchArenaModel):
    episode_id: Optional[str] = None
    seed: Optional[int] = None
    mode: Mode = Mode.MINI
    max_ticks: int = 12
    ticks_taken: int = 0
    invalid_actions: int = 0
    total_reward: float = 0.0
    final_verdict: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    action_trace: List[Action] = Field(default_factory=list)
    delivered_orders: int = 0
    expired_orders: int = 0


DispatchArenaAction = Action
DispatchArenaObservation = Observation
DispatchArenaState = State
