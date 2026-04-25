"""Reward model for Dispatch Arena transitions."""

from __future__ import annotations

from dataclasses import dataclass

from dispatch_arena.models import MiniActionType, NormalActionType, RewardBreakdown


@dataclass(frozen=True)
class RewardConfig:
    step_cost: float = -0.1
    mini_move_reward: float = 0.2
    mini_pickup_reward: float = 1.0
    pickup_progress_bonus: float = 0.5
    success_reward: float = 10.0
    on_time_bonus: float = 2.0
    invalid_penalty: float = -1.0
    timeout_penalty: float = -5.0
    late_penalty: float = -2.0
    idle_penalty: float = -0.05
    route_churn_penalty: float = -0.25
    fairness_penalty: float = -0.1


class RewardModel:
    """Builds decomposed rewards for each step."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def base(self) -> RewardBreakdown:
        return RewardBreakdown(step_cost=self.config.step_cost)

    def invalid(self, reward: RewardBreakdown) -> RewardBreakdown:
        reward.invalid_penalty += self.config.invalid_penalty
        return self.finalize(reward)

    def timeout(self, reward: RewardBreakdown) -> RewardBreakdown:
        reward.timeout_penalty += self.config.timeout_penalty
        return self.finalize(reward)

    def mini_progress(self, reward: RewardBreakdown, action_type: str) -> None:
        if action_type in {MiniActionType.GO_PICKUP.value, MiniActionType.GO_DROPOFF.value}:
            reward.progress_reward += self.config.mini_move_reward
        elif action_type == MiniActionType.PICKUP.value:
            reward.progress_reward += self.config.mini_pickup_reward
        elif action_type == MiniActionType.DROPOFF.value:
            reward.success_reward += self.config.success_reward

    def normal_action_progress(self, reward: RewardBreakdown, action_type: str) -> None:
        if action_type == NormalActionType.ASSIGN.value:
            reward.progress_reward += self.config.pickup_progress_bonus
        elif action_type == NormalActionType.REPOSITION.value:
            reward.progress_reward += self.config.mini_move_reward

    def delivered(self, reward: RewardBreakdown, on_time: bool) -> None:
        reward.success_reward += self.config.success_reward
        if on_time:
            reward.on_time_bonus += self.config.on_time_bonus
        else:
            reward.late_penalty += self.config.late_penalty

    def idle(self, reward: RewardBreakdown, idle_count: int) -> None:
        reward.idle_penalty += self.config.idle_penalty * idle_count

    def churn(self, reward: RewardBreakdown) -> None:
        reward.route_churn_penalty += self.config.route_churn_penalty

    def fairness(self, reward: RewardBreakdown, imbalance: int) -> None:
        if imbalance > 1:
            reward.fairness_penalty += self.config.fairness_penalty * (imbalance - 1)

    def finalize(self, reward: RewardBreakdown) -> RewardBreakdown:
        reward.total_reward = (
            reward.step_cost
            + reward.progress_reward
            + reward.invalid_penalty
            + reward.success_reward
            + reward.timeout_penalty
            + reward.on_time_bonus
            + reward.late_penalty
            + reward.idle_penalty
            + reward.route_churn_penalty
            + reward.fairness_penalty
        )
        return reward
