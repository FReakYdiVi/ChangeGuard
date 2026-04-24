"""Thin TRL wrapper for ChangeGuard environment.

This adapter intentionally keeps environment dynamics in the server and only
provides typed tool methods plus training-facing bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from changeguard.client import EnvClient
from changeguard.models import (
    Action,
    ActionType,
    EpisodeSummary,
    Observation,
    RolloutStage,
    StepResult,
    TenantId,
)


@dataclass
class ChangeGuardToolEnv:
    """Training adapter that delegates all state transitions to the server.

    The wrapper stores cumulative reward and verifier metadata so GRPO logging can
    read concise metrics without re-computing environment logic.
    """

    client: EnvClient
    scenario_id: Optional[str] = None
    prompt_style: Optional[str] = None
    current_observation: Optional[Observation] = None
    reward_total: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    episode_summary: Optional[EpisodeSummary] = None
    violation_flags: Dict[str, Any] = field(default_factory=dict)
    episode_metrics: Dict[str, Any] = field(default_factory=dict)

    def reset(
        self,
        *,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        scenario_id: Optional[str] = None,
        prompt_style: Optional[str] = None,
    ) -> Observation:
        """Reset the remote environment session.

        Args:
            difficulty: Difficulty level (`easy|medium|hard`).
            seed: Optional deterministic seed.
            scenario_id: Optional scenario label for experiment tracking.
            prompt_style: Optional prompt-style label for training experiments.

        Returns:
            The initial typed observation from the environment.
        """
        self.scenario_id = scenario_id
        self.prompt_style = prompt_style
        self.current_observation = self.client.reset(
            seed=seed,
            difficulty=difficulty,
            scenario_id=scenario_id,
            prompt_style=prompt_style,
        )
        self.reward_total = 0.0
        self.reward_components = {
            "progress_reward": 0.0,
            "inspection_reward": 0.0,
            "safety_reward": 0.0,
            "invalid_action_penalty": 0.0,
            "loop_penalty": 0.0,
            "terminal_bonus_or_penalty": 0.0,
            "total_reward": 0.0,
        }
        self.done = False
        self.episode_summary = None
        self.violation_flags = {}
        self.episode_metrics = {}
        return self.current_observation

    def available_tools(self) -> List[Dict[str, Any]]:
        """Return the public tool catalog exposed to trainer/model."""
        return [
            {"name": "inspect_tenant", "args": {"tenant": "TenantId"}},
            {"name": "inspect_compatibility", "args": {}},
            {"name": "inspect_logs", "args": {}},
            {"name": "canary_upgrade", "args": {"tenant": "TenantId=A"}},
            {"name": "promote_upgrade", "args": {"tenant": "TenantId(B|C)"}},
            {"name": "enable_compat_mode", "args": {"tenant": "TenantId=C"}},
            {"name": "request_approval", "args": {"tenant": "TenantId=C"}},
            {"name": "defer_tenant", "args": {"tenant": "TenantId=C"}},
            {"name": "rollback_tenant", "args": {"tenant": "Optional[TenantId]"}},
        ]

    def inspect_tenant(self, tenant: TenantId = TenantId.C) -> StepResult:
        """Inspect tenant profile/risk metadata.

        Args:
            tenant: Tenant identifier to focus inspection context.

        Returns:
            StepResult with updated observation and reward breakdown.
        """
        self._ensure_active("inspect_tenant")
        return self._apply_action(ActionType.INSPECT_TENANT_PROFILE, tenant)

    def inspect_compatibility(self) -> StepResult:
        """Inspect compatibility report for upgrade readiness.

        Args:
            None.

        Returns:
            StepResult with compatibility signal updates.
        """
        self._ensure_active("inspect_compatibility")
        return self._apply_action(ActionType.INSPECT_COMPATIBILITY_REPORT)

    def inspect_logs(self) -> StepResult:
        """Inspect export/service logs for hidden compatibility failures.

        Args:
            None.

        Returns:
            StepResult with log-derived risk signals.
        """
        self._ensure_active("inspect_logs")
        return self._apply_action(ActionType.INSPECT_EXPORT_JOB_STATUS)

    def canary_upgrade(self, tenant: TenantId = TenantId.A) -> StepResult:
        """Canary upgrade the low-risk tenant first.

        Args:
            tenant: Tenant to canary (must be Tenant A in v1).

        Returns:
            StepResult after canary attempt.

        Raises:
            ValueError: If called on unsupported tenant or wrong stage.
        """
        self._ensure_active("canary_upgrade")
        if tenant != TenantId.A:
            raise ValueError("canary_upgrade only supports TenantId.A in v1.")
        self._ensure_stage(
            allowed={RolloutStage.PLAN.value},
            tool_name="canary_upgrade",
        )
        return self._apply_action(ActionType.CANARY_ROLLOUT_TENANT_A, tenant)

    def promote_upgrade(self, tenant: TenantId) -> StepResult:
        """Promote rollout to next tenant.

        Args:
            tenant: Target tenant (`TenantId.B` or `TenantId.C`).

        Returns:
            StepResult after promotion attempt.

        Raises:
            ValueError: If tenant/stage combination is invalid.
        """
        self._ensure_active("promote_upgrade")
        if tenant not in {TenantId.B, TenantId.C}:
            raise ValueError("promote_upgrade supports only TenantId.B or TenantId.C.")
        if tenant == TenantId.B:
            self._ensure_stage(
                allowed={RolloutStage.CANARY_A.value, RolloutStage.ROLLOUT_B.value},
                tool_name="promote_upgrade(B)",
            )
            return self._apply_action(ActionType.EXPAND_ROLLOUT_TENANT_B, tenant)

        self._ensure_stage(
            allowed={RolloutStage.ROLLOUT_B.value, RolloutStage.GATED_C.value},
            tool_name="promote_upgrade(C)",
        )
        return self._apply_action(ActionType.FINALIZE_UPGRADE, tenant)

    def enable_compat_mode(self, tenant: TenantId = TenantId.C) -> StepResult:
        """Enable compatibility mode for Tenant C before finalization.

        Args:
            tenant: Must be Tenant C.

        Returns:
            StepResult after compatibility mode action.

        Raises:
            ValueError: If called for non-C tenant.
        """
        self._ensure_active("enable_compat_mode")
        if tenant != TenantId.C:
            raise ValueError("enable_compat_mode only supports TenantId.C in v1.")
        return self._apply_action(ActionType.ENABLE_COMPAT_MODE_TENANT_C, tenant)

    def request_approval(self, tenant: TenantId = TenantId.C) -> StepResult:
        """Request enterprise approval gate required for C finalization.

        Args:
            tenant: Must be Tenant C.

        Returns:
            StepResult after approval request.

        Raises:
            ValueError: If called for non-C tenant or before B rollout.
        """
        self._ensure_active("request_approval")
        if tenant != TenantId.C:
            raise ValueError("request_approval only supports TenantId.C in v1.")
        self._ensure_stage(
            allowed={RolloutStage.ROLLOUT_B.value, RolloutStage.GATED_C.value},
            tool_name="request_approval",
        )
        return self._apply_action(ActionType.REQUEST_APPROVAL_TENANT_C, tenant)

    def defer_tenant(self, tenant: TenantId = TenantId.C) -> StepResult:
        """Defer risky tenant rollout instead of unsafe promotion.

        Args:
            tenant: Tenant to defer (defaults to Tenant C).

        Returns:
            StepResult possibly leading to safe partial completion.
        """
        self._ensure_active("defer_tenant")
        return self._apply_action(ActionType.PAUSE_ROLLOUT, tenant)

    def rollback_tenant(self, tenant: Optional[TenantId] = None) -> StepResult:
        """Rollback a tenant or full rollout (when tenant is None).

        Args:
            tenant: Optional tenant id. If None, rollback all tenants.

        Returns:
            StepResult containing rollback outcome.
        """
        self._ensure_active("rollback_tenant")
        return self._apply_action(ActionType.ROLLBACK_UPGRADE, tenant)

    def call_tool(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> StepResult:
        """Dispatch a named tool call to the matching typed method.

        Args:
            tool_name: Public tool method name.
            tool_args: Optional dict of keyword arguments.

        Returns:
            StepResult from the selected tool.

        Raises:
            ValueError: If tool_name is unknown or arguments are invalid.
        """
        tool_args = tool_args or {}
        dispatch = {
            "inspect_tenant": lambda: self.inspect_tenant(_tenant_arg(tool_args.get("tenant", TenantId.C))),
            "inspect_compatibility": self.inspect_compatibility,
            "inspect_logs": self.inspect_logs,
            "canary_upgrade": lambda: self.canary_upgrade(_tenant_arg(tool_args.get("tenant", TenantId.A))),
            "promote_upgrade": lambda: self.promote_upgrade(_tenant_arg(tool_args.get("tenant"))),
            "enable_compat_mode": lambda: self.enable_compat_mode(_tenant_arg(tool_args.get("tenant", TenantId.C))),
            "request_approval": lambda: self.request_approval(_tenant_arg(tool_args.get("tenant", TenantId.C))),
            "defer_tenant": lambda: self.defer_tenant(_tenant_arg(tool_args.get("tenant", TenantId.C))),
            "rollback_tenant": lambda: self.rollback_tenant(
                _tenant_arg(tool_args["tenant"]) if "tenant" in tool_args and tool_args["tenant"] is not None else None
            ),
        }
        if tool_name not in dispatch:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available tools: {[t['name'] for t in self.available_tools()]}"
            )
        return dispatch[tool_name]()

    def get_episode_summary(self) -> EpisodeSummary:
        """Return episode summary from server (latest cached if done)."""
        if self.episode_summary is not None:
            return self.episode_summary
        return self.client.fetch_summary()

    # Internal helpers
    def _ensure_active(self, tool_name: str) -> None:
        if self.current_observation is None:
            raise ValueError(f"{tool_name} called before reset().")
        if self.done:
            raise ValueError(f"{tool_name} called after episode is done. Reset first.")

    def _ensure_stage(self, *, allowed: set[str], tool_name: str) -> None:
        assert self.current_observation is not None
        current = self.current_observation.stage.value
        if current not in allowed:
            raise ValueError(
                f"{tool_name} invalid in stage '{current}'. Allowed stages: {sorted(allowed)}"
            )

    def _apply_action(self, action_type: ActionType, tenant: Optional[TenantId] = None) -> StepResult:
        action = Action(action_type=action_type, target_tenant=tenant)
        result = self.client.step(action)
        self.current_observation = result.observation
        self.done = result.done

        # Store cumulative and per-component metrics on wrapper instance.
        self.reward_total += result.reward_total
        for name, value in result.reward_breakdown.to_dict().items():
            self.reward_components[name] = self.reward_components.get(name, 0.0) + float(value)

        vf = result.verifier_flags.to_dict()
        self.violation_flags = {
            k: vf[k]
            for k in [
                "approval_violation",
                "outage",
                "integrity_ok",
                "step_budget_respected",
                "verdict",
            ]
            if k in vf
        }

        if self.done:
            self.episode_summary = self.client.fetch_summary()
            self.episode_metrics = self.build_episode_metrics()

        return result

    def build_episode_metrics(self) -> Dict[str, Any]:
        """Build judge-friendly per-episode metrics for training/eval dashboards."""
        summary = self.get_episode_summary()
        verdict = summary.final_verdict.value
        safe_full = verdict == "safe_finalized"
        safe_partial = verdict == "safe_rollback"
        unsafe = not (safe_full or safe_partial)
        integrity_ok = self.violation_flags.get("integrity_ok", True)

        # Approximate blast radius as count of tenants ended in V2 in final observation.
        blast_radius = 0
        if self.current_observation is not None:
            blast_radius = sum(
                1
                for t in self.current_observation.tenants_visible.values()
                if t.schema_version.value == "V2"
            )

        metrics = {
            "safe_full_completion": int(safe_full),
            "safe_partial_completion": int(safe_partial),
            "unsafe_completion": int(unsafe),
            "approval_violation": int(bool(self.violation_flags.get("approval_violation", False))),
            "data_integrity_violation": int(not bool(integrity_ok)),
            "blast_radius": blast_radius,
            "invalid_action_count": summary.stats.invalid_actions,
            "mean_steps": summary.stats.steps_taken,
            "reward_total": self.reward_total,
        }
        metrics.update({k: float(v) for k, v in self.reward_components.items()})
        return metrics


def _tenant_arg(value: Any) -> TenantId:
    if value is None:
        raise ValueError("Missing required tool argument: tenant")
    if isinstance(value, TenantId):
        return value
    return TenantId(str(value))
