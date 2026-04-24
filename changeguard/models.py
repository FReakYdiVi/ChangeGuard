"""Typed shared data models for ChangeGuard.

This module defines the OpenEnv-facing action/observation models and
internal episode data structures used by server, client, wrappers, and tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class TenantId(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SchemaVersion(str, Enum):
    V1 = "V1"
    V2 = "V2"


class RolloutStage(str, Enum):
    PLAN = "plan"
    CANARY_A = "canary_a"
    ROLLOUT_B = "rollout_b"
    GATED_C = "gated_c"
    FINALIZED = "finalized"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ActionType(str, Enum):
    INSPECT_TENANT_PROFILE = "inspect_tenant_profile"
    INSPECT_COMPATIBILITY_REPORT = "inspect_compatibility_report"
    INSPECT_EXPORT_JOB_STATUS = "inspect_export_job_status"
    CANARY_ROLLOUT_TENANT_A = "canary_rollout_tenant_a"
    EXPAND_ROLLOUT_TENANT_B = "expand_rollout_tenant_b"
    PAUSE_ROLLOUT = "pause_rollout"
    ENABLE_COMPAT_MODE_TENANT_C = "enable_compat_mode_tenant_c"
    APPLY_BACKFILL = "apply_backfill"
    APPLY_ANNOUNCE_DEPRECATION = "apply_announce_deprecation"
    APPLY_DUAL_WRITE = "apply_dual_write"
    REQUEST_APPROVAL_TENANT_C = "request_approval_tenant_c"
    FINALIZE_UPGRADE = "finalize_upgrade"
    ROLLBACK_UPGRADE = "rollback_upgrade"


class DiffOpKind(str, Enum):
    """Kinds of schema diff operations a V1->V2 migration may contain."""

    RENAME_COL = "rename_col"              # requires compat_mode if any tenant has legacy_export
    ADD_NULL_COL = "add_null_col"          # always safe
    ADD_NOT_NULL_COL = "add_not_null_col"  # requires backfill if any tenant has strict_nullability
    DROP_COL = "drop_col"                  # requires announce_deprecation if any tenant has deprecation_policy
    CHANGE_TYPE = "change_type"            # requires dual_write if any tenant has type_sensitivity


class DependencyProfile(str, Enum):
    """Per-tenant sensitivity to specific diff-op kinds."""

    LEGACY_EXPORT = "legacy_export"            # breaks on RENAME_COL without compat_mode
    STRICT_NULLABILITY = "strict_nullability"  # breaks on ADD_NOT_NULL_COL without backfill
    DEPRECATION_POLICY = "deprecation_policy"  # breaks on DROP_COL without announce_deprecation
    TYPE_SENSITIVITY = "type_sensitivity"      # breaks on CHANGE_TYPE without dual_write


class MigrationMitigation(str, Enum):
    """Global migration strategies the agent can enable before finalize."""

    COMPAT_MODE = "compat_mode"
    BACKFILL = "backfill"
    ANNOUNCE_DEPRECATION = "announce_deprecation"
    DUAL_WRITE = "dual_write"


# Which mitigation satisfies which diff op kind.
MITIGATION_FOR_OP: Dict["DiffOpKind", "MigrationMitigation"] = {}
# Which dependency profile is sensitive to which op kind.
SENSITIVE_DEP_FOR_OP: Dict["DiffOpKind", "DependencyProfile"] = {}


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SignalLevel(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    FAILING = "failing"


class RiskHintLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerifierVerdict(str, Enum):
    IN_PROGRESS = "in_progress"
    SAFE_FINALIZED = "safe_finalized"
    SAFE_ROLLBACK = "safe_rollback"
    APPROVAL_VIOLATION = "approval_violation"
    COMPATIBILITY_VIOLATION = "compatibility_violation"
    UNSAFE_ROLLBACK = "unsafe_rollback"
    OUTAGE_FAILURE = "outage_failure"
    INTEGRITY_FAILURE = "integrity_failure"
    TIMEOUT_FAILURE = "timeout_failure"


MITIGATION_FOR_OP.update({
    DiffOpKind.RENAME_COL: MigrationMitigation.COMPAT_MODE,
    DiffOpKind.ADD_NOT_NULL_COL: MigrationMitigation.BACKFILL,
    DiffOpKind.DROP_COL: MigrationMitigation.ANNOUNCE_DEPRECATION,
    DiffOpKind.CHANGE_TYPE: MigrationMitigation.DUAL_WRITE,
    # ADD_NULL_COL intentionally absent — no mitigation needed.
})

SENSITIVE_DEP_FOR_OP.update({
    DiffOpKind.RENAME_COL: DependencyProfile.LEGACY_EXPORT,
    DiffOpKind.ADD_NOT_NULL_COL: DependencyProfile.STRICT_NULLABILITY,
    DiffOpKind.DROP_COL: DependencyProfile.DEPRECATION_POLICY,
    DiffOpKind.CHANGE_TYPE: DependencyProfile.TYPE_SENSITIVITY,
})


def _coerce_enum(enum_cls: Any, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _enum_as_value(data: Any) -> Any:
    if isinstance(data, Enum):
        return data.value
    if isinstance(data, dict):
        return {k: _enum_as_value(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_enum_as_value(v) for v in data]
    return data


@dataclass
class TenantVisibleState:
    """Visible per-tenant state exposed in observations.

    `dependencies_revealed` stays empty until the agent runs `inspect_tenant`
    on this tenant. Once revealed, the list shows which DependencyProfiles
    this tenant has, which lets the agent reason about required mitigations.
    """

    tenant_id: TenantId
    risk_tier: RiskTier
    schema_version: SchemaVersion = SchemaVersion.V1
    compat_mode_enabled: bool = False
    approval_required: bool = False
    dependencies_revealed: List[DependencyProfile] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tenant_id = _coerce_enum(TenantId, self.tenant_id)
        self.risk_tier = _coerce_enum(RiskTier, self.risk_tier)
        self.schema_version = _coerce_enum(SchemaVersion, self.schema_version)
        self.dependencies_revealed = [
            _coerce_enum(DependencyProfile, d) for d in (self.dependencies_revealed or [])
        ]

    def to_dict(self) -> Dict[str, Any]:
        return _enum_as_value(asdict(self))


@dataclass
class TenantHiddenState:
    """Hidden per-tenant state never directly exposed to policy."""

    tenant_id: TenantId
    has_legacy_export_dependency: bool = False
    export_job_health_internal: SignalLevel = SignalLevel.HEALTHY
    # Full dependency set (server-authoritative; visible field is a revealed copy).
    dependencies: List[DependencyProfile] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tenant_id = _coerce_enum(TenantId, self.tenant_id)
        self.export_job_health_internal = _coerce_enum(SignalLevel, self.export_job_health_internal)
        self.dependencies = [
            _coerce_enum(DependencyProfile, d) for d in (self.dependencies or [])
        ]

    def to_dict(self) -> Dict[str, Any]:
        return _enum_as_value(asdict(self))


@dataclass
class EpisodeConfig:
    """Per-episode configuration and deterministic setup."""

    seed: Optional[int] = None
    difficulty: DifficultyLevel = DifficultyLevel.EASY
    max_steps: int = 12
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.difficulty = _coerce_enum(DifficultyLevel, self.difficulty)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return _enum_as_value(asdict(self))


@dataclass
class EpisodeStats:
    """Mutable counters for episode analytics and summaries."""

    steps_taken: int = 0
    invalid_actions: int = 0
    inspections: int = 0
    defers: int = 0
    rollbacks: int = 0
    finalizations: int = 0
    cumulative_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeGuardAction:
    """OpenEnv-facing action payload.

    Uses ActionType enums to avoid magic string action names.
    """

    action_type: ActionType
    target_tenant: Optional[TenantId] = None
    arguments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_type = _coerce_enum(ActionType, self.action_type)
        if self.target_tenant is not None:
            self.target_tenant = _coerce_enum(TenantId, self.target_tenant)

    @property
    def name(self) -> str:
        """Compatibility alias for older call sites expecting `.name`."""
        return self.action_type.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_tenant": self.target_tenant.value if self.target_tenant else None,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChangeGuardAction":
        return cls(
            action_type=_coerce_enum(ActionType, data["action_type"]),
            target_tenant=(
                _coerce_enum(TenantId, data["target_tenant"])
                if data.get("target_tenant")
                else None
            ),
            arguments=dict(data.get("arguments", {})),
        )


@dataclass
class RewardBreakdown:
    """Machine-readable reward decomposition for debugging and training."""

    progress_reward: float = 0.0
    inspection_reward: float = 0.0
    safety_reward: float = 0.0
    invalid_action_penalty: float = 0.0
    loop_penalty: float = 0.0
    terminal_bonus_or_penalty: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifierFlags:
    """Verifier outputs attached to transitions and summaries."""

    verdict: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    safe_completion: bool = False
    safe_abort: bool = False
    outage: bool = False
    integrity_ok: bool = True
    approval_violation: bool = False
    step_budget_respected: bool = True

    def __post_init__(self) -> None:
        self.verdict = _coerce_enum(VerifierVerdict, self.verdict)

    def to_dict(self) -> Dict[str, Any]:
        return _enum_as_value(asdict(self))


@dataclass
class ChangeGuardObservation:
    """OpenEnv observation payload.

    Includes both concise model-facing summary text and machine-readable fields.
    Hidden tenant state is intentionally excluded.
    """

    stage: RolloutStage = RolloutStage.PLAN
    tenants_visible: Dict[TenantId, TenantVisibleState] = field(default_factory=dict)
    approval_granted_c: bool = False
    rollback_window_open: bool = True
    service_health_score: float = 1.0
    export_job_signal_c: SignalLevel = SignalLevel.UNKNOWN
    compat_report_signal: SignalLevel = SignalLevel.UNKNOWN
    risk_hint_level: RiskHintLevel = RiskHintLevel.UNKNOWN
    steps_remaining: int = 12
    legal_actions: List[ActionType] = field(default_factory=list)
    summary_text: str = "Tenant-safe rollout in progress."
    # V2 migration plan (always visible — it's the public changelog).
    schema_v2_diff: List[DiffOpKind] = field(default_factory=list)
    # Global mitigation flags the agent has enabled for this migration.
    mitigations_applied: List[MigrationMitigation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = _coerce_enum(RolloutStage, self.stage)
        self.export_job_signal_c = _coerce_enum(SignalLevel, self.export_job_signal_c)
        self.compat_report_signal = _coerce_enum(SignalLevel, self.compat_report_signal)
        self.risk_hint_level = _coerce_enum(RiskHintLevel, self.risk_hint_level)
        if not (0.0 <= self.service_health_score <= 1.0):
            raise ValueError("service_health_score must be in [0, 1]")
        if self.steps_remaining < 0:
            raise ValueError("steps_remaining must be >= 0")

        normalized: Dict[TenantId, TenantVisibleState] = {}
        for tenant_key, tenant_state in self.tenants_visible.items():
            key = _coerce_enum(TenantId, tenant_key)
            if isinstance(tenant_state, TenantVisibleState):
                normalized[key] = tenant_state
            else:
                normalized[key] = TenantVisibleState(**dict(tenant_state))
        self.tenants_visible = normalized

        self.legal_actions = [_coerce_enum(ActionType, action) for action in self.legal_actions]
        self.schema_v2_diff = [_coerce_enum(DiffOpKind, op) for op in self.schema_v2_diff]
        self.mitigations_applied = [
            _coerce_enum(MigrationMitigation, m) for m in self.mitigations_applied
        ]

    @property
    def phase(self) -> str:
        """Compatibility alias for older call sites expecting `obs.phase` string."""
        return self.stage.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "tenants_visible": {
                tenant_id.value: tenant_state.to_dict()
                for tenant_id, tenant_state in self.tenants_visible.items()
            },
            "approval_granted_c": self.approval_granted_c,
            "rollback_window_open": self.rollback_window_open,
            "service_health_score": self.service_health_score,
            "export_job_signal_c": self.export_job_signal_c.value,
            "compat_report_signal": self.compat_report_signal.value,
            "risk_hint_level": self.risk_hint_level.value,
            "steps_remaining": self.steps_remaining,
            "legal_actions": [action.value for action in self.legal_actions],
            "summary_text": self.summary_text,
            "schema_v2_diff": [op.value for op in self.schema_v2_diff],
            "mitigations_applied": [m.value for m in self.mitigations_applied],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChangeGuardObservation":
        tenants = {
            TenantId(tenant_id): TenantVisibleState(**tenant_data)
            for tenant_id, tenant_data in dict(data.get("tenants_visible", {})).items()
        }
        return cls(
            stage=_coerce_enum(RolloutStage, data.get("stage", RolloutStage.PLAN.value)),
            tenants_visible=tenants,
            approval_granted_c=bool(data.get("approval_granted_c", False)),
            rollback_window_open=bool(data.get("rollback_window_open", True)),
            service_health_score=float(data.get("service_health_score", 1.0)),
            export_job_signal_c=_coerce_enum(
                SignalLevel, data.get("export_job_signal_c", SignalLevel.UNKNOWN.value)
            ),
            compat_report_signal=_coerce_enum(
                SignalLevel, data.get("compat_report_signal", SignalLevel.UNKNOWN.value)
            ),
            risk_hint_level=_coerce_enum(
                RiskHintLevel, data.get("risk_hint_level", RiskHintLevel.UNKNOWN.value)
            ),
            steps_remaining=int(data.get("steps_remaining", 0)),
            legal_actions=[_coerce_enum(ActionType, a) for a in data.get("legal_actions", [])],
            summary_text=str(data.get("summary_text", "")),
            schema_v2_diff=[_coerce_enum(DiffOpKind, op) for op in data.get("schema_v2_diff", [])],
            mitigations_applied=[
                _coerce_enum(MigrationMitigation, m) for m in data.get("mitigations_applied", [])
            ],
        )


@dataclass
class StepResult:
    """Single transition result returned by environment layer."""

    observation: ChangeGuardObservation
    reward_total: float = 0.0
    reward_breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    done: bool = False
    truncated: bool = False
    verifier_flags: VerifierFlags = field(default_factory=VerifierFlags)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward_total": self.reward_total,
            "reward_breakdown": self.reward_breakdown.to_dict(),
            "done": self.done,
            "truncated": self.truncated,
            "verifier_flags": self.verifier_flags.to_dict(),
            "info": self.info,
        }


@dataclass
class EpisodeSummary:
    """Episode-level aggregation for evaluation and reporting."""

    episode_id: Optional[str] = None
    config: EpisodeConfig = field(default_factory=EpisodeConfig)
    stats: EpisodeStats = field(default_factory=EpisodeStats)
    final_verdict: VerifierVerdict = VerifierVerdict.IN_PROGRESS
    final_reward: float = 0.0
    action_trace: List[ActionType] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.final_verdict = _coerce_enum(VerifierVerdict, self.final_verdict)
        self.action_trace = [_coerce_enum(ActionType, action) for action in self.action_trace]

    @property
    def seed(self) -> Optional[int]:
        """Compatibility alias for older call sites expecting `summary.seed`."""
        return self.config.seed

    @property
    def difficulty(self) -> str:
        """Compatibility alias for older call sites expecting `summary.difficulty`."""
        return self.config.difficulty.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "final_verdict": self.final_verdict.value,
            "final_reward": self.final_reward,
            "action_trace": [action.value for action in self.action_trace],
        }


# Backward-compatible aliases for older stubs/imports.
Observation = ChangeGuardObservation
Action = ChangeGuardAction
