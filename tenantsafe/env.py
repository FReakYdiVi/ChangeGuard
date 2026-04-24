from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

ACTIONS = [
    "inspect_tenant_profile",
    "inspect_compatibility_report",
    "inspect_service_logs",
    "inspect_export_job_status",
    "canary_rollout_tenant_a",
    "expand_rollout_tenant_b",
    "pause_rollout",
    "enable_compat_mode_tenant_c",
    "rollback_tenant",
    "request_approval_tenant_c",
    "finalize_upgrade",
]


@dataclass
class TenantState:
    schema_version: str = "V1"
    compat_mode: bool = False


class TenantSafeEnv:
    """OpenEnv-style minimal environment for staged tenant migration safety.

    API style:
      - reset(seed=None) -> (observation, info)
      - step(action) -> (observation, reward, done, truncated, info)
    """

    def __init__(self, max_steps: int = 12) -> None:
        self.max_steps = max_steps
        self.rng = random.Random()
        self._initialized = False

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)

        self.step_count = 0
        self.done = False
        self.truncated = False
        self.phase = "plan"
        self.approval_c = False
        self.outage = False
        self.data_integrity_ok = True

        self.hidden = {
            "c_requires_compat": True,
            "rollback_window_close_step": self.rng.choice([8, 9, 10]),
        }

        self.tenants: Dict[str, TenantState] = {
            "A": TenantState(),
            "B": TenantState(),
            "C": TenantState(),
        }

        self.inspected = {
            "profile": False,
            "compat": False,
            "logs": False,
            "export": False,
        }

        self.last_action = None
        self._initialized = True
        return self._observation(), {"scenario": "default"}

    def _rollback_window_open(self) -> bool:
        return self.step_count < self.hidden["rollback_window_close_step"]

    def _observation(self) -> Dict[str, Any]:
        compat_score = 0.5
        if self.tenants["A"].schema_version == "V2":
            compat_score += 0.15
        if self.tenants["B"].schema_version == "V2":
            compat_score += 0.15
        if self.tenants["C"].compat_mode:
            compat_score += 0.2

        data_mismatch = 0.0
        if self.tenants["C"].schema_version == "V2" and not self.tenants["C"].compat_mode:
            data_mismatch = 0.7

        canary_error = 0.0
        if self.tenants["A"].schema_version == "V2" and not self.tenants["C"].compat_mode:
            canary_error = 0.1
        if self.outage:
            canary_error = 0.9

        risk_hint = {
            "profile": "known" if self.inspected["profile"] else "unknown",
            "compatibility": "risky_c" if self.inspected["compat"] else "unknown",
            "export": "legacy_dependency" if self.inspected["export"] else "unknown",
            "logs": ("errors_present" if self.outage else "clear") if self.inspected["logs"] else "unknown",
        }

        return {
            "phase": self.phase,
            "consumer_compat_score": round(min(1.0, compat_score), 3),
            "backfill_progress": 1.0 if self.phase in {"cutover", "done", "failed"} else 0.0,
            "data_mismatch_rate": data_mismatch,
            "canary_error_rate": canary_error,
            "rollback_window_open": self._rollback_window_open(),
            "approval_required": True,
            "approval_c": self.approval_c,
            "steps_remaining": self.max_steps - self.step_count,
            "last_action": self.last_action,
            "risk_hint": risk_hint,
            "tenant_versions": {k: v.schema_version for k, v in self.tenants.items()},
            "compat_mode_c": self.tenants["C"].compat_mode,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        self.step_count += 1
        self.last_action = action
        reward = 0.0
        info: Dict[str, Any] = {"invalid_transition": False}

        if action == "inspect_tenant_profile":
            self.inspected["profile"] = True
            reward += 0.5

        elif action == "inspect_compatibility_report":
            self.inspected["compat"] = True
            reward += 0.5

        elif action == "inspect_service_logs":
            self.inspected["logs"] = True
            reward += 0.25

        elif action == "inspect_export_job_status":
            self.inspected["export"] = True
            reward += 0.5

        elif action == "canary_rollout_tenant_a":
            if self.phase in {"plan", "compat"}:
                self.tenants["A"].schema_version = "V2"
                self.phase = "canary"
                reward += 1.0
            else:
                reward -= 1.0
                info["invalid_transition"] = True

        elif action == "expand_rollout_tenant_b":
            if self.phase in {"canary", "rollout"}:
                self.tenants["B"].schema_version = "V2"
                self.phase = "rollout"
                reward += 1.0
            else:
                reward -= 1.0
                info["invalid_transition"] = True

        elif action == "pause_rollout":
            reward += 0.1

        elif action == "enable_compat_mode_tenant_c":
            self.tenants["C"].compat_mode = True
            self.phase = "compat"
            reward += 1.0

        elif action == "rollback_tenant":
            if self._rollback_window_open():
                for tenant in self.tenants.values():
                    tenant.schema_version = "V1"
                self.phase = "plan"
                reward += 0.5
            else:
                reward -= 3.0
                self.outage = True

        elif action == "request_approval_tenant_c":
            # Approval is meaningful only when A and B progressed.
            if self.tenants["A"].schema_version == "V2" and self.tenants["B"].schema_version == "V2":
                self.approval_c = True
                reward += 1.0
            else:
                reward -= 1.0
                info["invalid_transition"] = True

        elif action == "finalize_upgrade":
            self.phase = "cutover"
            safe_finalize = (
                self.approval_c
                and self.tenants["C"].compat_mode
                and self.tenants["A"].schema_version == "V2"
                and self.tenants["B"].schema_version == "V2"
            )
            if safe_finalize:
                self.tenants["C"].schema_version = "V2"
                self.phase = "done"
                self.done = True
                reward += 10.0
            else:
                self.phase = "failed"
                self.done = True
                self.outage = True
                self.data_integrity_ok = False
                reward -= 10.0

        # Ongoing safety penalties.
        if not self.done:
            mismatch = self._observation()["data_mismatch_rate"]
            canary_error = self._observation()["canary_error_rate"]
            if mismatch > 0.3:
                reward -= 2.0
            if canary_error > 0.3:
                reward -= 2.0

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            self.truncated = True
            reward -= 5.0

        obs = self._observation()
        info.update(
            {
                "safe_completion": self.done and self.phase == "done" and self.data_integrity_ok,
                "outage": self.outage,
            }
        )
        return obs, reward, self.done, self.truncated, info
