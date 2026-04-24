from __future__ import annotations

from typing import Dict, Any


def safe_policy(obs: Dict[str, Any]) -> str:
    if obs["risk_hint"]["profile"] == "unknown":
        return "inspect_tenant_profile"
    if obs["risk_hint"]["compatibility"] == "unknown":
        return "inspect_compatibility_report"
    if obs["risk_hint"]["export"] == "unknown":
        return "inspect_export_job_status"
    if obs["tenant_versions"]["A"] == "V1":
        return "canary_rollout_tenant_a"
    if obs["tenant_versions"]["B"] == "V1":
        return "expand_rollout_tenant_b"
    if not obs["approval_c"]:
        return "request_approval_tenant_c"
    if obs["risk_hint"]["logs"] == "unknown":
        return "inspect_service_logs"
    if not obs["compat_mode_c"]:
        return "enable_compat_mode_tenant_c"
    return "finalize_upgrade"


def risky_policy(obs: Dict[str, Any]) -> str:
    if obs["tenant_versions"]["A"] == "V1":
        return "canary_rollout_tenant_a"
    if obs["tenant_versions"]["B"] == "V1":
        return "expand_rollout_tenant_b"
    return "finalize_upgrade"
