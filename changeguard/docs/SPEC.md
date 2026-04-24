# ChangeGuard SPEC

## Problem
Tenant-aware rollout of an invoice schema migration (V1 -> V2) across three tenants `A`, `B`, `C`, where `C` has a hidden legacy export dependency requiring compatibility mode plus explicit approval before finalize. The agent must sequence inspection, staged rollout, mitigation, approval, and finalize-or-rollback within a 10-12 step horizon.

## Action Space

| Action | Effect |
|---|---|
| `inspect_tenant` | Reveal per-tenant risk tier / approval requirement. |
| `inspect_compatibility` | Reveal compatibility signal for tenant C. |
| `inspect_logs` | Reveal export job health signal (tenant C truth, tenant B hidden risk). |
| `canary_upgrade` (A) | Promote tenant A to V2. Must be first rollout. |
| `promote_upgrade` (B) | Promote tenant B to V2. Requires A at V2. |
| `promote_upgrade` (C) | Finalize tenant C to V2. Terminal. Requires A=V2, B=V2, compat enabled, approval granted. |
| `enable_compat_mode` (C) | Enable compatibility mode for tenant C. |
| `request_approval` (C) | Request approval for tenant C finalize. Requires B=V2 (and compat-inspected on hard). |
| `defer_tenant` (C) | Pause; if A=V2, B=V2, C=V1 -> terminal safe partial completion. |
| `rollback_tenant` | Rollback. Safe terminal if rollback window open, else failed. |

## Observation Fields
- `stage` (plan / canary_a / rollout_b / gated_c / finalized / rolled_back / failed)
- `tenants_visible[A|B|C]`: schema version, risk tier, approval_required, compat_mode_enabled
- `approval_granted_c: bool`
- `rollback_window_open: bool`
- `service_health_score: float in [0,1]`
- `export_job_signal_c`, `compat_report_signal`: `unknown` until inspected
- `risk_hint_level`: `unknown|low|high` (derived from inspections + compat state)
- `steps_remaining: int`
- `legal_actions: list[ActionType]`
- `summary_text: str`

Hidden state never exposed: `has_legacy_export_dependency`, `rollback_deadline_step`, `b_has_hidden_risk`, raw `export_job_health_internal` (before inspection).

## Reward Breakdown
Per-step components (summed into `total_reward`):

| Component | Sign / magnitude | Trigger |
|---|---|---|
| `progress_reward` | +1.2 | Successful `canary_upgrade` A or `promote_upgrade` B. |
| `inspection_reward` | +0.35 / +0.45 | First-time inspection (tenant / compat / logs). Repeats earn 0. |
| `safety_reward` | +0.9 compat enable, +0.8 approval, +0.7 defer; -0.4 if B promoted with hidden risk w/o `inspect_logs`; -1.5 on unsafe finalize; +1.0 on safe finalize. |
| `invalid_action_penalty` | -0.2 to -0.9 | Phase-invalid action, wrong target, repeat promote. |
| `loop_penalty` | -0.10 to -0.40 | Repeated inspection or redundant action. |
| `terminal_bonus_or_penalty` | +8 safe finalize; +3 safe rollback; +3 safe partial (defer C after A,B at V2); -8 unsafe finalize; -6 unsafe rollback; -4 timeout. |

## Verifier Verdicts
- `safe_finalized` - all gates satisfied, C at V2, integrity preserved, no outage.
- `safe_rollback` - rollback within open window, OR terminal via safe defer of C.
- `approval_violation` - finalize attempted without `approval_granted_c`.
- `compatibility_violation` - finalize attempted without `compat_mode_enabled` on C.
- `outage_failure` - terminal unsafe side effect.
- `integrity_failure` - data/compat integrity broken (default conservative failure).
- `timeout_failure` - step budget exhausted with no safe terminal.
- `in_progress` - non-terminal transient state.

## Anti-Hacking Invariants
- Hidden state (`has_legacy_export_dependency`, `rollback_deadline_step`, `b_has_hidden_risk`) is never returned by any observation, summary, or tool response.
- Repeated identical inspections are net-negative (no reward + loop penalty).
- Timeout is forced non-positive: if cumulative reward would be > 0 at timeout, terminal penalty offsets it to <= -0.5.
- Unsafe finalize terminal reward (-8 + -1.5) is strictly worse than any safe terminal (safe defer +3, safe rollback +3, safe finalize +8 + +1.0).

## Seed Packs
Defined in `training/train_grpo.py` as `SEED_PACKS`:

| Pack | Seeds | Mix | Purpose |
|---|---|---|---|
| `smoke` | 3 | all easy | Loop sanity; fast non-zero reward check. |
| `short_train` | 8 | 5 easy + 3 medium | Quick iteration; upward trend check. |
| `final_demo` | 12 | 4 easy + 5 medium + 3 hard | Fixed before/after demo set. |

Seed determinism: same `(seed, difficulty, scenario_id)` reproduces identical world signature, transitions, and verifier verdict.
