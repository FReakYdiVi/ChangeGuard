# ChangeGuard: Tenant-Aware Upgrade Commander — SPEC_V1

## 1) Problem Statement
Enterprise SaaS teams must roll out schema upgrades safely across tenants with different risk profiles. In v1, we model one concrete high-value case: invoice schema migration from `V1` to `V2` across exactly three tenants (`A`, `B`, `C`). Tenant `C` has a hidden legacy export dependency and requires approval before irreversible migration completion. The agent must sequence inspection, staged rollout, risk mitigation, approval, and finalize/rollback decisions within a short horizon (10–12 steps), maximizing safe completion and minimizing outages/data issues.

## 2) Why This Is a Valid RL Task
This is a sequential decision-making problem with delayed consequences and partial observability:
- Actions taken early (inspect vs rush rollout) change later failure probabilities.
- Hidden dependency in tenant `C` is not directly given at start; the agent must gather evidence.
- Irreversible step (finalize) creates long-horizon credit assignment.
- Optimal behavior requires balancing speed and safety under step limits.
This structure maps naturally to an episodic MDP/POMDP and is appropriate for policy optimization (e.g., GRPO).

## 3) Why Outputs Are Objectively Verifiable
Episode outcomes are judged by deterministic programmatic checks, not subjective interpretation:
- `safe_completion` (boolean): true only if migration reaches done state without integrity or outage violations.
- `outage` (boolean): true if unsafe finalize/late failure occurs.
- `integrity_ok` (boolean): true only if compatibility constraints were respected.
- `approval_violation` (boolean): true if finalize attempted without required tenant `C` approval.
- `step_budget_respected` (boolean): true if episode ends within configured horizon.
Because transitions are seed-deterministic, policy performance is reproducible and directly comparable.

## 4) Why the Task Is Not Too Hard for RL
Complexity is deliberately constrained:
- Exactly 3 tenants, 1 migration type, 1 hidden dependency, 1 approval gate.
- Horizon capped at 10–12 steps.
- Action space capped at 8–9 meaningful actions.
- Deterministic seeds reduce variance for stable training signals.
- Dense intermediate rewards can be added for safe progress while preserving terminal safety objective.
This keeps learning tractable while still requiring non-trivial planning.

## 5) Novelty vs Generic SRE / Incident Response
Generic incident-response tasks are reactive firefighting after failure signals. Tenant-Aware Upgrade Commander is proactive change governance:
- Emphasizes staged tenant rollout under hidden compatibility risk.
- Includes enterprise approval workflow before irreversible operations.
- Explicitly models migration reversibility windows and policy discipline.
- Measures correctness/safety of change execution, not just MTTR.
This gives a distinct “safe evolution of production state” angle beyond standard incident environments.

## 6) Exact v1 Boundaries
- Tenants: fixed to `A` (low risk), `B` (medium risk), `C` (high risk).
- Change type: only `invoice schema V1 -> V2`.
- Hidden dependency: only tenant `C` legacy export integration.
- Approval: only tenant `C` pre-finalization approval gate.
- Horizon: max 10–12 steps.
- Determinism: seeded deterministic transitions required.
- Action space (9 actions):
  1. `inspect_tenant_profile`
  2. `inspect_compatibility_report`
  3. `inspect_export_job_status`
  4. `canary_rollout_tenant_a`
  5. `expand_rollout_tenant_b`
  6. `pause_rollout`
  7. `enable_compat_mode_tenant_c`
  8. `request_approval_tenant_c`
  9. `finalize_or_rollback` (single terminal control action parameterized by mode)
- Evaluator: objective verifier returns episode verdict + metrics.
- Training artifact requirement: GRPO-ready per-step logs (`obs`, `action`, `reward`, `done`, `verifier_flags`).

## 7) Explicit Non-Goals for v1
1. Multi-change orchestration (no multiple schema changes, no feature flags matrix).
2. Multi-tenant scale realism beyond 3 tenants (no dynamic tenant counts, no tenant graph discovery).
3. Real infrastructure integration (no live DB, Kubernetes, or cloud APIs; simulator only).

## 8) Why This Matches OpenEnv Hackathon Judging Well
This v1 is judge-friendly because it is easy to understand (“safe tenant upgrade under hidden risk”), objectively scoreable, and demonstrably trainable within hackathon time. It showcases real enterprise relevance (approval gates, legacy dependency risk), has clear long-horizon RL structure (inspect -> stage -> mitigate -> approve -> finalize), and produces reproducible benchmark outcomes via deterministic seeds. The project can show visible policy improvement over episodes with simple metrics, which aligns strongly with OpenEnv priorities around clear environment design, measurable learning progress, and practical usefulness.

## Recommendation
**Build now.**
