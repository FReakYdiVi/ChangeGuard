# ChangeGuard: Tenant-Aware Upgrade Commander — TRAINING_STRATEGY

## 1) Recommended Starting Model Behavior Assumptions
- The model can follow short structured instructions and output one valid action token/string per step.
- The model can remember 2-4 recent facts (e.g., whether approval was already granted).
- The model is not initially good at long-horizon safety planning.
- The model may over-prefer terminal actions (`finalize`) unless penalties are strong and curriculum is staged.
- The model needs explicit action schema constraints to avoid invalid outputs.

## 2) Whether SFT Is Needed
- Recommendation: **No full SFT for v1**.
- Use only a **light formatting/tool-use warm start** if needed.
- Warm start target: 50-200 short traces only for
  - valid action formatting,
  - legal action vocabulary,
  - basic phase ordering (`A -> B -> compat/approval -> finalize or rollback`).
- Stop warm start once invalid action rate is low enough (for example <10-15% on easy seeds).
- Main learning should come from GRPO rollouts and verifier-driven rewards.

## 3) Minimum Curriculum for Non-Zero Reward
- Stage 0 (format sanity): deterministic easy seeds, short episodes, reward only for valid action + phase progress.
- Stage 1 (first success): easy difficulty only; make safe path discoverable with low ambiguity.
- Stage 2 (stability): easy + medium mix; keep positive shaping for inspections and safe defer.
- Stage 3 (real objective): medium dominant + small hard fraction; terminal safety rewards/penalties dominate.
- Promotion rule: move to next stage only after non-zero success rate is consistent (for example 20-30 consecutive eval episodes with at least one safe completion and low invalid-action rate).

## 4) Easiest Initial Scenario That Still Proves the Concept
- Difficulty: `easy`.
- Deterministic setup:
  - `max_steps=12`,
  - late rollback close,
  - clean signals after inspections,
  - approval granted immediately after B rollout.
- Required hidden-risk proof still present:
  - tenant C legacy dependency is active,
  - finalize without compat/approval still fails deterministically.
- Canonical successful trajectory:
  - `inspect_compatibility_report`
  - `inspect_export_job_status`
  - `canary_rollout_tenant_a`
  - `expand_rollout_tenant_b`
  - `enable_compat_mode_tenant_c`
  - `request_approval_tenant_c`
  - `finalize_or_rollback(finalize)`

## 5) Failure Modes Likely to Cause Zero Reward
- Premature finalize loop: policy repeatedly finalizes before approval/compat, causing immediate terminal failures.
- Invalid-action churn: model emits off-policy or phase-invalid actions and never reaches progress states.
- Inspection-only stalling: policy keeps inspecting and times out.
- Pause-spam stalling: policy avoids risk forever and hits timeout.
- Late rollback attempts: policy chooses rollback only after rollback window closes.
- Hidden-state leakage assumptions: policy relies on non-existent direct latent flags and fails when not exposed.

### OpenEnv/TRL Mapping (Required Architecture Split)
- OpenEnv environment core should expose canonical methods:
  - `reset()` initializes seeded episode state,
  - `step(action)` applies transition and writes reward-relevant fields,
  - `state()` (or equivalent getter) returns current public observation payload.
- TRL `environment_factory` tool-path should expose action methods as callable tools or validated action strings.
- Reward consumption in trainer should read environment output fields, not free-form text judgment.
- Verifier fields should be first-class in transitions/logs (`verdict`, `safe_completion`, `approval_violation`, `integrity_ok`, `outage`).
- GRPO logs should store per-step: `seed`, `obs`, `action`, `reward`, `done`, `verifier_flags`.

## 6) How to Detect If the Task Is Too Hard
- Success rate remains at or near 0 across multiple curriculum stages.
- Invalid action rate stays high despite warm start.
- Average episode length trends to timeout ceiling.
- Reward variance is high but mean reward does not improve over checkpoints.
- Policy collapses to one degenerate action pattern (`finalize` spam or `pause` spam).
- Easy-seed performance does not exceed random/scripted baseline.

## 7) What To Simplify First If Learning Stalls
- Keep task fixed; simplify training knobs in this order:
  1. Reduce observation ambiguity on easy/medium (clearer inspection signals).
  2. Increase terminal penalty contrast (unsafe finalize and timeout clearly worse).
  3. Increase positive shaping for safe intermediate milestones.
  4. Tighten action parser/decoder constraints to reduce invalid outputs.
  5. Temporarily shorten action set usage by masking low-value actions early (keep spec actions intact, only training-time masking).
  6. Increase easy-seed ratio before reintroducing medium/hard.

## Tiny Deterministic Seed Sets

### Local Smoke Test (very fast)
- Seeds: `[101, 102, 103]`
- Mix: all `easy`
- Goal: verify loop works, non-zero reward appears, no parser failures.

### Short Training Run (quick iteration)
- Seeds: `[201, 202, 203, 204, 205, 206, 207, 208]`
- Mix: `easy x5`, `medium x3`
- Goal: confirm upward trend vs random/risky baseline within limited steps.

### Final Before/After Demo Set (stable showcase)
- Seeds: `[301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312]`
- Mix: `easy x4`, `medium x5`, `hard x3`
- Goal: show reproducible improvement and safety-verdict gains on fixed public seeds.

## Bottom Line
- Start without full SFT.
- Use only light formatting warm start if invalid actions block progress.
- Win condition for v1 training is visible improvement in deterministic verifier outcomes, not language fluency.
