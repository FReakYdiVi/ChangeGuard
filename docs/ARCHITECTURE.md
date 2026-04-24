# ChangeGuard Architecture (OpenEnv Native + TRL Wrapper)

This document maps `SPEC_V1`, `FORMAL_ENV_SPEC`, and `POLICIES_AND_INVARIANTS` to a two-layer implementation:
- Layer A: **native OpenEnv environment server** (authoritative environment logic)
- Layer B: **thin TRL tool-wrapper** (GRPO-facing adapter)

No training logic is duplicated across layers.

## 1) Exact File Tree

```text
changeguard/
  __init__.py
  models.py
  client.py
  openenv.yaml
  server/
    __init__.py
    app.py
    changeguard_environment.py
training/
  changeguard_tool_env.py
  train_grpo.py
tests/
docs/
  SPEC_V1.md
  FORMAL_ENV_SPEC.md
  POLICIES_AND_INVARIANTS.md
  TRAINING_STRATEGY.md
  ARCHITECTURE.md
```

## 2) Per-File Responsibilities

### `changeguard/__init__.py`
- Public package exports.
- Re-export stable types (`Observation`, `Action`, `StepResult`, `EpisodeSummary`) for users/tests.

### `changeguard/models.py`
- Shared typed data models used by server + client + training wrapper.
- Defines:
  - observation schema,
  - action schema,
  - reward breakdown,
  - verifier flags,
  - episode summary.
- Single source of truth for serialization keys.

### `changeguard/client.py`
- Minimal Python client for local dev and training integration.
- Handles server transport calls and model (de)serialization.
- No environment logic.

### `changeguard/openenv.yaml`
- OpenEnv manifest: environment name, version, tool surface, endpoint bindings.
- Declares **meaningful tool names** and schemas.

### `changeguard/server/__init__.py`
- Server package exports.

### `changeguard/server/app.py`
- Web/OpenEnv server entrypoint.
- Wires routes/tools to environment instance methods.
- Handles request validation, response formatting, and health/version endpoints.

### `changeguard/server/changeguard_environment.py`
- Authoritative environment engine.
- Implements state machine, transitions, verifier logic, reward computation, and summaries.
- No trainer-specific code.

### `training/changeguard_tool_env.py`
- Thin adapter that exposes environment as TRL-compatible tool environment.
- Converts model outputs to valid tool calls and converts server responses into GRPO-ready records.
- No domain logic duplication.

### `training/train_grpo.py`
- GRPO orchestration script.
- Configures model, curriculum seed sets, rollout loop, logging, checkpoint evaluation.
- Reads reward/verifier fields from wrapper outputs.

### `tests/`
- Contract tests for schemas and tool API.
- Deterministic seed replay tests.
- Verifier and invariant tests.

### `docs/`
- Design and training specifications.

## 3) Native OpenEnv Classes and Methods

## `ChangeGuardEnvironment` (in `changeguard/server/changeguard_environment.py`)

Authoritative class owning episode state and transition semantics.

### Core lifecycle methods
- `reset(seed: int | None, difficulty: str = "easy") -> ResetResult`
  - Initializes episode with deterministic seed.
  - Returns initial observation + metadata.
- `state() -> Observation`
  - Returns current public observation only (no latent leakage).
- `apply_action(action: Action) -> StepResult`
  - Applies one validated action and returns new observation, reward, done, and verifier fields.

### Meaningful OpenEnv tools (no generic `step(action)` tool name)
Expose tools at server layer, each mapped to `apply_action` internally:
- `inspect_tenant_profile()`
- `inspect_compatibility_report()`
- `inspect_export_job_status()`
- `canary_rollout_tenant_a()`
- `expand_rollout_tenant_b()`
- `pause_rollout()`
- `enable_compat_mode_tenant_c()`
- `request_approval_tenant_c()`
- `finalize_upgrade()`
- `rollback_upgrade()`

### Episode/query methods
- `get_episode_summary() -> EpisodeSummary`
- `get_reward_breakdown() -> RewardBreakdown`
- `get_verifier_flags() -> VerifierFlags`

## `ChangeGuardServerApp` (in `changeguard/server/app.py`)
- `create_app()`
- route/tool registration methods
- health and metadata endpoints (`/health`, `/version`, `/schema`)

## 4) TRL Wrapper Classes and Methods

## `ChangeGuardToolEnv` (in `training/changeguard_tool_env.py`)
Thin bridge between TRL/GRPO and OpenEnv server tools.

### Responsibilities
- Convert model generation -> selected tool call + args.
- Invoke OpenEnv tool via `changeguard.client`.
- Return structured rollout item for GRPO.

### Methods
- `reset(seed: int, difficulty: str) -> dict`
  - Calls server `reset`, returns trainer-friendly observation payload.
- `available_tools() -> list[ToolSpec]`
  - Returns tool definitions with names + arg schemas.
- `call_tool(tool_name: str, tool_args: dict | None) -> dict`
  - Executes one action tool and returns step payload.
- `get_reward(step_payload: dict) -> float`
  - Reads numeric reward from server response (no local reward recomputation).
- `get_done(step_payload: dict) -> bool`
  - Reads terminal status.
- `get_episode_summary() -> dict`
  - Pulls summary from server at episode end.
- `to_grpo_record(...) -> dict`
  - Packs `obs/action/reward/done/verifier_flags` for logs.

## `training/train_grpo.py`
- `load_train_config()`
- `build_environment_factory()`
- `run_rollouts()`
- `evaluate_checkpoint(seed_set)`
- `save_training_artifacts()`

## 5) Shared Reward/State Structures

Defined centrally in `changeguard/models.py` and reused everywhere.

### `Observation`
- `phase`
- `tenant_versions` (`A/B/C`)
- `compat_mode_enabled_c`
- `approval_required_c`
- `approval_granted_c`
- `rollback_window_open`
- `service_health_score`
- `export_job_signal_c`
- `compat_report_signal`
- `risk_hint_level`
- `last_action`
- `steps_remaining`

### `Action`
- `name` (one of the meaningful tool names)
- `arguments` (empty for most v1 actions)

### `RewardBreakdown`
- `progress_reward`
- `inspection_reward`
- `safety_reward`
- `invalid_action_penalty`
- `loop_penalty`
- `terminal_bonus_or_penalty`
- `total_reward`

### `VerifierFlags`
- `verdict`
- `safe_completion`
- `safe_abort`
- `outage`
- `integrity_ok`
- `approval_violation`
- `step_budget_respected`

### `StepResult`
- `observation`
- `reward_total`
- `reward_breakdown`
- `done`
- `truncated`
- `verifier_flags`
- `info`

### `EpisodeSummary`
- `episode_id`
- `seed`
- `difficulty`
- `steps_used`
- `action_trace`
- `final_verdict`
- `final_reward`
- aggregate counters (`invalid_actions`, `defers`, `inspections`)

## 6) End-to-End Connection (Local Dev -> Server -> Training)

## Local development flow
1. Start server from `changeguard/server/app.py`.
2. Use `changeguard/client.py` in notebooks/scripts to run deterministic episodes.
3. Validate invariants and verifier outputs manually and via `tests/`.

## OpenEnv server flow
1. `openenv.yaml` registers environment + named tools.
2. Tool call enters `app.py` route.
3. Route maps to `ChangeGuardEnvironment` method.
4. Engine returns `StepResult` including reward breakdown + verifier flags.

## GRPO training flow
1. `train_grpo.py` builds `ChangeGuardToolEnv` via environment factory.
2. Model chooses a tool from `available_tools()`.
3. Wrapper calls server tool through `client.py`.
4. Wrapper extracts reward/done/verifier and emits GRPO log record.
5. Trainer updates policy.
6. Evaluation uses fixed seed sets and compares before/after `EpisodeSummary` + verdict rates.

## Separation guarantees
- Environment logic lives only in `changeguard/server/changeguard_environment.py`.
- Training logic lives only under `training/`.
- Shared contracts live in `changeguard/models.py`.
- Reward/verifier are computed once (server-side), consumed everywhere else.
