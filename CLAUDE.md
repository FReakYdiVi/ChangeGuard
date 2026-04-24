# ChangeGuard

OpenEnv-style RL environment for tenant-aware schema upgrades (invoice V1 -> V2 across tenants A/B/C with hidden legacy dependency on C) plus a thin TRL GRPO training wrapper.

## Layout
```
changeguard/           Authoritative HTTP OpenEnv environment.
  server/              FastAPI-style app + ChangeGuardEnvironment engine (ground truth).
  models.py            Shared typed data models (Observation, Action, RewardBreakdown, ...).
  internal_state.py    Runtime state; never exposed via observations.
  client.py            Minimal Python client (transport only, no env logic).
  openenv.yaml         OpenEnv manifest.
training/              TRL GRPO wrapper. Reads reward/verifier from server; never recomputes.
  train_grpo.py        GRPO entrypoint; defines SEED_PACKS.
  changeguard_tool_env.py   Adapter between model tool calls and OpenEnv server.
  evaluate_policy.py   Fixed-seed before/after evaluation.
tests/                 Contract, determinism, verifier, anti-hacking tests.
docs/                  SPEC.md (canonical). archive/ holds deprecated design docs.
scripts/               One-command pipeline helpers.
```

## Common Commands
```bash
# Environment setup (uv-based; Python 3.11)
uv venv --python 3.11
uv sync --extra training

# Run OpenEnv server
uv run python -m changeguard.server.app

# Tests
uv run python -m unittest discover -s tests

# GRPO dry-run (cheap, no model weights)
uv run python -m training.train_grpo --dry-run --seed-pack smoke

# Real GRPO run (T4 16GB, LoRA)
uv run python -m training.train_grpo --no-dry-run --lora --seed-pack smoke --max-steps 4

# Fixed-seed evaluation
uv run python -m training.evaluate_policy --seed-pack final_demo --candidate trained_like
```

## Conventions
- **Determinism**: `(seed, difficulty, scenario_id)` fully determines world, transitions, and verifier verdict. Preserve this in any change to env logic.
- **Hidden state is sacred**: `has_legacy_export_dependency`, `rollback_deadline_step`, `b_has_hidden_risk`, and raw `export_job_health_internal` must never be returned by observations, summaries, tool responses, or the client. If you add a field, prove it is post-inspection-derived.
- **Reward logic lives ONLY in the server**: `changeguard/server/changeguard_environment.py` is the single source of truth for rewards and verifier verdicts. The training wrapper consumes numeric fields; it never recomputes.
- **One canonical spec**: `docs/SPEC.md`. Historical design docs live in `docs/archive/` and are not authoritative.

## Where to Look First
- Ground truth for rewards, transitions, and verdicts: `changeguard/server/changeguard_environment.py`.
- Shared schemas / enums: `changeguard/models.py`.
- Training loop + seed packs: `training/train_grpo.py`.
- Spec: `docs/SPEC.md`.
