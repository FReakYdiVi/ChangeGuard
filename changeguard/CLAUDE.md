# ChangeGuard

OpenEnv-style RL environment for tenant-aware schema upgrades (invoice V1 -> V2 across tenants A/B/C with hidden legacy dependency on C) plus a thin TRL GRPO training wrapper.

## Layout (flat; project dir IS the python package)
```
<repo-root>/
  .venv/                         uv-managed venv.
  changeguard/                   Project dir AND python package (has __init__.py).
    pyproject.toml, Dockerfile, openenv.yaml, README, CLAUDE.
    __init__.py, models.py, client.py, internal_state.py
    server/                      OpenEnv HTTP app + ChangeGuardEnvironment engine (ground truth).
      app.py                     HTTP server + session manager.
      changeguard_environment.py Authoritative env logic: transitions, rewards, verifier.
    training/                    TRL GRPO wrapper.
      train_grpo.py              GRPO entrypoint; defines SEED_PACKS.
      changeguard_tool_env.py    Adapter between model tool calls and OpenEnv server.
      evaluate_policy.py         Fixed-seed before/after evaluation.
    tests/                       28 tests: contract, determinism, verifier, anti-hacking, reward-hacking.
    docs/                        SPEC.md (canonical). archive/ holds deprecated design docs.
    scripts/                     One-command pipeline helpers + plot_training.py.
    artifacts/                   Training outputs (gitignored).
```

## Common Commands (run from repo root or anywhere)
```bash
# One-time setup
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e "./changeguard[training]"

# Run OpenEnv server
.venv/bin/python -m changeguard.server.app

# Tests
.venv/bin/python -m unittest discover -s changeguard/tests

# GRPO dry-run (cheap, no model weights)
.venv/bin/python -m changeguard.training.train_grpo --dry-run --seed-pack smoke

# Real GRPO run (T4 16GB, LoRA, Qwen2.5-1.5B default)
.venv/bin/python -m changeguard.training.train_grpo --no-dry-run --lora --seed-pack short_train --max-steps 10

# Fixed-seed evaluation
.venv/bin/python -m changeguard.training.evaluate_policy --seed-pack final_demo --candidate trained_like
```

## Conventions
- **Determinism**: `(seed, difficulty, scenario_id)` fully determines world, transitions, and verifier verdict. Preserve this in any change to env logic.
- **Hidden state is sacred**: `has_legacy_export_dependency`, `rollback_deadline_step`, `b_has_hidden_risk`, and raw `export_job_health_internal` must never be returned by observations, summaries, tool responses, or the client. If you add a field, prove it is post-inspection-derived.
- **Reward logic lives ONLY in the server**: `changeguard/server/changeguard_environment.py` is the single source of truth for rewards and verifier verdicts. The training wrapper consumes numeric fields; it never recomputes.
- **One canonical spec**: `docs/SPEC.md`. Historical design docs live in `docs/archive/` and are not authoritative.

## Where to Look First
- Ground truth for rewards, transitions, and verdicts: `changeguard/server/changeguard_environment.py`.
- Shared schemas / enums: `changeguard/models.py`.
- Training loop + seed packs: `changeguard/training/train_grpo.py`.
- Spec: `docs/SPEC.md`.
