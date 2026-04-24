# ChangeGuard

Tenant-aware upgrade RL environment (OpenEnv server + TRL GRPO wrapper). Simulates invoice schema migration V1 -> V2 across tenants A/B/C, where C has a hidden legacy dependency requiring compatibility mode and approval before finalize.

See [docs/SPEC.md](docs/SPEC.md) for the environment contract. See [CLAUDE.md](CLAUDE.md) for the repo layout and conventions.

## Setup
```bash
uv venv --python 3.11
uv sync --extra training    # or: uv pip install -e ".[training]"
```

## Run server
```bash
uv run python -m changeguard.server.app
```

## Tests
```bash
uv run python -m unittest discover -s tests
```

## GRPO dry-run
```bash
uv run python -m training.train_grpo --dry-run --seed-pack smoke
```

## GRPO real run (T4 16GB, LoRA)
```bash
uv run python -m training.train_grpo --no-dry-run --lora --seed-pack smoke --max-steps 4
```

## Fixed-seed evaluation
```bash
uv run python -m training.evaluate_policy --seed-pack final_demo --candidate trained_like
```

## Seed packs
- `smoke` (3 seeds, easy) - loop sanity.
- `short_train` (8 seeds, 5 easy + 3 medium) - quick iteration.
- `final_demo` (12 seeds, 4 easy + 5 medium + 3 hard) - before/after demo.
