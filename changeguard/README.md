# ChangeGuard

Tenant-aware upgrade RL environment (OpenEnv server + TRL GRPO wrapper). Simulates invoice schema migration V1 -> V2 across tenants A/B/C, where C has a hidden legacy dependency requiring compatibility mode and approval before finalize.

See [docs/SPEC.md](docs/SPEC.md) for the environment contract. See [CLAUDE.md](CLAUDE.md) for the repo layout and conventions.

## Repo layout
```
<repo-root>/
├── .venv/
└── changeguard/           <-- project dir AND python package (single flat layout)
    ├── pyproject.toml
    ├── __init__.py, models.py, client.py, internal_state.py
    ├── server/            <-- OpenEnv HTTP app + env engine
    ├── training/          <-- TRL GRPO wrapper
    ├── tests/             (28 tests: contract, determinism, anti-hacking, reward-hacking)
    ├── docs/              (SPEC.md + archive/)
    ├── scripts/
    └── artifacts/         (gitignored)
```

## Setup
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e "./changeguard[training]"
```

## Run server
```bash
.venv/bin/python -m changeguard.server.app
```

## Tests
```bash
.venv/bin/python -m unittest discover -s changeguard/tests
```

## GRPO dry-run
```bash
.venv/bin/python -m changeguard.training.train_grpo --dry-run --seed-pack smoke
```

## GRPO real run (T4 16GB, LoRA, Qwen2.5-1.5B default)
```bash
.venv/bin/python -m changeguard.training.train_grpo --no-dry-run --lora --seed-pack short_train --max-steps 10
```

## Fixed-seed evaluation
```bash
.venv/bin/python -m changeguard.training.evaluate_policy --seed-pack final_demo --candidate trained_like
```

## Seed packs
- `smoke` (3 seeds, easy) - loop sanity.
- `short_train` (8 seeds, 5 easy + 3 medium) - quick iteration.
- `final_demo` (12 seeds, 4 easy + 5 medium + 3 hard) - before/after demo.
