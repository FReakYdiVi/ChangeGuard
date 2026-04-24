# ChangeGuard

Tenant-aware upgrade RL environment with OpenEnv-style server API, typed client, GRPO wrapper, fixed-seed eval, and anti-hacking verifier checks.

## Local Setup
```bash
cd "/Users/shivanshmundra/Downloads/CloudWatch AI"
python3.10 -m venv .venv310
source .venv310/bin/activate
python --version
```

Optional (for non-dry-run GRPO):
```bash
pip install datasets trl
```

## Local Server Run
```bash
source .venv310/bin/activate
python -m changeguard.server.app
```

## Local Tests
```bash
source .venv310/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
```

## OpenEnv Validate
```bash
# If OpenEnv CLI is installed
openenv validate changeguard/openenv.yaml
```

## HF Space Deployment

### Docker/Server assumptions
- Server is plain Python HTTP (`changeguard.server.app`).
- Concurrency is session-based and configurable with `CHANGEGUARD_MAX_CONCURRENT_ENVS`.
- Container listens on `PORT` (default `7860`).

### Build locally
```bash
docker build -t changeguard:local .
docker run --rm -p 7860:7860 -e CHANGEGUARD_MAX_CONCURRENT_ENVS=16 changeguard:local
```

### Push to HF Space (Docker Space)
```bash
# create/select a Docker Space repo, then:
git add .
git commit -m "Prepare ChangeGuard for local + HF Spaces"
git push
```

## Short Training Run
```bash
# Dry-run: cheap and debuggable
python -m training.train_grpo --dry-run --seed-pack short_train

# Real GRPO (requires trl + datasets)
python -m training.train_grpo --no-dry-run --seed-pack short_train --max-steps 8 --prompt-repeats 8
```

## Fixed-Seed Evaluation
```bash
# Baseline vs trained-like policy on fixed seeds
python -m training.evaluate_policy --seed-pack final_demo --candidate trained_like
```

## One-command Local Pipeline
Runs baseline -> short train -> optional real train -> same-seed evaluation:
```bash
./scripts/run_fixed_seed_pipeline.sh

# Enable real GRPO training inside pipeline:
REAL_TRAIN=1 ./scripts/run_fixed_seed_pipeline.sh
```
