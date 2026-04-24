# ChangeGuard Training Loop And Dataset Design

## 1) Is our agent training like the diagram?
Short answer: **yes in structure, simpler in infrastructure**.

Your screenshot shows a full self-improving loop with:
- adversarial scenario generation
- real infra execution
- judge-driven reward
- curriculum controller
- GRPO update

Current ChangeGuard implements the same core RL loop, but with a compact local stack:
- **Scenario generator**: deterministic seed packs (`smoke`, `short_train`, `final_demo`)
- **Environment**: simulated tenant-aware migration world (not real GKE)
- **Judge/verifier**: programmatic verifier + reward components in env
- **Curriculum**: difficulty levels + seed packs
- **Policy update**: GRPO via TRL (when `--no-dry-run` and deps installed)

## 2) End-to-end improvement loop
1. Trainer sends a prompt to the model.
2. Model chooses tool actions (`inspect_compatibility`, `canary_upgrade`, `promote_upgrade`, etc.).
3. Tool wrapper calls the OpenEnv server.
4. Environment advances hidden/visible state and returns:
   - next observation
   - reward breakdown
   - violation flags
   - done / terminal verdict
5. GRPO collects rollout trajectories and updates policy weights.
6. Updated policy is re-evaluated on fixed seeds.
7. Curriculum moves from easy to medium to hard to reduce exploit-only learning.

This repeats until fixed-seed metrics improve (safe completion up, violations down).

## 3) What “dataset” means in this repo
This project is mainly **online RL data**, not a large static offline dataset.

### A) Prompt dataset (small static text)
Defined in `training/train_grpo.py` (`build_tiny_prompt_dataset`).
- Content: repeated task instruction prompt(s)
- Role: gives GRPO starting context for tool-use rollouts
- Not logs/schema data

### B) Seed/scenario packs (structured episode definitions)
Defined in `training/train_grpo.py` (`SEED_PACKS`).
- Content per row:
  - `seed`
  - `difficulty`
  - `scenario_id`
- Role: controls which world instances are sampled
- This is your train/eval scenario set

### C) Rollout trajectories (generated during training)
Produced at runtime from model-environment interaction.
- Typical fields:
  - observation summary + machine-readable state fields
  - action/tool call
  - reward total + reward components
  - done flag
  - verifier flags / violations
- Role: actual learning signal for GRPO

### D) Fixed-seed evaluation set
Reuses the deterministic seed packs for before/after comparison.
- Role: objective benchmark for judge-friendly metrics

## 4) Do we use logs / API schema / auth schema as training data?
**Not in v1 by default.**

In v1, “logs” are environment signals returned by inspect actions (simulated service/export indicators), not raw production log corpora.

If you want to extend later:
- API/schema metadata can be added as observation fields or scenario files
- auth policy constraints can be encoded as extra verifier rules
- real logs can be injected as scenario-conditioned evidence strings

## 5) Why this is good for hackathon execution
- Fast to run locally
- Deterministic and objectively verifiable
- Easy to show policy improvement live
- Low infra risk compared to real-cluster training

## 6) Practical mapping to your current files
- Loop + seeds + GRPO entry: `training/train_grpo.py`
- Tool wrapper and metrics extraction: `training/changeguard_tool_env.py`
- Evaluation on fixed seeds: `training/evaluate_policy.py`
- Core rewards/verifier/state transitions: `changeguard/server/changeguard_environment.py`

## 7) Suggested v2 data extensions (optional)
1. `data/scenarios/*.jsonl`: explicit scenario manifests with hidden truth labels.
2. `data/prompts/*.jsonl`: multiple operator prompt styles.
3. `artifacts/rollouts/*.jsonl`: persisted per-step trajectories for audit/analysis.
4. `data/eval/final_holdout.jsonl`: strict never-train evaluation pack.
