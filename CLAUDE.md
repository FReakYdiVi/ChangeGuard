# ChangeGuard / Dispatch Arena — Claude Context

## What this repo is

Single OpenEnv-compatible package: [dispatch_arena/](dispatch_arena/). Despite the
parent dir name `ChangeGuard`, the only live project is the dispatch
simulation. Older variants (`food_delivery_env`, `changeguard/`) are gone — the
git status shows them as `D` because they were deleted on the current branch
(`food-delivery-env-v0`) before the rename to `dispatch_arena/`.

The package is a deterministic, server-authoritative delivery-dispatch
environment for RL training, replay, and a thin demo UI. It exposes both
OpenEnv-style endpoints (`/reset`, `/step`, `/state`, `/summary`) and a
friendlier `/api/sessions/...` namespace with WebSocket streaming.

## Layout (read these first)

- [dispatch_arena/models.py](dispatch_arena/models.py) — Pydantic models:
  `Action`, `Config`, `Courier`, `Order`, `State`, `Observation`,
  `RewardBreakdown`, `EpisodeSummary`, plus `Mode` / `*ActionType` / `*Status`
  enums.
- [dispatch_arena/server/env.py](dispatch_arena/server/env.py) —
  `DispatchArenaEnvironment`: the simulator core (`reset`, `step`, `state`,
  `legal_actions`, `action_mask`, `get_episode_summary`). All transition logic
  lives here.
- [dispatch_arena/server/scenarios.py](dispatch_arena/server/scenarios.py) —
  seeded scenario generation for `mini` and `normal` modes; defines per-bucket
  prep/deadline distributions (`easy`, `tight`, `long_tail`,
  `shifted_distribution`).
- [dispatch_arena/server/rewards.py](dispatch_arena/server/rewards.py) —
  `RewardModel` with decomposed components: `step_cost`, `progress_reward`,
  `invalid_penalty`, `success_reward`, `timeout_penalty`, `on_time_bonus`,
  `late_penalty`, `idle_penalty`, `route_churn_penalty`, `fairness_penalty`.
- [dispatch_arena/server/serializers.py](dispatch_arena/server/serializers.py)
  — `public_state` / `make_observation`. Strips hidden state (esp.
  `prep_remaining`) before anything leaves the simulator.
- [dispatch_arena/server/app.py](dispatch_arena/server/app.py) — FastAPI app,
  session manager, REST + WebSocket + OpenEnv endpoints, static UI mount.
- [dispatch_arena/server/replay_store.py](dispatch_arena/server/replay_store.py)
  — JSONL replay persistence per session, used by the UI and the summary
  endpoint.
- [dispatch_arena/client.py](dispatch_arena/client.py) —
  `DispatchArenaClient`, an `urllib`-only HTTP wrapper with `reset`, `step`,
  `fetch_state`, `fetch_summary`, `fetch_replay`.
- [dispatch_arena/openenv.yaml](dispatch_arena/openenv.yaml) — environment
  manifest declaring action schemas and observation fields.
- [dispatch_arena/Dockerfile](dispatch_arena/Dockerfile) — `python:3.11-slim`
  + `uv`, exposes `7860` for HF Space-style deployment.
- [dispatch_arena/server/static/index.html](dispatch_arena/server/static/index.html)
  — vanilla-JS demo UI. The [dispatch_arena/frontend/](dispatch_arena/frontend/)
  Vite scaffold is a thin placeholder that builds into `server/static/`.
- [dispatch_arena/tests/](dispatch_arena/tests/) — unittest suite covering
  determinism, reward components, hidden-state leakage, invalid-action
  handling, and one full HTTP roundtrip.
- [dispatch_arena/scripts/demo_client_episode.py](dispatch_arena/scripts/demo_client_episode.py)
  — runs one mini episode end-to-end against an in-thread server.

## Modes today

- **mini**: 1 courier, 1 order, three nodes (`hub`, `pickup`, `dropoff`).
  Actions: `wait`, `go_pickup`, `pickup`, `go_dropoff`, `dropoff`. The only
  hidden variable is `prep_remaining`.
- **normal**: 2–5 couriers, 3–10 orders, hub + multiple stores + customers.
  Centralized dispatcher actions: `assign(courier_id, order_id)`,
  `reposition(courier_id, node_id)`, `hold([courier_id])`,
  `prioritize([order_id])`. Pickup/dropoff happen automatically when a
  courier reaches the relevant node and the order is ready.

## Hard invariants — do not violate

- **Server-authoritative state.** Anything observable goes through
  `serializers.public_state` / `make_observation`. Don't construct
  observations or replay records by hand from `_state`.
- **Hidden `prep_remaining` never leaks** unless `Config.visible_prep=True`.
  The string literal `"prep_remaining"` must not appear in any public blob —
  there's a test (`test_hidden_prep_remaining_never_appears_publicly`) that
  enforces this.
- **Determinism.** `(mode, seed, config, action_trace)` must reproduce the
  same observations, reward components, and replay records. New randomness
  must thread through the env's `random.Random` (or an explicitly seeded
  child), never `random.*` at module scope.
- **Step ordering** is fixed (see
  [docs/SPEC.md](dispatch_arena/docs/SPEC.md#step-ordering)): reject if
  terminal → step cost → tick++ → progress prep → validate vs pre-transition
  legal set → apply or invalid-penalty → advance couriers → expire orders →
  timeout → recompute derived state. New mechanics slot into this order, not
  around it.
- **Reward decomposition is the contract.** Keep adding fields to
  `RewardBreakdown` rather than collapsing into the scalar — multi-component
  rewards are how training will tell *why* the policy improved.

## Vision (what we're building toward)

Dispatch Arena should become a **noise-rich dispatch operator environment**
where the agent learns to manage uncertainty, not just route a single order.
The drivers (per the user's brainstorm doc):

- **Prep-time noise** — restaurant prep delays sampled per order; already
  partly there as hidden `prep_remaining`, but currently fixed at scenario
  generation. Want *rolling* delays (restaurants slipping mid-shift).
- **Delivery / travel noise** — traffic shocks that perturb travel times
  after assignment. Today `travel_time_matrix` is static.
- **Restaurant timing windows** — restaurants open/close, batch their cooking,
  go offline temporarily. Today restaurants are always available.
- **Bundling** — let the dispatcher (or the simulator) pair two orders going
  to nearby drops onto one courier. Today every order is solo.
- **Rolling order arrivals** — orders trickle in over the shift instead of
  all being known at `reset`. Today the full order list is fixed at t=0.

The training story is GRPO via TRL's OpenEnv `environment_factory`, with the
`RewardBreakdown` columns surfaced as separate reward functions. The full
design rationale (scoring rubric, ablations, anti-hacking, sprint plan) lives
in the brainstorm PDF the user shared in conversation — treat it as the
source of truth for *why* a change is worth doing.

## What's already in place vs. still to build

| Capability | State |
|---|---|
| OpenEnv contract (`reset`/`step`/`state`/`summary`) | Done |
| Mini + Normal modes | Done |
| Hidden prep, action masks, decomposed rewards | Done |
| Replay JSONL + WebSocket streaming | Done |
| FastAPI + Docker for HF Space deploy | Done |
| Rolling order arrivals during a shift | Done (normal mode, `Config.rolling_arrivals`) |
| Travel/traffic noise applied at step time | Done (normal mode, `Config.traffic_noise`) |
| Restaurant availability windows | **Not yet** |
| Order bundling (one courier, two drops) | **Not yet** |
| Stockouts / substitutions / customer messaging | **Not yet** |
| TRL GRPO trainer wiring + `environment_factory` wrapper | Done ([scripts/train_grpo_smoke.py](dispatch_arena/scripts/train_grpo_smoke.py)) |
| Heuristic / prompt-only baselines for before-after demos | **Not yet** |

When adding new mechanics, prefer extending `Config` (new flags,
distributions) and `Order` / `Courier` / `State` over forking new modes.
Scenario buckets in `scenarios.py` are the right place to dial difficulty.

## Common commands

```bash
# install (project uses uv)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ./dispatch_arena

# run server
.venv/bin/python -m dispatch_arena.server.app   # serves on :8080 (PORT env to override)

# run tests
.venv/bin/python -m unittest discover -s dispatch_arena/tests

# one-shot end-to-end demo episode
.venv/bin/python -m dispatch_arena.scripts.demo_client_episode
```

## Working notes

- Don't introduce a separate "simulator core" module yet — `DispatchArenaEnvironment`
  *is* the core, and `server/app.py` is just session glue. Keep it that way
  until there's a real reason to split.
- Tests live under [dispatch_arena/tests/](dispatch_arena/tests/) and run
  fast; add a new test alongside any new mechanic, especially a hidden-state
  leak test (mirror `test_hidden_prep_remaining_never_appears_publicly`) for
  any new private variable.
- The `frontend/` Vite project is currently a stub; the live UI is the
  vanilla-JS [server/static/index.html](dispatch_arena/server/static/index.html).
- Branch `food-delivery-env-v0` is the working branch; `main` is the older
  baseline.
