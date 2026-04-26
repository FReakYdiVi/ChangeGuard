# Dispatch Arena: Training a Delivery Dispatch Agent with OpenEnv and TRL

> How one wrong food-delivery location turned into a simulator for training RL agents on messy dispatch decisions.

## The Problem I Accidentally Found

A few days before the hackathon, I ordered food on Zomato.

Nothing unusual. I was staying with my brother in Bangalore, I picked the saved address, placed the order, and waited. The delivery partner accepted it and started moving. Then he called and said he had arrived.

I went outside. He was not there.

He called again. I could not fully understand what he was saying because of the language gap, so I checked the obvious thing first: did I put the wrong address?

The written address looked correct.

Then he asked me to share my Google Maps location. I sent it. After a few confusing calls, I realized what had happened: he was almost 4-5 kilometers away from my actual location.

That was the small bug that made me think.

In food delivery apps, the courier does not only receive a written address. They also receive a map pin. In practice, the pin often wins. That makes sense most of the time, but when the address and the pin disagree, somebody needs to reason about the mismatch.

A human might ask:

- Is the saved address old?
- Is the map pin wrong?
- Should we ask the customer to verify before dispatching?
- Should the courier start moving or wait for confirmation?

That sounded like an agent problem to me. Not just a chatbot problem, but an agent operating inside a real workflow where every action changes the next state.

My first idea was too broad: can I build an RL agent for food delivery problems?

That was vague. Food delivery includes address correction, restaurant prep, courier routing, batching, support, refunds, ETA prediction, and probably fifty other things I had not even thought about. So I started reading more, and I came across logistics engineering discussions like Swiggy Bytes posts about delivery systems. One recurring theme was dispatch: deciding which courier should serve which order, when, and under what uncertainty.

That became the narrower problem.

Not "solve food delivery."

Train an agent to learn dispatch decisions.

## The Real Dispatch Problem

Today, millions of orders are placed every day across apps like Swiggy and Zomato. Behind each order, some system is making a fast decision:

- which courier should get this order
- whether to assign now or wait for prep
- whether to reposition an idle courier
- whether one courier is getting overloaded
- whether a tight-deadline order should be prioritized

Most dispatch systems are not one magical model thinking end to end. A lot of the real machinery is built from algorithms, heuristics, forecasts, and operational rules.

That is not a criticism. Heuristics are useful. Rules like "send the nearest free courier", "wait if the restaurant looks slow", or "rebalance couriers every few seconds" can work beautifully on quiet shifts.

They break when the shift gets messy.

Restaurants run late. Traffic changes. New orders arrive while older ones are still waiting. One courier gets too many assignments while another sits idle. A decision that looked locally good at tick 3 becomes bad at tick 10.

When these small decisions fail, the cost is not abstract:

- customers get cold food
- couriers wait at restaurants
- deadlines are missed
- refunds happen
- the same mistake repeats in the next shift because the rule did not learn

Multiplied across millions of orders, a tiny dispatch inefficiency becomes a very real business problem.

This is why dispatch is interesting for RL. It is not a single prediction. It is a long-horizon decision problem under partial information.

![Dispatch problem animation](./blog_assets/dispatch_problem.svg)

The animation above is the core idea. The left side is a naive policy: assign immediately, reach the restaurant early, wait, get hit by traffic, and deliver late. The right side is a better dispatch policy: wait or reposition first, assign closer to food readiness, and deliver on time.

The map can look almost the same, but the timing is different. That timing is the dispatch problem.

## Why I Built Dispatch Arena

I wanted an environment where an agent could actually act, fail, receive feedback, and improve.

So I built **Dispatch Arena**: a deterministic, server-authoritative delivery-dispatch environment. It exposes OpenEnv-style endpoints like `/reset`, `/step`, `/state`, and `/summary`, plus a replay/demo API for visualization.

The environment has two modes.

**Mini mode** is the small training sandbox:

- 1 courier
- 1 order
- 1 pickup
- 1 dropoff
- hidden restaurant prep time
- simple actions like `wait`, `pickup`, and `dropoff`

This was useful for testing the simulator loop, but it was too easy. A hard-coded policy can solve it.

**Normal mode** is closer to the real dispatch story:

- multiple couriers
- multiple active orders
- restaurant prep uncertainty
- rolling order arrivals
- traffic noise
- deadlines
- courier load balance
- centralized dispatcher actions

The agent does not directly move every courier step by step. It acts like a dispatcher. It can call tools such as:

- `view_dashboard`
- `assign`
- `reposition`
- `hold`
- `prioritize`
- `finish_shift`

Each tool call changes the simulator state. The next observation depends on what happened before.

## What Makes This an RL Environment

The important part is that the agent is not answering a static prompt.

It is inside a world.

At every step, it sees a dashboard: courier locations, order statuses, deadlines, travel times, and recent events. Some state is intentionally hidden. For example, exact restaurant prep time is not exposed unless visible-prep mode is enabled.

Then the model chooses a tool call.

The server validates that action, advances time, updates couriers and orders, applies traffic/prep effects, and returns:

- the next observation
- whether the episode is done
- legal actions
- a decomposed reward
- a scalar reward used for training

The reward is not just "good" or "bad". It has components:

- step cost
- progress reward
- invalid action penalty
- success reward
- on-time bonus
- late penalty
- idle courier penalty
- route churn penalty
- fairness penalty

That decomposition matters because it tells us why the policy is improving or failing.

## Training with TRL GRPO

The training setup is inspired by the OpenEnv + TRL style shown in the Hugging Face CARLA article, where a model learns by tool-calling into a simulator and receiving rewards from the environment: [Bringing Autonomous Driving RL to OpenEnv and TRL](https://huggingface.co/blog/sergiopaniego/bringing-carla-to-openenv-trl).

Dispatch Arena uses the same high-level pattern, but for delivery dispatch instead of autonomous driving.

The stack is:

- **Model:** `Qwen/Qwen3-1.7B`
- **RL trainer:** TRL `GRPOTrainer`
- **Fine-tuning:** PEFT LoRA
- **Dataset:** HuggingFace `Dataset` built from the scenario catalog
- **Environment server:** FastAPI Dispatch Arena server
- **Client:** `DispatchArenaClient`
- **Logging:** TensorBoard
- **Saved artifact:** final LoRA adapter

The catalog contains scenario specs. In the current training run, it loads 100 scenarios and splits them into 70 train and 30 held-out eval scenarios.

Each dataset row contains:

- a system prompt
- a user kickoff message
- a seed
- a normal-mode environment config

The difficulty labels and scenario tags are kept as metadata. They are not leaked into the model prompt.

![Dispatch Arena RL training loop](./blog_assets/rl_training_loop.svg)

The loop works like this:

1. TRL samples a scenario row.
2. `GRPOTrainer` creates grouped rollouts for the same prompt.
3. The Qwen policy reads the dashboard and emits one tool call.
4. `DispatchToolEnv` converts that tool call into a server action.
5. The FastAPI environment advances the dispatch world.
6. The reward model returns the step reward.
7. The rollout accumulates `reward_total`.
8. GRPO compares grouped rollouts and updates the LoRA adapter.

In code, the trainer is wired like this:

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=[reward_total],
    args=config,
    train_dataset=train_ds,
    environment_factory=DispatchToolEnv,
    peft_config=lora_config,
)
```

The actual hyperparameters are intentionally small enough for a hackathon-style run:

- `max_steps=50`
- `num_generations=2`
- `per_device_train_batch_size=2`
- `learning_rate=1e-5`
- `max_tool_calling_iterations=20`
- `fp16=True`
- LoRA rank `r=16`

This is not meant to be a production dispatch model. It is a proof that the simulator, reward loop, tool-calling interface, and training path are connected end to end.

## What the Agent Is Learning

The agent is not learning only "nearest courier wins."

It has to learn trade-offs like:

- assigning too early can make a courier wait at the restaurant
- assigning too late can miss the deadline
- holding can be useful, but too much holding wastes time
- repositioning can help if future demand is predictable
- one courier should not get every order while others idle
- traffic noise can turn a safe-looking route into a late delivery

This is why RL fits better than a one-step classifier. The quality of an action depends on what it causes later.

For example, assigning the nearest courier may look correct now. But if the food is not ready and another order will appear near that courier two ticks later, the locally obvious decision may be globally worse.

That is the kind of mistake I wanted Dispatch Arena to expose.

## What Is Done

The current project already has:

- OpenEnv-compatible `/reset`, `/step`, `/state`, `/summary`
- mini and normal modes
- hidden prep time
- traffic noise
- rolling order arrivals
- action masks and legal actions
- decomposed rewards
- replay storage
- WebSocket/demo UI support
- GRPO smoke training
- catalog-driven normal-mode training
- held-out evaluation scripts

The trainer saves the final LoRA adapter to:

```text
dispatch_arena/scripts/_grpo_normal_out/final_lora
```

## What Is Still Missing

There are still many real delivery problems that Dispatch Arena does not model yet:

- address/pin mismatch verification
- restaurant open/close windows
- restaurant going offline temporarily
- order bundling
- customer messaging
- stockouts and substitutions
- richer courier behavior
- adversarial scenario generation

The address mismatch story is what started this project, but the current environment focuses on dispatch because it is a clean RL problem to build first.

That is probably the right scope for now.

## Why This Matters

Dispatch is one of those problems that looks simple from outside.

Food has to go from restaurant to customer. Send a courier. Done.

But inside the system, every decision is made with incomplete information. Restaurant prep is hidden. Traffic changes. Orders arrive later. Couriers are not interchangeable. A bad action can look harmless for a few ticks and only show its cost near the deadline.

That is exactly the kind of workflow where an agent should not just follow a fixed rule. It should learn from scenarios, fail safely in simulation, and improve before touching the real world.

Dispatch Arena is my attempt to make that learning loop small enough to run, inspect, replay, and explain.

## Try It Locally

Install the package:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ./dispatch_arena
```

Run the server:

```bash
.venv/bin/python -m dispatch_arena.server.app
```

Run the tests:

```bash
.venv/bin/python -m unittest discover -s dispatch_arena/tests
```

Run one demo episode:

```bash
.venv/bin/python -m dispatch_arena.scripts.demo_client_episode
```

Run the normal-mode GRPO training script on a GPU:

```bash
.venv/bin/python -m dispatch_arena.scripts.train_grpo
```

## Resources

- Reference article: [Bringing Autonomous Driving RL to OpenEnv and TRL](https://huggingface.co/blog/sergiopaniego/bringing-carla-to-openenv-trl)
- TODO: add the exact Swiggy Bytes logistics/dispatch article link here.
- Main training script: `dispatch_arena/scripts/train_grpo.py`
- Environment core: `dispatch_arena/server/env.py`
- Reward model: `dispatch_arena/server/rewards.py`
- Scenario catalog: `dispatch_arena/catalog/catalog.json`
