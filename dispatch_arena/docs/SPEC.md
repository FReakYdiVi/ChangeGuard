# Dispatch Arena Spec

## Contract

```python
reset(seed=None, episode_id=None, config=None) -> Observation
step(action) -> Observation
state -> State
```

The same `DispatchArenaEnvironment` powers training, REST, WebSocket streaming, replay, and the frontend.

## Mini Mode

Mini mode has one courier, one order, and three nodes: `hub`, `pickup`, `dropoff`.

| Action | Legal when | Effect |
|---|---|---|
| `wait` | Episode is active | Consumes a tick while prep can progress. |
| `go_pickup` | Courier is not carrying and not at pickup | Moves courier to `pickup`. |
| `pickup` | Courier at pickup and order ready | Courier carries the order. |
| `go_dropoff` | Courier carrying and not at dropoff | Moves courier to `dropoff`. |
| `dropoff` | Courier carrying at dropoff | Terminal success. |

## Normal Mode

Normal mode has 2-5 couriers and 3-10 orders. The agent is a centralized dispatcher.

| Action | Required args | Effect |
|---|---|---|
| `assign` | `courier_id`, `order_id` | Sends an idle courier to an unassigned open order. |
| `reposition` | `courier_id`, `node_id` | Moves an idle courier to a graph node. |
| `hold` | optional `courier_id` | Consumes a tick. |
| `prioritize` | optional `order_id` | Records dispatch intent without privileged state. |

Pickup and dropoff are automatic once an assigned courier reaches the relevant node and the order is ready.

## Observation

- `state`: sanitized public state.
- `reward`: reward from the most recent transition.
- `done`: terminal flag.
- `truncated`: true only for hard timeout.
- `verifier_status`: `in_progress`, `delivered_successfully`, `partial_success`, or `timeout_failure`.
- `reward_breakdown`: all reward components and total.
- `legal_actions`: action names currently available.
- `action_mask`: mask in mode-specific action order.
- `summary_text`: compact public explanation.
- `info`: transition metadata such as invalid action reasons and events.

Hidden state never exposed in hidden mode: exact `prep_remaining`.

## Step Ordering

1. Reject the step if the episode is terminal.
2. Apply base step cost.
3. Consume one tick.
4. Progress prep timers.
5. Validate the requested action against the pre-transition legal state.
6. Apply valid action effects or invalid penalty.
7. In normal mode, advance courier travel and automatic pickup/dropoff.
8. Expire overdue open orders.
9. Apply timeout if max ticks is reached.
10. Recompute public state, legal actions, metrics, and replay payload.

## Determinism

The same `(mode, seed, config, action_trace)` reproduces the same scenario, observations, reward components, and replay records.
