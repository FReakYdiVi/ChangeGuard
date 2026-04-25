# Food Delivery Env Spec

## Problem

One courier must complete one restaurant delivery in a finite-state simulation. The public state uses stable IDs (`courier_0`, `order_0`, `restaurant_0`, `customer_0`) so later versions can scale to multiple couriers, orders, and locations.

## Contract

```python
reset(seed=None, episode_id=None) -> FoodDeliveryObservation
step(action) -> FoodDeliveryObservation
state -> FoodDeliveryState
```

`reward`, `done`, and `truncated` live on `FoodDeliveryObservation`.

## Actions

| Action | Legal when | Effect |
|---|---|---|
| `assign_order` | `order_status == unassigned` | Assigns `order_0` to `courier_0`. |
| `move_to_restaurant` | Assigned order, courier at dispatch | Moves courier to `restaurant_0`. |
| `wait_at_restaurant` | Courier at restaurant, food not ready | Consumes a step. Cooking progresses before action validation. |
| `pickup_order` | Courier at restaurant, food ready | Marks order as picked up. |
| `move_to_customer` | Order picked up, courier at restaurant | Moves courier to `customer_0`. |
| `deliver_order` | Order picked up, courier at customer | Terminal success. |

## Observation

- `state`: sanitized `FoodDeliveryState`.
- `reward`: reward from the most recent transition.
- `done`: terminal flag.
- `truncated`: true only for timeout failure.
- `verifier_status`: `in_progress`, `delivered_successfully`, or `timeout_failure`.
- `reward_breakdown`: step, progress, invalid, terminal, and total reward components.
- `legal_actions`: explainable action list.
- `action_mask`: numeric mask in `ActionType` enum order.
- `summary_text`: compact public state text.
- `info`: transition metadata such as invalid action reasons.

Hidden state never exposed: exact `prep_delay_steps`.

## Step Ordering

1. Reject the step if the episode is terminal.
2. Apply base step penalty.
3. Consume one step.
4. Progress the hidden prep timer if the order is assigned and food is not ready.
5. Apply the action if it is legal; otherwise apply invalid penalty without changing courier/order location or status.
6. Recompute public state, legal actions, and mask.
7. If `deliver_order` succeeded, return terminal success.
8. Otherwise, if no steps remain, return terminal timeout failure.

This ordering lets a valid `deliver_order` on the final step succeed.

## Cooking Model

The reset seed samples hidden `prep_delay_steps` from `{1, 2, 3}`. Cooking progresses in the background after assignment, once per consumed step. Agents can only observe the derived `food_ready` boolean.

## Rewards

| Component | Value |
|---|---:|
| Step penalty | `-0.1` |
| `assign_order` | `+0.5` |
| `move_to_restaurant` | `+0.2` |
| `pickup_order` | `+1.0` |
| `move_to_customer` | `+0.2` |
| Invalid action | `-1.0` |
| `deliver_order` terminal success | `+10.0` |
| Timeout terminal failure | `-5.0` |

## Verifier

- `in_progress`: non-terminal episode.
- `delivered_successfully`: order delivered to `customer_0`.
- `timeout_failure`: step budget exhausted before delivery.

## Determinism

The same `(seed, max_steps)` reproduces the same hidden cooking delay and therefore the same legal-action trajectory for identical actions.
