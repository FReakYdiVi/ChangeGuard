# Food Delivery Env

A lean OpenEnv-style food delivery simulation. The courier must assign one order, reach the restaurant, wait only when the food is not ready, pick up, move to the customer, and deliver before the step budget expires.

## Layout

```text
food_delivery_env/
  pyproject.toml
  models.py
  client.py
  server/
    app.py
    food_delivery_environment.py
  tests/
  docs/SPEC.md
```

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ./food_delivery_env
```

## Run Server

```bash
.venv/bin/python -m food_delivery_env.server.app
```

## Run Tests

```bash
python -m unittest discover -s food_delivery_env/tests
```

## Actions

| Action | Legal when | Effect |
|---|---|---|
| `assign_order` | Order is unassigned | Assigns `order_0` to `courier_0`. |
| `move_to_restaurant` | Order assigned and courier at dispatch | Moves courier to `restaurant_0`. |
| `wait_at_restaurant` | Courier at restaurant and food not ready | Consumes a step while cooking progresses. |
| `pickup_order` | Courier at restaurant and food ready | Marks the order as picked up. |
| `move_to_customer` | Order picked up and courier at restaurant | Moves courier to `customer_0`. |
| `deliver_order` | Order picked up and courier at customer | Terminal success. |

## Observation Schema

`reset()` and `step()` return `FoodDeliveryObservation`:

```json
{
  "state": {
    "episode_id": "episode-1",
    "courier_id": "courier_0",
    "order_id": "order_0",
    "restaurant_id": "restaurant_0",
    "customer_id": "customer_0",
    "courier_location": "dispatch",
    "courier_assigned_order_id": null,
    "order_status": "unassigned",
    "food_ready": false,
    "steps_taken": 0,
    "steps_remaining": 8,
    "last_action": null,
    "verifier_status": "in_progress"
  },
  "reward": 0.0,
  "done": false,
  "truncated": false,
  "verifier_status": "in_progress",
  "legal_actions": ["assign_order"],
  "action_mask": [1, 0, 0, 0, 0, 0],
  "summary_text": "Courier=courier_0 at dispatch; order=unassigned; assigned_order=none; food_ready=false; steps_remaining=8."
}
```

Hidden state such as the exact cooking delay is never included in observations, public state, summaries, or text.

## Rewards

| Component | Value |
|---|---:|
| Step penalty | `-0.1` |
| Assign order | `+0.5` |
| Move to restaurant | `+0.2` |
| Pickup order | `+1.0` |
| Move to customer | `+0.2` |
| Invalid action | `-1.0` |
| Deliver order | `+10.0` |
| Timeout | `-5.0` |

## Client Example

```python
from food_delivery_env.client import FoodDeliveryClient
from food_delivery_env.models import ActionType

client = FoodDeliveryClient()
obs = client.reset(seed=7)

while not obs.done:
    action = obs.legal_actions[0]
    if ActionType.WAIT_AT_RESTAURANT in obs.legal_actions:
        action = ActionType.WAIT_AT_RESTAURANT
    obs = client.step(action)

print(obs.verifier_status, obs.reward)
```
