import unittest

from dispatch_arena.models import (
    Action,
    Config,
    Courier,
    DispatchArenaObservation,
    Mode,
    Observation,
    Order,
    OrderStatus,
    State,
)


class DispatchArenaModelsTests(unittest.TestCase):
    def test_action_roundtrip(self):
        action = Action(action_type="assign", courier_id="courier_0", order_id="order_0")

        self.assertEqual(action.name, "assign")
        self.assertEqual(Action.from_dict(action.to_dict()), action)

    def test_config_clamps_normal_size(self):
        config = Config(mode=Mode.NORMAL, num_couriers=9, num_orders=99)

        self.assertEqual(config.num_couriers, 5)
        self.assertEqual(config.num_orders, 10)

    def test_state_serializes_public_fields_without_hidden_prep_when_none(self):
        state = State(
            orders=[
                Order(
                    id="order_0",
                    pickup_node_id="pickup",
                    dropoff_node_id="dropoff",
                    status=OrderStatus.READY,
                    prep_remaining=None,
                )
            ],
            couriers=[Courier(id="courier_0", node_id="hub")],
        )
        data = state.to_dict()

        self.assertEqual(data["orders"][0]["status"], "ready")
        self.assertNotIn("prep_remaining", str(data))

    def test_observation_alias_exports(self):
        obs = Observation()

        self.assertIsInstance(obs, DispatchArenaObservation)


if __name__ == "__main__":
    unittest.main()
