import unittest

from food_delivery_env.models import (
    Action,
    ActionType,
    FoodDeliveryObservation,
    FoodDeliveryState,
    Location,
    OrderStatus,
)


class FoodDeliveryModelsTests(unittest.TestCase):
    def test_action_roundtrip(self):
        action = Action(action_type="assign_order")

        self.assertEqual(action.action_type, ActionType.ASSIGN_ORDER)
        self.assertEqual(action.name, "assign_order")
        self.assertEqual(Action.from_dict(action.to_dict()), action)

    def test_state_serializes_public_fields(self):
        state = FoodDeliveryState(
            courier_location=Location.RESTAURANT,
            order_status=OrderStatus.READY_AT_RESTAURANT,
            food_ready=True,
        )
        data = state.to_dict()

        self.assertEqual(data["courier_location"], "restaurant")
        self.assertEqual(data["order_status"], "ready_at_restaurant")
        self.assertNotIn("prep_delay_steps", data)

    def test_observation_rejects_bad_mask(self):
        with self.assertRaises(ValueError):
            FoodDeliveryObservation(
                legal_actions=[ActionType.ASSIGN_ORDER],
                action_mask=[0, 0, 0, 0, 0, 0],
            )


if __name__ == "__main__":
    unittest.main()
