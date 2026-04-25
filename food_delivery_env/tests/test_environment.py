import unittest

from food_delivery_env.models import ActionType, Location, OrderStatus, VerifierVerdict
from food_delivery_env.server.food_delivery_environment import FoodDeliveryEnvironment


def play_greedy_episode(env: FoodDeliveryEnvironment):
    obs = env.reset(seed=1)
    while not obs.done:
        action = obs.legal_actions[0]
        obs = env.step(action)
    return obs


class FoodDeliveryEnvironmentTests(unittest.TestCase):
    def test_golden_successful_trajectory(self):
        env = FoodDeliveryEnvironment(max_steps=8)
        obs = play_greedy_episode(env)

        self.assertTrue(obs.done)
        self.assertFalse(obs.truncated)
        self.assertEqual(obs.verifier_status, VerifierVerdict.DELIVERED_SUCCESSFULLY)
        self.assertEqual(obs.state.order_status, OrderStatus.DELIVERED)
        self.assertGreater(env.get_episode_summary()["total_reward"], 0.0)

    def test_invalid_action_penalty_without_courier_or_order_mutation(self):
        env = FoodDeliveryEnvironment(max_steps=8)
        env.reset(seed=1)
        before = env.state

        obs = env.step(ActionType.PICKUP_ORDER)
        after = env.state

        self.assertTrue(obs.info["invalid_action"])
        self.assertLess(obs.reward_breakdown.invalid_action_penalty, 0.0)
        self.assertEqual(after.courier_location, before.courier_location)
        self.assertEqual(after.courier_assigned_order_id, before.courier_assigned_order_id)
        self.assertEqual(after.order_status, before.order_status)
        self.assertEqual(after.steps_taken, before.steps_taken + 1)

    def test_timeout_trajectory_is_negative(self):
        env = FoodDeliveryEnvironment(max_steps=2)
        env.reset(seed=1)

        env.step(ActionType.ASSIGN_ORDER)
        obs = env.step(ActionType.MOVE_TO_RESTAURANT)

        self.assertTrue(obs.done)
        self.assertTrue(obs.truncated)
        self.assertEqual(obs.verifier_status, VerifierVerdict.TIMEOUT_FAILURE)
        self.assertLess(env.get_episode_summary()["total_reward"], 0.0)

    def test_hidden_prep_delay_never_appears_publicly(self):
        env = FoodDeliveryEnvironment(max_steps=8)
        obs = env.reset(seed=3)
        env.step(ActionType.ASSIGN_ORDER)

        public_blob = " ".join(
            [
                str(obs.to_dict()),
                str(env.state.to_dict()),
                obs.summary_text,
                str(env.get_episode_summary()),
            ]
        )

        self.assertNotIn("prep_delay", public_blob)
        self.assertNotIn("prep_delay_steps", public_blob)

    def test_action_mask_matches_legal_action_list(self):
        env = FoodDeliveryEnvironment(max_steps=8)
        obs = env.reset(seed=4)

        for action_type, mask_value in zip(ActionType, obs.action_mask):
            self.assertEqual(mask_value, 1 if action_type in obs.legal_actions else 0)

        obs = env.step(ActionType.ASSIGN_ORDER)
        for action_type, mask_value in zip(ActionType, obs.action_mask):
            self.assertEqual(mask_value, 1 if action_type in obs.legal_actions else 0)

    def test_observed_legal_wait_action_is_not_invalidated_by_cooking_progress(self):
        env = FoodDeliveryEnvironment(max_steps=8)
        obs = env.reset(seed=7)
        env.step(ActionType.ASSIGN_ORDER)
        obs = env.step(ActionType.MOVE_TO_RESTAURANT)

        self.assertIn(ActionType.WAIT_AT_RESTAURANT, obs.legal_actions)
        obs = env.step(ActionType.WAIT_AT_RESTAURANT)

        self.assertFalse(obs.info["invalid_action"])
        self.assertEqual(env.get_episode_summary()["invalid_actions"], 0)
        self.assertIn(ActionType.PICKUP_ORDER, obs.legal_actions)

    def test_seeded_reset_is_reproducible(self):
        env1 = FoodDeliveryEnvironment(max_steps=8)
        env2 = FoodDeliveryEnvironment(max_steps=8)
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)

        self.assertEqual(obs1.to_dict(), obs2.to_dict())

        actions = [
            ActionType.ASSIGN_ORDER,
            ActionType.MOVE_TO_RESTAURANT,
            ActionType.WAIT_AT_RESTAURANT,
            ActionType.PICKUP_ORDER,
        ]
        trace1 = [env1.step(action).to_dict() for action in actions]
        trace2 = [env2.step(action).to_dict() for action in actions]
        self.assertEqual(trace1, trace2)

    def test_deliver_order_on_final_step_succeeds(self):
        env = FoodDeliveryEnvironment(max_steps=7)
        env.reset(seed=1)
        assert env._world is not None
        env._world.prep_delay_steps = 3

        env.step(ActionType.ASSIGN_ORDER)
        env.step(ActionType.MOVE_TO_RESTAURANT)
        env.step(ActionType.WAIT_AT_RESTAURANT)
        env.step(ActionType.WAIT_AT_RESTAURANT)
        env.step(ActionType.PICKUP_ORDER)
        obs = env.step(ActionType.MOVE_TO_CUSTOMER)

        self.assertEqual(obs.state.courier_location, Location.CUSTOMER)
        self.assertFalse(obs.done)
        obs = env.step(ActionType.DELIVER_ORDER)

        self.assertTrue(obs.done)
        self.assertFalse(obs.truncated)
        self.assertEqual(obs.verifier_status, VerifierVerdict.DELIVERED_SUCCESSFULLY)


if __name__ == "__main__":
    unittest.main()
