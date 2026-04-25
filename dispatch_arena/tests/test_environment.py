import random
import unittest

from dispatch_arena.models import Config, Mode, OrderStatus, VerifierVerdict
from dispatch_arena.server.env import DispatchArenaEnvironment


def heuristic_action(obs):
    state = obs.state
    if state.mode == Mode.NORMAL:
        courier = next((c for c in state.couriers if c.status == "idle" and c.load is None), None)
        order = next(
            (
                o
                for o in state.orders
                if o.status in {"queued", "ready"} and o.assigned_courier_id is None
            ),
            None,
        )
        if courier and order:
            return {"action_type": "assign", "courier_id": courier.id, "order_id": order.id}
        return {"action_type": "hold"}
    for action in ["pickup", "dropoff", "go_pickup", "go_dropoff", "wait"]:
        if action in obs.legal_actions:
            return action
    return obs.legal_actions[0]


def play_episode(env: DispatchArenaEnvironment, seed: int = 1):
    obs = env.reset(seed=seed)
    while not obs.done:
        obs = env.step(heuristic_action(obs))
    return obs


class DispatchArenaEnvironmentTests(unittest.TestCase):
    def test_mini_golden_successful_trajectory(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12))
        obs = play_episode(env, seed=1)

        self.assertTrue(obs.done)
        self.assertFalse(obs.truncated)
        self.assertEqual(obs.verifier_status, VerifierVerdict.DELIVERED_SUCCESSFULLY)
        self.assertEqual(obs.state.orders[0].status, OrderStatus.DELIVERED)
        self.assertGreater(env.get_episode_summary()["total_reward"], 0.0)

    def test_mini_reward_components_are_returned(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12))
        obs = env.reset(seed=1)
        self.assertIn("go_pickup", obs.legal_actions)

        obs = env.step("go_pickup")

        self.assertEqual(obs.reward_breakdown.step_cost, -0.1)
        self.assertGreater(obs.reward_breakdown.progress_reward, 0.0)
        self.assertEqual(obs.reward, obs.reward_breakdown.total_reward)

    def test_invalid_action_penalty_without_mutation(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12))
        env.reset(seed=1)
        before = env.state

        obs = env.step("dropoff")
        after = env.state

        self.assertTrue(obs.info["invalid_action"])
        self.assertLess(obs.reward_breakdown.invalid_penalty, 0.0)
        self.assertEqual(after.couriers[0].node_id, before.couriers[0].node_id)
        self.assertEqual(after.orders[0].status, before.orders[0].status)
        self.assertEqual(after.tick, before.tick + 1)

    def test_timeout_trajectory_is_negative(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=1))
        env.reset(seed=1)

        obs = env.step("wait")

        self.assertTrue(obs.done)
        self.assertTrue(obs.truncated)
        self.assertEqual(obs.verifier_status, VerifierVerdict.TIMEOUT_FAILURE)
        self.assertLess(obs.reward_breakdown.timeout_penalty, 0.0)

    def test_hidden_prep_remaining_never_appears_publicly(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12, visible_prep=False))
        obs = env.reset(seed=3)
        env.step("wait")

        public_blob = " ".join([str(obs.to_dict()), str(env.state.to_dict()), obs.summary_text, str(env.get_episode_summary())])

        self.assertNotIn("prep_remaining", public_blob)

    def test_visible_mode_can_expose_ready_now_and_prep(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12, visible_prep=True))
        obs = env.reset(seed=3)

        data = obs.to_dict()

        self.assertIn("prep_remaining", str(data))
        self.assertIn("ready_now", str(data))

    def test_action_mask_matches_legal_action_list(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12))
        obs = env.reset(seed=4)

        for action, mask_value in zip(["wait", "go_pickup", "go_dropoff", "pickup", "dropoff"], obs.action_mask):
            self.assertEqual(mask_value, 1 if action in obs.legal_actions else 0)

    def test_seeded_reset_is_reproducible(self):
        config = Config(mode=Mode.MINI, max_ticks=12)
        env1 = DispatchArenaEnvironment(config)
        env2 = DispatchArenaEnvironment(config)
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)

        self.assertEqual(obs1.to_dict(), obs2.to_dict())

        actions = ["go_pickup", "wait", "pickup"]
        trace1 = [env1.step(action).to_dict() for action in actions]
        trace2 = [env2.step(action).to_dict() for action in actions]
        self.assertEqual(trace1, trace2)

    def test_random_rollout_never_breaks_invariants(self):
        env = DispatchArenaEnvironment(Config(mode=Mode.MINI, max_ticks=12))
        obs = env.reset(seed=9)
        rng = random.Random(9)

        while not obs.done:
            action = rng.choice(["wait", "go_pickup", "go_dropoff", "pickup", "dropoff"])
            obs = env.step(action)
            delivered = obs.state.orders[0].status == OrderStatus.DELIVERED
            self.assertFalse(delivered and obs.state.couriers[0].load)
            self.assertLessEqual(obs.state.tick, obs.state.max_ticks)

    def test_normal_heuristic_rollout_delivers_some_orders(self):
        config = Config(mode=Mode.NORMAL, max_ticks=18, num_couriers=3, num_orders=5)
        env = DispatchArenaEnvironment(config)
        obs = play_episode(env, seed=5)

        self.assertTrue(obs.done)
        self.assertGreaterEqual(env.get_episode_summary()["delivered_orders"], 1)
        for order in obs.state.orders:
            self.assertFalse(order.status == OrderStatus.DELIVERED and order.assigned_courier_id is None)

    def test_normal_invalid_duplicate_assignment_is_penalized(self):
        config = Config(mode=Mode.NORMAL, max_ticks=18, num_couriers=2, num_orders=3)
        env = DispatchArenaEnvironment(config)
        env.reset(seed=5)
        env.step({"action_type": "assign", "courier_id": "courier_0", "order_id": "order_0"})

        obs = env.step({"action_type": "assign", "courier_id": "courier_1", "order_id": "order_0"})

        self.assertTrue(obs.info["invalid_action"])
        self.assertLess(obs.reward_breakdown.invalid_penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
