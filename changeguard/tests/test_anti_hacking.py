import unittest

from changeguard.models import Action, ActionType, TenantId, VerifierVerdict
from changeguard.server.changeguard_environment import ChangeGuardEnvironment


class AntiHackingTests(unittest.TestCase):
    def test_hidden_dependency_not_exposed_in_observation(self):
        env = ChangeGuardEnvironment(max_steps=12)
        obs = env.reset(seed=11, difficulty="easy")
        payload = obs.to_dict()
        text_blob = str(payload) + " " + obs.summary_text

        self.assertNotIn("has_legacy_export_dependency", text_blob)
        self.assertNotIn("legacy_dependency", text_blob)
        self.assertNotIn("rollback_deadline_step", text_blob)

    def test_repeated_inspect_loop_is_mildly_penalized(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=12, difficulty="easy")

        first = env.step("inspect_compatibility")
        second = env.step("inspect_compatibility")

        self.assertGreater(first.reward_total, 0.0)
        self.assertLess(second.reward_total, 0.0)
        self.assertLess(second.reward_breakdown.loop_penalty, 0.0)

    def test_promote_c_before_compat_is_violation(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=13, difficulty="easy")

        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
        result = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.COMPATIBILITY_VIOLATION)
        self.assertEqual(result.info.get("violation_reason"), "compatibility_missing_for_tenant_c_finalize")

    def test_finalize_without_approval_is_violation(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=14, difficulty="easy")

        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        env.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
        result = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.APPROVAL_VIOLATION)
        self.assertTrue(result.verifier_flags.approval_violation)

    def test_timeout_cannot_increase_total_reward(self):
        env = ChangeGuardEnvironment(max_steps=6)
        env.reset(seed=15, difficulty="easy")

        result = None
        for _ in range(6):
            result = env.step("inspect_tenant")
        assert result is not None

        summary = env.get_episode_summary()
        self.assertTrue(result.done)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.TIMEOUT_FAILURE)
        self.assertLessEqual(summary.final_reward, 0.0)

    def test_invalid_actions_are_visible_and_penalized(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=16, difficulty="easy")

        result = env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))

        self.assertFalse(result.done)
        self.assertTrue(result.info.get("invalid_action"))
        self.assertIsNotNone(result.info.get("invalid_reason"))
        self.assertIn("invalid_action_count", result.info)
        self.assertLess(result.reward_breakdown.invalid_action_penalty, 0.0)

    def test_reset_same_seed_same_world(self):
        env1 = ChangeGuardEnvironment(max_steps=12)
        env2 = ChangeGuardEnvironment(max_steps=12)

        env1.reset(seed=2026, difficulty="medium")
        env2.reset(seed=2026, difficulty="medium")

        self.assertEqual(env1.world_signature(), env2.world_signature())

    def test_different_seeds_vary_only_allowed_dimensions(self):
        env = ChangeGuardEnvironment(max_steps=12)

        signatures = []
        for seed in [31, 32, 33, 34, 35, 36]:
            env.reset(seed=seed, difficulty="medium")
            signatures.append(env.world_signature())

        # Backcompat scenarios (default) use fixed presets, so schema_v2_diff and
        # tenant_deps are invariant across seeds.
        invariant_keys = {
            "difficulty", "max_steps", "tenant_ids", "c_requires_approval",
            "scenario_id", "schema_v2_diff", "tenant_deps",
        }
        for key in invariant_keys:
            values = {sig[key] if not isinstance(sig[key], list) else tuple(sig[key]) for sig in signatures}
            self.assertEqual(len(values), 1, f"invariant key varied unexpectedly: {key}")

        # Allowed varying dimensions.
        varying_keys = {"rollback_deadline_step", "b_has_hidden_risk"}
        for sig in signatures:
            self.assertEqual(set(sig.keys()), invariant_keys | varying_keys)


if __name__ == "__main__":
    unittest.main()
