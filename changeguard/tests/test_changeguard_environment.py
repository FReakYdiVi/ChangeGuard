import unittest

from changeguard.models import Action, ActionType, TenantId, VerifierVerdict
from changeguard.server.changeguard_environment import ChangeGuardEnvironment


class ChangeGuardEnvironmentTests(unittest.TestCase):
    def test_naive_unsafe_rollout_path(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=7, difficulty="easy")

        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        result = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.APPROVAL_VIOLATION)
        self.assertLess(result.reward_total, 0.0)

    def test_safe_a_then_b_then_defer_c(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=8, difficulty="easy")

        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        result = env.step(Action(action_type=ActionType.PAUSE_ROLLOUT, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertTrue(result.verifier_flags.safe_abort)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.SAFE_ROLLBACK)

    def test_safe_finalize_after_compat_and_approval(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=9, difficulty="easy")

        env.step("inspect_compatibility")
        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        env.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
        env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
        result = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertTrue(result.verifier_flags.safe_completion)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.SAFE_FINALIZED)

    def test_invalid_approval_bypass(self):
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=10, difficulty="easy")

        env.step("canary_upgrade")
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        env.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
        result = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

        self.assertTrue(result.done)
        self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.APPROVAL_VIOLATION)
        self.assertTrue(result.verifier_flags.approval_violation)

    def test_deterministic_same_seed_same_trajectory(self):
        actions = [
            "inspect_compatibility",
            "canary_upgrade",
            Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B),
            Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C),
            Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C),
            Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C),
        ]

        env1 = ChangeGuardEnvironment(max_steps=12)
        env2 = ChangeGuardEnvironment(max_steps=12)
        env1.reset(seed=1234, difficulty="medium")
        env2.reset(seed=1234, difficulty="medium")

        trail1 = []
        trail2 = []

        for action in actions:
            r1 = env1.step(action)
            r2 = env2.step(action)
            trail1.append((r1.reward_total, r1.verifier_flags.verdict.value, r1.observation.summary_text))
            trail2.append((r2.reward_total, r2.verifier_flags.verdict.value, r2.observation.summary_text))
            if r1.done:
                break

        self.assertEqual(trail1, trail2)
        self.assertEqual(env1.get_episode_summary().final_verdict, env2.get_episode_summary().final_verdict)


if __name__ == "__main__":
    unittest.main()
