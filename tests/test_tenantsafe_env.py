import unittest

from tenantsafe.env import TenantSafeEnv
from tenantsafe.policies import safe_policy


class TenantSafeEnvTests(unittest.TestCase):
    def test_safe_policy_completes_successfully(self):
        env = TenantSafeEnv(max_steps=12)
        obs, _ = env.reset(seed=123)
        done = False
        truncated = False
        info = {}

        while not done and not truncated:
            action = safe_policy(obs)
            obs, _, done, truncated, info = env.step(action)

        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertTrue(info.get("safe_completion", False))
        self.assertFalse(info.get("outage", True))

    def test_finalize_without_approval_fails(self):
        env = TenantSafeEnv(max_steps=12)
        env.reset(seed=42)
        env.step("canary_rollout_tenant_a")
        env.step("expand_rollout_tenant_b")
        _, _, done, _, info = env.step("finalize_upgrade")

        self.assertTrue(done)
        self.assertFalse(info.get("safe_completion", True))
        self.assertTrue(info.get("outage", False))


if __name__ == "__main__":
    unittest.main()
