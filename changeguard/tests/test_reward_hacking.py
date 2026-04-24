"""Active reward-hacking tests.

Each test codifies one shortcut a GRPO policy might try to exploit and asserts
the env punishes it. These are the "tripwires" — if any of these start passing
trivially in the future (zero cost for the exploit), we've regressed.
"""

from __future__ import annotations

import unittest

from changeguard.models import (
    Action,
    ActionType,
    MigrationMitigation,
    TenantId,
)
from changeguard.server.changeguard_environment import ChangeGuardEnvironment


class RewardHackingTests(unittest.TestCase):
    # ---------- #1 per-tenant inspect_tenant de-dup ----------

    def test_inspect_tenant_per_tenant_is_not_a_loop(self):
        """inspect_tenant(A) then inspect_tenant(B) must both earn first-time reward."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=501, difficulty="easy", scenario_id="procedural_easy")

        r1 = env.step(Action(action_type=ActionType.INSPECT_TENANT_PROFILE, target_tenant=TenantId.A))
        r2 = env.step(Action(action_type=ActionType.INSPECT_TENANT_PROFILE, target_tenant=TenantId.B))
        r3 = env.step(Action(action_type=ActionType.INSPECT_TENANT_PROFILE, target_tenant=TenantId.C))

        self.assertGreater(r1.reward_breakdown.inspection_reward, 0)
        self.assertGreater(r2.reward_breakdown.inspection_reward, 0)
        self.assertGreater(r3.reward_breakdown.inspection_reward, 0)
        self.assertEqual(r2.reward_breakdown.loop_penalty, 0.0)
        self.assertEqual(r3.reward_breakdown.loop_penalty, 0.0)

    def test_inspect_tenant_same_tenant_twice_is_a_loop(self):
        """Re-inspecting the same tenant does trigger the loop penalty."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=502, difficulty="easy", scenario_id="procedural_easy")

        env.step(Action(action_type=ActionType.INSPECT_TENANT_PROFILE, target_tenant=TenantId.C))
        repeat = env.step(Action(action_type=ActionType.INSPECT_TENANT_PROFILE, target_tenant=TenantId.C))
        self.assertLess(repeat.reward_breakdown.loop_penalty, 0.0)

    # ---------- #2 defer is a safety valve, not a reward pump ----------

    def test_defer_spam_gives_diminishing_returns(self):
        """Three defers in a row: first +0.7, others loop-penalized."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=503, difficulty="easy", scenario_id="procedural_easy")

        r1 = env.step(Action(action_type=ActionType.PAUSE_ROLLOUT, target_tenant=TenantId.C))
        r2 = env.step(Action(action_type=ActionType.PAUSE_ROLLOUT, target_tenant=TenantId.C))
        r3 = env.step(Action(action_type=ActionType.PAUSE_ROLLOUT, target_tenant=TenantId.C))

        # First defer pays; subsequent defers get loop penalty, not safety reward.
        self.assertAlmostEqual(r1.reward_breakdown.safety_reward, 0.7)
        self.assertEqual(r2.reward_breakdown.safety_reward, 0.0)
        self.assertLess(r2.reward_breakdown.loop_penalty, 0.0)
        self.assertLess(r3.reward_breakdown.loop_penalty, 0.0)

    # ---------- #3 unnecessary mitigations earn nothing ----------

    def test_unnecessary_mitigation_pays_zero(self):
        """Applying a mitigation for an op not in the diff earns 0 (was +0.1)."""
        env = ChangeGuardEnvironment(max_steps=12)
        # easy_stable: diff = [rename_col] only. dual_write is never required here.
        env.reset(seed=504, difficulty="easy", scenario_id="easy_stable")
        r = env.step("apply_dual_write")
        self.assertEqual(r.reward_breakdown.safety_reward, 0.0)

    def test_required_mitigation_pays_full(self):
        env = ChangeGuardEnvironment(max_steps=12)
        # easy_stable: diff = [rename_col] + C has legacy_export -> compat_mode required.
        env.reset(seed=505, difficulty="easy", scenario_id="easy_stable")
        r = env.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
        self.assertAlmostEqual(r.reward_breakdown.safety_reward, 0.8)

    # ---------- #4 speed bonus ----------

    def test_safe_finalize_pays_speed_bonus(self):
        """Finishing earlier within the step budget earns more terminal bonus."""
        # Safe path on easy_stable (only compat_mode needed).
        def play(max_steps):
            env = ChangeGuardEnvironment(max_steps=max_steps)
            env.reset(seed=506, difficulty="easy", scenario_id="easy_stable")
            env.step("inspect_compatibility")
            env.step(Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A))
            env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
            env.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
            env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
            r = env.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))
            return r.reward_breakdown.terminal_bonus_or_penalty

        # Same plan, different step budgets -> different steps_remaining at finalize.
        bonus_tight = play(max_steps=6)    # finalize on last step: steps_remaining = 0
        bonus_loose = play(max_steps=12)   # finalize with 6 steps to spare
        self.assertGreater(bonus_loose, bonus_tight)
        # Each spare step should be worth 0.3.
        self.assertAlmostEqual(bonus_loose - bonus_tight, 0.3 * 6, places=4)

    # ---------- #7 blind-approval penalty ----------

    def test_blind_approval_is_penalized(self):
        """Requesting approval without inspect_compatibility costs safety reward."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=507, difficulty="easy", scenario_id="easy_stable")

        env.step(Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A))
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        r = env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
        # Approval granted but with blind-approval penalty folded in.
        # Net safety_reward = +0.8 - 0.3 = +0.5.
        self.assertAlmostEqual(r.reward_breakdown.safety_reward, 0.5, places=4)
        self.assertEqual(r.info.get("violation_reason"), "blind_approval_without_compat_inspection")

    def test_inspected_approval_gets_full_safety_reward(self):
        """With inspect_compatibility first, approval pays the full +0.8."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=508, difficulty="easy", scenario_id="easy_stable")

        env.step("inspect_compatibility")
        env.step(Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A))
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        r = env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
        self.assertAlmostEqual(r.reward_breakdown.safety_reward, 0.8, places=4)

    # ---------- approval-spam guard ----------

    def test_approval_regrant_is_a_noop(self):
        """Calling request_approval twice pays only once; second call is loop-penalized."""
        env = ChangeGuardEnvironment(max_steps=12)
        env.reset(seed=509, difficulty="easy", scenario_id="easy_stable")

        env.step("inspect_compatibility")
        env.step(Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A))
        env.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
        first = env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
        second = env.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))

        self.assertAlmostEqual(first.reward_breakdown.safety_reward, 0.8)
        self.assertEqual(second.reward_breakdown.safety_reward, 0.0)
        self.assertLess(second.reward_breakdown.loop_penalty, 0.0)
        self.assertEqual(second.info.get("invalid_reason"), "approval_already_granted")


if __name__ == "__main__":
    unittest.main()
