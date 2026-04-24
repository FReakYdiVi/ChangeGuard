import unittest

from changeguard.models import (
    ActionType,
    ChangeGuardAction,
    ChangeGuardObservation,
    RiskHintLevel,
    RolloutStage,
    SignalLevel,
    TenantHiddenState,
    TenantId,
    TenantVisibleState,
)


class ModelsValidationTests(unittest.TestCase):
    def test_action_enum_coercion_and_roundtrip(self):
        action = ChangeGuardAction(action_type="inspect_tenant_profile", target_tenant="C")
        self.assertEqual(action.action_type, ActionType.INSPECT_TENANT_PROFILE)
        self.assertEqual(action.target_tenant, TenantId.C)

        payload = action.to_dict()
        restored = ChangeGuardAction.from_dict(payload)
        self.assertEqual(restored.action_type, action.action_type)
        self.assertEqual(restored.target_tenant, action.target_tenant)

    def test_observation_validation_and_serialization(self):
        obs = ChangeGuardObservation(
            stage=RolloutStage.PLAN,
            tenants_visible={
                TenantId.A: TenantVisibleState(tenant_id=TenantId.A, risk_tier="low"),
                TenantId.B: TenantVisibleState(tenant_id=TenantId.B, risk_tier="medium"),
                TenantId.C: TenantVisibleState(
                    tenant_id=TenantId.C,
                    risk_tier="high",
                    approval_required=True,
                ),
            },
            service_health_score=0.95,
            export_job_signal_c=SignalLevel.WARNING,
            compat_report_signal=SignalLevel.HEALTHY,
            risk_hint_level=RiskHintLevel.MEDIUM,
            steps_remaining=10,
            legal_actions=["inspect_tenant_profile", ActionType.PAUSE_ROLLOUT],
            summary_text="A and B ready; C requires compat+approval.",
        )

        data = obs.to_dict()
        self.assertIn("summary_text", data)
        self.assertIn("legal_actions", data)
        self.assertNotIn("has_legacy_export_dependency", str(data))

        restored = ChangeGuardObservation.from_dict(data)
        self.assertEqual(restored.stage, RolloutStage.PLAN)
        self.assertEqual(restored.risk_hint_level, RiskHintLevel.MEDIUM)
        self.assertEqual(restored.phase, "plan")

    def test_observation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            ChangeGuardObservation(service_health_score=1.2)

        with self.assertRaises(ValueError):
            ChangeGuardObservation(steps_remaining=-1)

    def test_hidden_state_is_separate_model(self):
        hidden = TenantHiddenState(
            tenant_id=TenantId.C,
            has_legacy_export_dependency=True,
            export_job_health_internal=SignalLevel.FAILING,
        )
        hidden_dict = hidden.to_dict()
        self.assertTrue(hidden_dict["has_legacy_export_dependency"])


if __name__ == "__main__":
    unittest.main()
