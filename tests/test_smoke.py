import unittest

from changeguard import Action, ActionType, Observation, RolloutStage
from changeguard.client import ChangeGuardClient
from changeguard.server.app import ChangeGuardServerApp, create_app
from changeguard.server.changeguard_environment import ChangeGuardEnvironment
from training.changeguard_tool_env import ChangeGuardToolEnv


class SmokeImportsTests(unittest.TestCase):
    def test_imports_and_object_creation(self):
        env = ChangeGuardEnvironment()
        app_env = create_app()
        client = ChangeGuardClient()
        tool_env = ChangeGuardToolEnv(client=client)

        self.assertIsInstance(env, ChangeGuardEnvironment)
        self.assertIsInstance(app_env, ChangeGuardServerApp)
        self.assertIsInstance(tool_env, ChangeGuardToolEnv)

        action = Action(action_type=ActionType.INSPECT_TENANT_PROFILE)
        obs = Observation(stage=RolloutStage.PLAN)
        self.assertEqual(action.name, "inspect_tenant_profile")
        self.assertEqual(obs.phase, "plan")


if __name__ == "__main__":
    unittest.main()
