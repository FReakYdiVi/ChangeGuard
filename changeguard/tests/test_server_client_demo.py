import unittest

from changeguard.client import EnvClient
from changeguard.models import Action, ActionType, TenantId, VerifierVerdict
from changeguard.server.app import run_local_server_in_thread


class ServerClientDemoTests(unittest.TestCase):
    def test_one_episode_over_client(self):
        try:
            server, thread = run_local_server_in_thread(port=0, max_concurrent_envs=4)
        except PermissionError:
            self.skipTest("Socket bind not permitted in current sandbox")
            return

        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        try:
            client = EnvClient(base_url=base_url)

            health = client.health()
            ready = client.ready()
            self.assertEqual(health["status"], "ok")
            self.assertTrue(ready["ready"])

            client.reset(seed=33, difficulty="easy")
            client.step(Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A))
            client.step(Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B))
            client.step(Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C))
            client.step(Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C))
            result = client.step(Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C))

            self.assertTrue(result.done)
            self.assertEqual(result.verifier_flags.verdict, VerifierVerdict.SAFE_FINALIZED)

            summary = client.fetch_summary()
            self.assertEqual(summary.final_verdict, VerifierVerdict.SAFE_FINALIZED)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
