import unittest

from food_delivery_env.client import FoodDeliveryClient
from food_delivery_env.models import VerifierVerdict
from food_delivery_env.server.app import FoodDeliveryServerApp, create_app, run_local_server_in_thread
from food_delivery_env.server.food_delivery_environment import FoodDeliveryEnvironment


class FoodDeliveryServerClientTests(unittest.TestCase):
    def test_imports_and_object_creation(self):
        env = FoodDeliveryEnvironment()
        app = create_app()
        client = FoodDeliveryClient()

        self.assertIsInstance(env, FoodDeliveryEnvironment)
        self.assertIsInstance(app, FoodDeliveryServerApp)
        self.assertIsInstance(client, FoodDeliveryClient)

    def test_one_episode_over_client(self):
        try:
            server, thread = run_local_server_in_thread(port=0, max_concurrent_envs=4)
        except PermissionError:
            self.skipTest("Socket bind not permitted in current sandbox")
            return

        host, port = server.server_address
        client = FoodDeliveryClient(base_url=f"http://{host}:{port}")

        try:
            self.assertEqual(client.health()["service"], "food_delivery_env")
            obs = client.reset(seed=7)
            while not obs.done:
                obs = client.step(obs.legal_actions[0])

            self.assertEqual(obs.verifier_status, VerifierVerdict.DELIVERED_SUCCESSFULLY)
            self.assertEqual(client.fetch_summary()["final_verdict"], "delivered_successfully")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
