import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.models import Config, Mode, VerifierVerdict
from dispatch_arena.server.app import DispatchArenaServerApp, create_app, run_local_server_in_thread
from dispatch_arena.server.env import DispatchArenaEnvironment
from dispatch_arena.server.replay_store import ReplayStore


class DispatchArenaServerClientTests(unittest.TestCase):
    def test_imports_and_object_creation(self):
        env = DispatchArenaEnvironment()
        app = create_app()
        client = DispatchArenaClient()

        self.assertIsInstance(env, DispatchArenaEnvironment)
        self.assertIsInstance(app.state.dispatch_arena, DispatchArenaServerApp)
        self.assertIsInstance(client, DispatchArenaClient)

    def test_fastapi_session_replay_and_openenv_paths(self):
        app = create_app(max_concurrent_envs=4)
        client = TestClient(app)

        health = client.get("/healthz").json()
        self.assertEqual(health["service"], "dispatch_arena")

        created = client.post("/api/sessions", json={"mode": "mini", "seed": 7, "config": {"max_ticks": 12}}).json()
        session_id = created["session_id"]
        obs = created["observation"]

        while not obs["done"]:
            for action in ["pickup", "dropoff", "go_pickup", "go_dropoff", "wait"]:
                if action in obs["legal_actions"]:
                    break
            obs = client.post(f"/api/sessions/{session_id}/step", json={"action": action}).json()["observation"]

        self.assertEqual(obs["verifier_status"], "delivered_successfully")
        state = client.get(f"/api/sessions/{session_id}/state").json()["state"]
        self.assertTrue(state["done"])
        replay = client.get(f"/api/sessions/{session_id}/replay").json()["records"]
        self.assertGreaterEqual(len(replay), state["tick"] + 1)
        self.assertEqual(replay[0]["type"], "reset")
        self.assertEqual(replay[-1]["type"], "summary")

        reset = client.post("/reset", json={"seed": 1, "config": {"mode": "mini", "max_ticks": 12}}).json()
        openenv_session = reset["session_id"]
        openenv_state = client.get("/state", params={"session_id": openenv_session}).json()["state"]
        self.assertEqual(openenv_state["mode"], "mini")

    def test_replay_store_persists_reward_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReplayStore(root=Path(tmp))
            manager = DispatchArenaServerApp(replay_store=store)
            session_id, obs = manager.create_session(Config(mode=Mode.MINI, max_ticks=12), seed=7)

            self.assertFalse(obs.done)
            obs = manager.step(session_id, "go_pickup")
            records = manager.replay(session_id)

            self.assertEqual(records[0]["type"], "reset")
            self.assertEqual(records[1]["type"], "step")
            self.assertIn("reward_breakdown", records[1])
            self.assertEqual(records[1]["reward_breakdown"]["total_reward"], obs.reward)

    def test_one_episode_over_client(self):
        try:
            server, thread = run_local_server_in_thread(port=0, max_concurrent_envs=4)
        except PermissionError:
            self.skipTest("Socket bind not permitted in current sandbox")
            return

        host, port = server.server_address
        client = DispatchArenaClient(base_url=f"http://{host}:{port}")

        try:
            self.assertEqual(client.health()["service"], "dispatch_arena")
            obs = client.reset(seed=7)
            while not obs.done:
                obs = client.step(obs.legal_actions[0])

            self.assertEqual(obs.verifier_status, VerifierVerdict.DELIVERED_SUCCESSFULLY)
            self.assertEqual(client.fetch_summary()["final_verdict"], "delivered_successfully")
            self.assertGreaterEqual(len(client.fetch_replay()), obs.state.tick + 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
