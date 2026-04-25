"""Run one Dispatch Arena mini episode through the local HTTP client."""

from dispatch_arena.client import DispatchArenaClient
from dispatch_arena.server.app import run_local_server_in_thread


def main() -> None:
    server, thread = run_local_server_in_thread(port=8081, max_concurrent_envs=8)
    try:
        client = DispatchArenaClient(base_url="http://127.0.0.1:8081")
        obs = client.reset(seed=42)
        print("reset:", obs.summary_text)

        while not obs.done:
            for preferred in ["pickup", "dropoff", "go_pickup", "go_dropoff", "wait"]:
                if preferred in obs.legal_actions:
                    action = preferred
                    break
            obs = client.step(action)
            print(
                f"action={action} reward={obs.reward:.2f} "
                f"done={obs.done} verdict={obs.verifier_status.value}"
            )

        print("final summary:", client.fetch_summary())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
