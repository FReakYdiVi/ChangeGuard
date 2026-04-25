"""Run one food delivery episode through the local HTTP client."""

from food_delivery_env.client import FoodDeliveryClient
from food_delivery_env.models import ActionType
from food_delivery_env.server.app import run_local_server_in_thread


def main() -> None:
    server, thread = run_local_server_in_thread(port=8081, max_concurrent_envs=8)
    try:
        client = FoodDeliveryClient(base_url="http://127.0.0.1:8081")
        obs = client.reset(seed=42)
        print("reset:", obs.summary_text)

        while not obs.done:
            if ActionType.WAIT_AT_RESTAURANT in obs.legal_actions:
                action = ActionType.WAIT_AT_RESTAURANT
            else:
                action = obs.legal_actions[0]
            obs = client.step(action)
            print(
                f"action={action.value} reward={obs.reward:.2f} "
                f"done={obs.done} verdict={obs.verifier_status.value}"
            )

        print("final summary:", client.fetch_summary())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
