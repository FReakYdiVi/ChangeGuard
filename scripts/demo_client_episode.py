"""Tiny local demo: run one full episode through EnvClient over HTTP server."""

from changeguard.client import EnvClient
from changeguard.models import Action, ActionType, TenantId
from changeguard.server.app import run_local_server_in_thread


def main() -> None:
    server, thread = run_local_server_in_thread(port=8081, max_concurrent_envs=8)
    try:
        client = EnvClient(base_url="http://127.0.0.1:8081")
        obs = client.reset(seed=42, difficulty="easy")
        print("reset:", obs.summary_text)

        steps = [
            Action(action_type=ActionType.CANARY_ROLLOUT_TENANT_A, target_tenant=TenantId.A),
            Action(action_type=ActionType.EXPAND_ROLLOUT_TENANT_B, target_tenant=TenantId.B),
            Action(action_type=ActionType.ENABLE_COMPAT_MODE_TENANT_C, target_tenant=TenantId.C),
            Action(action_type=ActionType.REQUEST_APPROVAL_TENANT_C, target_tenant=TenantId.C),
            Action(action_type=ActionType.FINALIZE_UPGRADE, target_tenant=TenantId.C),
        ]

        for idx, action in enumerate(steps, start=1):
            result = client.step(action)
            print(f"step {idx}: action={action.name} reward={result.reward_total:.2f} done={result.done}")
            if result.done:
                print("verdict:", result.verifier_flags.verdict.value)
                break

        summary = client.fetch_summary()
        print("final summary:", summary.to_dict())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
