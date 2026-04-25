"""Local HTTP server wiring for the food delivery environment."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from food_delivery_env.models import Action, Observation
from food_delivery_env.server.food_delivery_environment import (
    DEFAULT_MAX_STEPS,
    FoodDeliveryEnvironment,
)

SUPPORTS_CONCURRENT_SESSIONS = True
DEFAULT_MAX_CONCURRENT_ENVS = 16
MAX_CONCURRENT_ENVS_ENV = "FOOD_DELIVERY_MAX_CONCURRENT_ENVS"


@dataclass
class _Session:
    env: FoodDeliveryEnvironment
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class FoodDeliveryServerApp:
    """Session manager for concurrent local environments."""

    max_concurrent_envs: int = DEFAULT_MAX_CONCURRENT_ENVS
    max_steps: int = DEFAULT_MAX_STEPS
    _sessions: Dict[str, _Session] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, Observation]:
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
            else:
                if len(self._sessions) >= self.max_concurrent_envs:
                    raise RuntimeError(
                        f"Max concurrent envs reached ({self.max_concurrent_envs})."
                    )
                session_id = session_id or str(uuid.uuid4())
                session = _Session(env=FoodDeliveryEnvironment(max_steps=self.max_steps))
                self._sessions[session_id] = session

        with session.lock:
            obs = session.env.reset(seed=seed, episode_id=episode_id or session_id)
        return session_id, obs

    def step(self, session_id: str, action: Action | str) -> Observation:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.step(action)

    def state(self, session_id: str) -> dict:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.state.to_dict()

    def summary(self, session_id: str) -> dict:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.get_episode_summary()

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "food_delivery_env",
            "supports_concurrent_sessions": SUPPORTS_CONCURRENT_SESSIONS,
        }

    def ready(self) -> dict:
        with self._lock:
            return {
                "ready": True,
                "active_sessions": len(self._sessions),
                "max_concurrent_envs": self.max_concurrent_envs,
            }

    def _require_session(self, session_id: str) -> _Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            return self._sessions[session_id]


def create_app(
    max_concurrent_envs: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> FoodDeliveryServerApp:
    if max_concurrent_envs is None:
        max_concurrent_envs = int(
            os.getenv(MAX_CONCURRENT_ENVS_ENV, str(DEFAULT_MAX_CONCURRENT_ENVS))
        )
    return FoodDeliveryServerApp(
        max_concurrent_envs=max_concurrent_envs,
        max_steps=max_steps,
    )


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _make_handler(app: FoodDeliveryServerApp):
    class FoodDeliveryHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/health":
                    _json_response(self, 200, app.health())
                    return
                if parsed.path == "/ready":
                    _json_response(self, 200, app.ready())
                    return
                if parsed.path == "/state":
                    session_id = _required_qs(qs, "session_id")
                    _json_response(self, 200, {"session_id": session_id, "state": app.state(session_id)})
                    return
                if parsed.path == "/summary":
                    session_id = _required_qs(qs, "session_id")
                    _json_response(self, 200, {"session_id": session_id, "episode_summary": app.summary(session_id)})
                    return
                _json_response(self, 404, {"error": "not_found"})
            except KeyError as exc:
                _json_response(self, 404, {"error": str(exc)})
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                _json_response(self, 500, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = _read_json(self)
                if parsed.path == "/reset":
                    session_id, obs = app.reset(
                        seed=payload.get("seed"),
                        episode_id=payload.get("episode_id"),
                        session_id=payload.get("session_id"),
                    )
                    _json_response(self, 200, {"session_id": session_id, "observation": obs.to_dict()})
                    return
                if parsed.path == "/step":
                    session_id = payload.get("session_id")
                    if not session_id:
                        raise ValueError("Missing required field: session_id")
                    action_payload = payload.get("action")
                    if action_payload is None:
                        raise ValueError("Missing required field: action")
                    obs = app.step(session_id=session_id, action=_parse_action(action_payload))
                    _json_response(self, 200, {"session_id": session_id, "observation": obs.to_dict()})
                    return
                _json_response(self, 404, {"error": "not_found"})
            except KeyError as exc:
                _json_response(self, 404, {"error": str(exc)})
            except RuntimeError as exc:
                _json_response(self, 409, {"error": str(exc)})
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                _json_response(self, 500, {"error": str(exc)})

    return FoodDeliveryHandler


def _required_qs(qs: dict, key: str) -> str:
    values = qs.get(key)
    if not values or not values[0]:
        raise ValueError(f"Missing query parameter: {key}")
    return values[0]


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    content_len = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_len) if content_len > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _parse_action(action_payload: object) -> Action | str:
    if isinstance(action_payload, str):
        return action_payload
    if isinstance(action_payload, dict):
        return Action.from_dict(action_payload)
    raise ValueError("Invalid action payload. Use an action string or {'action_type': ...}.")


def run_local_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    max_concurrent_envs: Optional[int] = None,
) -> Tuple[str, int]:
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    actual_host, actual_port = server.server_address
    print(f"Food delivery server running at http://{actual_host}:{actual_port}")
    print(f"max_concurrent_envs={app.max_concurrent_envs} (env: {MAX_CONCURRENT_ENVS_ENV})")
    server.serve_forever()
    return actual_host, actual_port


def run_local_server_in_thread(
    host: str = "127.0.0.1",
    port: int = 0,
    max_concurrent_envs: Optional[int] = None,
):
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    run_local_server()


if __name__ == "__main__":
    main()
