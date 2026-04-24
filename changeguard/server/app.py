"""Local HTTP server wiring for ChangeGuard.

This module exposes a simple, stable API for local development and training:
- POST /reset
- POST /step
- GET /state?session_id=...
- GET /summary?session_id=...
- GET /health
- GET /ready
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from changeguard.models import Action, ActionType, Observation, StepResult

from .changeguard_environment import ChangeGuardEnvironment

SUPPORTS_CONCURRENT_SESSIONS = True
DEFAULT_MAX_CONCURRENT_ENVS = 16
MAX_CONCURRENT_ENVS_ENV = "CHANGEGUARD_MAX_CONCURRENT_ENVS"


@dataclass
class _Session:
    env: ChangeGuardEnvironment
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ChangeGuardServerApp:
    """Session manager for concurrent ChangeGuard environments."""

    max_concurrent_envs: int = DEFAULT_MAX_CONCURRENT_ENVS
    _sessions: Dict[str, _Session] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "easy",
        session_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        prompt_style: Optional[str] = None,
    ) -> Tuple[str, Observation]:
        """Create or reset a session and return `(session_id, observation)`."""
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
            else:
                if len(self._sessions) >= self.max_concurrent_envs:
                    raise RuntimeError(
                        f"Max concurrent envs reached ({self.max_concurrent_envs})."
                    )
                session_id = session_id or str(uuid.uuid4())
                session = _Session(env=ChangeGuardEnvironment(max_steps=12))
                self._sessions[session_id] = session

        with session.lock:
            obs = session.env.reset(
                seed=seed,
                difficulty=difficulty,
                scenario_id=scenario_id,
                prompt_style=prompt_style,
            )
        return session_id, obs

    def step(self, session_id: str, action: Action | str) -> StepResult:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.step(action)

    def state(self, session_id: str) -> Observation:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.state

    def summary(self, session_id: str) -> dict:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.get_episode_summary().to_dict()

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "changeguard",
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


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def create_app(max_concurrent_envs: Optional[int] = None) -> ChangeGuardServerApp:
    """Build configured server app/session manager instance."""
    if max_concurrent_envs is None:
        max_concurrent_envs = int(os.getenv(MAX_CONCURRENT_ENVS_ENV, str(DEFAULT_MAX_CONCURRENT_ENVS)))
    return ChangeGuardServerApp(max_concurrent_envs=max_concurrent_envs)


def _make_handler(app: ChangeGuardServerApp):
    class ChangeGuardHandler(BaseHTTPRequestHandler):
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
                    obs = app.state(session_id)
                    _json_response(self, 200, {"session_id": session_id, "observation": obs.to_dict()})
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
                        difficulty=payload.get("difficulty", "easy"),
                        session_id=payload.get("session_id"),
                        scenario_id=payload.get("scenario_id"),
                        prompt_style=payload.get("prompt_style"),
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

                    action = _parse_action(action_payload)
                    result = app.step(session_id=session_id, action=action)
                    _json_response(self, 200, {"session_id": session_id, "step_result": result.to_dict()})
                    return

                _json_response(self, 404, {"error": "not_found"})
            except KeyError as exc:
                _json_response(self, 404, {"error": str(exc)})
            except RuntimeError as exc:
                _json_response(self, 429, {"error": str(exc)})
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                _json_response(self, 500, {"error": str(exc)})

    return ChangeGuardHandler


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
        if "action_type" in action_payload:
            return Action.from_dict(action_payload)
        if "name" in action_payload:
            # Friendly shorthand for quick local usage.
            return action_payload["name"]
    raise ValueError("Invalid action payload. Use string or action dict.")


def run_local_server(host: str = "127.0.0.1", port: int = 8080, max_concurrent_envs: Optional[int] = None) -> Tuple[str, int]:
    """Run local threaded HTTP server for ChangeGuard."""
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    actual_host, actual_port = server.server_address
    print(f"ChangeGuard server running at http://{actual_host}:{actual_port}")
    print(f"max_concurrent_envs={app.max_concurrent_envs} (env: {MAX_CONCURRENT_ENVS_ENV})")
    server.serve_forever()
    return actual_host, actual_port


def main() -> None:
    """CLI entrypoint required by OpenEnv multi-mode deployment checks."""
    run_local_server()


def run_local_server_in_thread(host: str = "127.0.0.1", port: int = 0, max_concurrent_envs: Optional[int] = None):
    """Start server in a background thread (useful for tests/demo)."""
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    main()
