"""Typed client for the Dispatch Arena server API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dispatch_arena.models import Action, Config, Observation, State


class EnvClientError(RuntimeError):
    """Raised when the server returns a non-success response."""


@dataclass
class DispatchArenaClient:
    """Small typed wrapper around reset, step, state, replay, and health endpoints."""

    base_url: str = "http://127.0.0.1:8080"
    session_id: Optional[str] = None
    timeout_seconds: int = 10

    def create_session(self, mode: str = "mini", seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> Observation:
        data = self._post("/api/sessions", {"mode": mode, "seed": seed, "config": config or {}})
        self.session_id = data["session_id"]
        return Observation.from_dict(data["observation"])

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        config: Optional[Config | Dict[str, Any]] = None,
    ) -> Observation:
        payload: Dict[str, Any] = {
            "seed": seed,
            "episode_id": episode_id,
            "session_id": self.session_id,
            "config": config.to_dict() if isinstance(config, Config) else config or {},
        }
        data = self._post("/reset", payload)
        self.session_id = data["session_id"]
        return Observation.from_dict(data["observation"])

    def step(self, action: Action | str | Dict[str, Any]) -> Observation:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._post("/step", {"session_id": self.session_id, "action": self._action_payload(action)})
        return Observation.from_dict(data["observation"])

    def fetch_state(self) -> State:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._get("/state", {"session_id": self.session_id})
        return State.model_validate(data["state"])

    def fetch_summary(self) -> Dict[str, Any]:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._get("/summary", {"session_id": self.session_id})
        return dict(data["episode_summary"])

    def fetch_replay(self) -> list[dict]:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._get(f"/api/sessions/{self.session_id}/replay")
        return list(data["records"])

    def health(self) -> Dict[str, Any]:
        return self._get("/healthz")

    def ready(self) -> Dict[str, Any]:
        return self._get("/ready")

    def state(self) -> State:
        return self.fetch_state()

    def _action_payload(self, action: Action | str | Dict[str, Any]) -> Any:
        if isinstance(action, Action):
            return action.to_dict()
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            return action
        raise TypeError("action must be Action, str, or dict")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            self.base_url.rstrip("/") + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request_json(req)

    def _get(self, path: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        if query:
            cleaned = {key: value for key, value in query.items() if value is not None}
            url += "?" + urlencode(cleaned)
        req = Request(url, method="GET")
        return self._request_json(req)

    def _request_json(self, req: Request) -> Dict[str, Any]:
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8") if exc.fp else str(exc)
            raise EnvClientError(f"HTTP {exc.code}: {message}") from exc


EnvClient = DispatchArenaClient
