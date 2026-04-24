"""Typed client for ChangeGuard local server API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import (
    Action,
    ActionType,
    EpisodeConfig,
    EpisodeStats,
    EpisodeSummary,
    Observation,
    RewardBreakdown,
    StepResult,
    VerifierFlags,
    VerifierVerdict,
)


class EnvClientError(RuntimeError):
    """Raised when server returns non-success responses."""


@dataclass
class EnvClient:
    """Stable typed client for reset/step/state operations."""

    base_url: str = "http://127.0.0.1:8080"
    session_id: Optional[str] = None
    timeout_seconds: int = 10

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "easy",
        scenario_id: Optional[str] = None,
        prompt_style: Optional[str] = None,
    ) -> Observation:
        payload: Dict[str, Any] = {
            "seed": seed,
            "difficulty": difficulty,
            "session_id": self.session_id,
            "scenario_id": scenario_id,
            "prompt_style": prompt_style,
        }
        data = self._post("/reset", payload)
        self.session_id = data["session_id"]
        return Observation.from_dict(data["observation"])

    def step(self, action: Action | ActionType | str) -> StepResult:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        action_payload = self._action_payload(action)
        data = self._post("/step", {"session_id": self.session_id, "action": action_payload})
        return self._parse_step_result(data["step_result"])

    def fetch_state(self) -> Observation:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._get("/state", {"session_id": self.session_id})
        return Observation.from_dict(data["observation"])

    def fetch_summary(self) -> EpisodeSummary:
        if not self.session_id:
            raise EnvClientError("Session not initialized. Call reset() first.")
        data = self._get("/summary", {"session_id": self.session_id})
        return self._parse_episode_summary(data["episode_summary"])

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def ready(self) -> Dict[str, Any]:
        return self._get("/ready")

    # Backward-compatible aliases used by existing wrappers.
    def state(self) -> Observation:
        return self.fetch_state()

    def call_tool(self, action: Action) -> StepResult:
        return self.step(action)

    def episode_summary(self) -> EpisodeSummary:
        return self.fetch_summary()

    def _action_payload(self, action: Action | ActionType | str) -> Any:
        if isinstance(action, Action):
            return action.to_dict()
        if isinstance(action, ActionType):
            return action.value
        if isinstance(action, str):
            return action
        raise TypeError("action must be Action, ActionType, or str")

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
            cleaned = {k: v for k, v in query.items() if v is not None}
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

    def _parse_step_result(self, data: Dict[str, Any]) -> StepResult:
        return StepResult(
            observation=Observation.from_dict(data["observation"]),
            reward_total=float(data.get("reward_total", 0.0)),
            reward_breakdown=RewardBreakdown(**dict(data.get("reward_breakdown", {}))),
            done=bool(data.get("done", False)),
            truncated=bool(data.get("truncated", False)),
            verifier_flags=VerifierFlags(**dict(data.get("verifier_flags", {}))),
            info=dict(data.get("info", {})),
        )

    def _parse_episode_summary(self, data: Dict[str, Any]) -> EpisodeSummary:
        config = EpisodeConfig(**dict(data.get("config", {})))
        stats = EpisodeStats(**dict(data.get("stats", {})))
        return EpisodeSummary(
            episode_id=data.get("episode_id"),
            config=config,
            stats=stats,
            final_verdict=VerifierVerdict(data.get("final_verdict", VerifierVerdict.IN_PROGRESS.value)),
            final_reward=float(data.get("final_reward", 0.0)),
            action_trace=list(data.get("action_trace", [])),
        )


# Backward-compatible client name for existing code.
ChangeGuardClient = EnvClient
