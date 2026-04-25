"""Request models for Dispatch Arena's REST API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from dispatch_arena.models import Action, Config, Mode


class CreateSessionRequest(BaseModel):
    mode: Mode = Mode.MINI
    seed: Optional[int] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    def resolved_config(self, default_max_ticks: int) -> Config:
        payload = dict(self.config)
        payload.setdefault("mode", self.mode)
        payload.setdefault("max_ticks", default_max_ticks)
        return Config.model_validate(payload)


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    session_id: Optional[str] = None
    mode: Optional[Mode] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    def resolved_config(self, fallback: Config) -> Config:
        payload = fallback.to_dict()
        payload.update(self.config)
        if self.mode is not None:
            payload["mode"] = self.mode
        return Config.model_validate(payload)


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Action | str | Dict[str, Any]
