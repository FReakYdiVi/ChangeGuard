"""FastAPI assembly for Dispatch Arena."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dispatch_arena.models import Action, Config, Observation, State
from dispatch_arena.server.api import CreateSessionRequest, ResetRequest, StepRequest
from dispatch_arena.server.env import DEFAULT_MAX_TICKS, DispatchArenaEnvironment
from dispatch_arena.server.metrics import episode_metrics
from dispatch_arena.server.replay_store import ReplayStore
from dispatch_arena.server.scenarios import scenario_catalog

SUPPORTS_CONCURRENT_SESSIONS = True
DEFAULT_MAX_CONCURRENT_ENVS = 16
MAX_CONCURRENT_ENVS_ENV = "DISPATCH_ARENA_MAX_CONCURRENT_ENVS"


@dataclass
class _Session:
    env: DispatchArenaEnvironment
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class DispatchArenaServerApp:
    """Session manager used by REST, WebSocket, and OpenEnv endpoints."""

    max_concurrent_envs: int = DEFAULT_MAX_CONCURRENT_ENVS
    max_ticks: int = DEFAULT_MAX_TICKS
    replay_store: ReplayStore = field(default_factory=ReplayStore)
    _sessions: Dict[str, _Session] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create_session(self, config: Config, seed: Optional[int] = None) -> tuple[str, Observation]:
        session_id = str(uuid.uuid4())
        with self._lock:
            if len(self._sessions) >= self.max_concurrent_envs:
                raise RuntimeError(f"Max concurrent envs reached ({self.max_concurrent_envs}).")
            env = DispatchArenaEnvironment(config=config)
            self._sessions[session_id] = _Session(env=env)
            self.replay_store.start_session(session_id)
        obs = self.reset(session_id=session_id, seed=seed, config=config, episode_id=session_id)
        return session_id, obs

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> Observation:
        session = self._get_or_create_session(session_id=session_id, config=config)
        resolved_session_id = self._session_id_for(session)
        with session.lock:
            obs = session.env.reset(seed=seed, episode_id=episode_id or resolved_session_id, config=config)
            self.replay_store.start_session(resolved_session_id)
            self.replay_store.append_reset(resolved_session_id, obs)
            return obs

    def step(self, session_id: str, action: Action | str | dict) -> Observation:
        session = self._require_session(session_id)
        with session.lock:
            parsed = action if isinstance(action, Action) else Action.from_dict(action) if isinstance(action, dict) else Action(action_type=action)
            obs = session.env.step(parsed)
            self.replay_store.append_step(session_id, parsed, obs)
            if obs.done:
                self.replay_store.append_summary(session_id, session.env.get_episode_summary())
            return obs

    def state(self, session_id: str) -> State:
        session = self._require_session(session_id)
        with session.lock:
            return session.env.state

    def summary(self, session_id: str) -> dict:
        session = self._require_session(session_id)
        with session.lock:
            summary = session.env.get_episode_summary()
            summary["metrics"] = episode_metrics(session.env.state)
            return summary

    def replay(self, session_id: str) -> list[dict]:
        self._require_session(session_id)
        return self.replay_store.load(session_id)

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "dispatch_arena",
            "supports_concurrent_sessions": SUPPORTS_CONCURRENT_SESSIONS,
        }

    def ready(self) -> dict:
        with self._lock:
            return {
                "ready": True,
                "active_sessions": len(self._sessions),
                "max_concurrent_envs": self.max_concurrent_envs,
            }

    def _get_or_create_session(self, session_id: Optional[str], config: Optional[Config]) -> _Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            if len(self._sessions) >= self.max_concurrent_envs:
                raise RuntimeError(f"Max concurrent envs reached ({self.max_concurrent_envs}).")
            resolved_session_id = session_id or str(uuid.uuid4())
            env = DispatchArenaEnvironment(config=config or Config(max_ticks=self.max_ticks))
            session = _Session(env=env)
            self._sessions[resolved_session_id] = session
            self.replay_store.start_session(resolved_session_id)
            return session

    def _session_id_for(self, session: _Session) -> str:
        with self._lock:
            for session_id, candidate in self._sessions.items():
                if candidate is session:
                    return session_id
        raise KeyError("Unknown session")

    def _require_session(self, session_id: str) -> _Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            return self._sessions[session_id]


def create_app(
    max_concurrent_envs: Optional[int] = None,
    max_ticks: int = DEFAULT_MAX_TICKS,
) -> FastAPI:
    if max_concurrent_envs is None:
        max_concurrent_envs = int(os.getenv(MAX_CONCURRENT_ENVS_ENV, str(DEFAULT_MAX_CONCURRENT_ENVS)))
    manager = DispatchArenaServerApp(max_concurrent_envs=max_concurrent_envs, max_ticks=max_ticks)
    app = FastAPI(title="Dispatch Arena", version="0.1.0")
    app.state.dispatch_arena = manager

    @app.get("/healthz")
    def healthz() -> dict:
        return manager.health()

    @app.get("/health")
    def health() -> dict:
        return manager.health()

    @app.get("/ready")
    def ready() -> dict:
        return manager.ready()

    @app.post("/api/sessions")
    def api_create_session(payload: CreateSessionRequest) -> dict:
        try:
            session_id, obs = manager.create_session(payload.resolved_config(max_ticks), seed=payload.seed)
            return {"session_id": session_id, "observation": obs.to_dict()}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/reset")
    def api_reset(session_id: str, payload: ResetRequest | None = None) -> dict:
        payload = payload or ResetRequest()
        try:
            fallback = manager._require_session(session_id).env.config if session_id in manager._sessions else Config(max_ticks=max_ticks)
            obs = manager.reset(
                session_id=session_id,
                seed=payload.seed,
                episode_id=payload.episode_id or session_id,
                config=payload.resolved_config(fallback),
            )
            return {"session_id": session_id, "observation": obs.to_dict()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/step")
    def api_step(session_id: str, payload: StepRequest) -> dict:
        try:
            obs = manager.step(session_id=session_id, action=payload.action)
            return {"session_id": session_id, "observation": obs.to_dict()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/state")
    def api_state(session_id: str) -> dict:
        try:
            return {"session_id": session_id, "state": manager.state(session_id).to_dict()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/replay")
    def api_replay(session_id: str) -> dict:
        try:
            return {"session_id": session_id, "records": manager.replay(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/scenarios")
    def api_scenarios() -> dict:
        return {"scenarios": scenario_catalog()}

    @app.websocket("/api/sessions/{session_id}/stream")
    async def api_stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "state": manager.state(session_id).to_dict(),
                        "replay": manager.replay(session_id),
                    }
                )
                await websocket.receive_text()
        except (WebSocketDisconnect, KeyError):
            return

    @app.post("/reset")
    def openenv_reset(payload: ResetRequest) -> dict:
        config = Config(max_ticks=max_ticks)
        if payload.config or payload.mode is not None:
            config = payload.resolved_config(config)
        try:
            session_id = payload.session_id or str(uuid.uuid4())
            obs = manager.reset(
                session_id=session_id,
                seed=payload.seed,
                episode_id=payload.episode_id or session_id,
                config=config,
            )
            return {"session_id": session_id, "observation": obs.to_dict()}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/step")
    def openenv_step(payload: StepRequest) -> dict:
        if not payload.session_id:
            raise HTTPException(status_code=400, detail="Missing required field: session_id")
        try:
            obs = manager.step(session_id=payload.session_id, action=payload.action)
            return {"session_id": payload.session_id, "observation": obs.to_dict()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/state")
    def openenv_state(session_id: str) -> dict:
        try:
            return {"session_id": session_id, "state": manager.state(session_id).to_dict()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/summary")
    def openenv_summary(session_id: str) -> dict:
        try:
            return {"session_id": session_id, "episode_summary": manager.summary(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir / "assets"
    if static_dir.exists():
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app


class _UvicornThreadHandle:
    def __init__(self, server: uvicorn.Server, thread: threading.Thread, address: tuple[str, int]) -> None:
        self._server = server
        self._thread = thread
        self.server_address = address

    def shutdown(self) -> None:
        self._server.should_exit = True

    def server_close(self) -> None:
        return None


def run_local_server(host: str = "127.0.0.1", port: int = 8080, max_concurrent_envs: Optional[int] = None) -> Tuple[str, int]:
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    uvicorn.run(app, host=host, port=port)
    return host, port


def run_local_server_in_thread(host: str = "127.0.0.1", port: int = 0, max_concurrent_envs: Optional[int] = None):
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = sock.getsockname()[1]
    app = create_app(max_concurrent_envs=max_concurrent_envs)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return _UvicornThreadHandle(server, thread, (host, port)), thread


def main() -> None:
    port = int(os.getenv("PORT", "8080"))
    run_local_server(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
