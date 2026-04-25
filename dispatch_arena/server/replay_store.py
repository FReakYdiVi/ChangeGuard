"""JSONL replay persistence for Dispatch Arena sessions."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dispatch_arena.models import Action, Observation


DEFAULT_REPLAY_DIR = Path(tempfile.gettempdir()) / "dispatch_arena_replays"
REPLAY_DIR_ENV = "DISPATCH_ARENA_REPLAY_DIR"


@dataclass
class ReplayStore:
    root: Path = field(default_factory=lambda: Path(os.getenv(REPLAY_DIR_ENV, DEFAULT_REPLAY_DIR)))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def start_session(self, session_id: str) -> None:
        with self._lock:
            self._path(session_id).write_text("", encoding="utf-8")

    def append_reset(self, session_id: str, observation: Observation) -> None:
        self.append(
            session_id,
            {
                "type": "reset",
                "tick": observation.state.tick,
                "observation": observation.to_dict(),
            },
        )

    def append_step(self, session_id: str, action: Action, observation: Observation) -> None:
        self.append(
            session_id,
            {
                "type": "step",
                "tick": observation.state.tick,
                "action": action.to_dict(),
                "observation": observation.to_dict(),
                "reward_breakdown": observation.reward_breakdown.to_dict(),
                "events": observation.info.get("events", []),
            },
        )

    def append_summary(self, session_id: str, summary: Dict[str, Any]) -> None:
        self.append(session_id, {"type": "summary", "episode_summary": summary})

    def append(self, session_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            with self._path(session_id).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        with self._lock:
            lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
        return self.root / f"{safe}.jsonl"
