"""In-memory session state for voice agent conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DebateSessionState:
    session_id: str
    config: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    ended_by: str | None = None

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content, "timestamp": _now_iso()})

    def history_for_ai(self) -> list[dict[str, Any]]:
        return [
            {"is_user": m["role"] == "user", "content": m["content"]}
            for m in self.messages
        ]

    def finalize_payload(self, ended_by: str = "agent") -> dict[str, Any]:
        self.ended_by = ended_by
        return {
            "messages": self.messages,
            "ended_by": ended_by,
        }


@dataclass
class InterviewSessionState:
    session_id: str
    agent_state: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now_iso)
    ended_by: str | None = None

    def finalize_payload(self, ended_by: str = "agent") -> dict[str, Any]:
        self.ended_by = ended_by
        return {
            "agent_state": self.agent_state,
            "messages": self.agent_state.get("messages", []),
            "questions_asked": self.agent_state.get("questions_asked", []),
            "ended_by": ended_by,
        }
