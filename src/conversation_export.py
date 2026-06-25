"""Export LiveKit agent chat context for Django finalize payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from livekit.agents import Agent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def chat_context_to_messages(agent: Agent) -> list[dict[str, Any]]:
    """Map agent chat history to Django voice finalize message format."""
    messages: list[dict[str, Any]] = []
    for item in agent.chat_ctx.items:
        if getattr(item, "type", None) != "message":
            continue
        role_raw = getattr(item, "role", None)
        if role_raw not in ("user", "assistant"):
            continue
        text = (getattr(item, "text_content", None) or "").strip()
        if not text:
            continue
        messages.append(
            {
                "role": "user" if role_raw == "user" else "ai",
                "content": text,
                "timestamp": datetime.fromtimestamp(
                    getattr(item, "created_at", 0), tz=timezone.utc
                ).isoformat()
                if getattr(item, "created_at", None)
                else _now_iso(),
            }
        )
    return messages


def debate_finalize_payload(agent: Agent, *, ended_by: str, duration_seconds: int) -> dict[str, Any]:
    return {
        "messages": chat_context_to_messages(agent),
        "ended_by": ended_by,
        "duration_seconds": duration_seconds,
    }


def interview_finalize_payload(
    agent: Agent,
    *,
    context: dict[str, Any],
    ended_by: str,
    duration_seconds: int,
) -> dict[str, Any]:
    messages = chat_context_to_messages(agent)
    agent_state: dict[str, Any] = {
        "session_id": context.get("session_id", ""),
        "company": context.get("company", ""),
        "role": context.get("role", ""),
        "difficulty_tier": context.get("difficulty_tier", "EASY"),
        "mock_mode": context.get("mock_mode", "RECRUITER_WARMUP"),
        "job_description": context.get("job_description", ""),
        "resume_text": context.get("resume_text", ""),
        "resume_anchors": context.get("resume_anchors", []),
        "messages": [
            {
                "role": "user" if m["role"] == "user" else "assistant",
                "content": m["content"],
                "type": "answer" if m["role"] == "user" else "question",
                "timestamp": m.get("timestamp", _now_iso()),
            }
            for m in messages
        ],
        "termination_reason": ended_by if ended_by in ("user", "disconnect") else "natural",
        "should_continue": False,
    }
    return {
        "agent_state": agent_state,
        "messages": messages,
        "ended_by": ended_by,
        "duration_seconds": duration_seconds,
    }
