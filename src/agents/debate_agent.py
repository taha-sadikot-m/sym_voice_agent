"""LiveKit voice agent for AI debate — pipeline STT → LLM → TTS only."""

from __future__ import annotations

import logging
import time
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from ..conversation_export import debate_finalize_payload
from ..django_client import DjangoClient
from ..prompts.debate import debate_instructions, debate_opening_reply_instructions

logger = logging.getLogger(__name__)


class DebateVoiceAgent(Agent):
    """Realtime debate opponent — single pipeline handles all speech."""

    def __init__(
        self,
        *,
        session_id: str,
        django: DjangoClient,
        context: dict[str, Any],
    ):
        self._session_id = session_id
        self._django = django
        self._finalized = False
        self._started_at = time.time()

        self._config = {
            "topic": context.get("topic", "the topic"),
            "topic_type": context.get("topic_type", "TOPIC"),
            "difficulty": context.get("difficulty", "MEDIUM"),
            "creator_position": context.get("creator_position", "PRO"),
            "opponent_position": context.get("opponent_position", "CON"),
            "opening_speaker": context.get("opening_speaker", "CREATOR"),
        }
        self._ai_opens = self._config["opening_speaker"] == "OPPONENT"

        super().__init__(
            instructions=debate_instructions(
                topic=self._config["topic"],
                difficulty=self._config["difficulty"],
                user_position=self._config["creator_position"],
                ai_position=self._config["opponent_position"],
                opening_speaker=self._config["opening_speaker"],
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=debate_opening_reply_instructions(
                ai_opens=self._ai_opens,
                topic=self._config["topic"],
                ai_position=self._config["opponent_position"],
                difficulty=self._config["difficulty"],
            ),
            allow_interruptions=True,
        )

    @function_tool
    async def end_debate(self, context: RunContext) -> str:
        """End the debate when the user wants to stop."""
        await self.finalize(ended_by="user")
        return "Understood. Your debate is ending and analysis will be ready shortly."

    async def finalize(self, ended_by: str = "agent") -> None:
        if self._finalized:
            return
        self._finalized = True
        duration = int(time.time() - self._started_at)
        payload = debate_finalize_payload(self, ended_by=ended_by, duration_seconds=duration)
        try:
            await self._django.finalize_debate(self._session_id, payload)
            logger.info("Debate %s finalized (%s)", self._session_id, ended_by)
        except Exception as exc:
            logger.exception("Debate finalize failed: %s", exc)

    async def handle_end_signal(self) -> None:
        await self.finalize(ended_by="user")

    async def handle_disconnect(self) -> None:
        await self.finalize(ended_by="disconnect")
