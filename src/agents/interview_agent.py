"""LiveKit voice agent for AI interview — pipeline STT → LLM → TTS only."""

from __future__ import annotations

import logging
import time
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from ..conversation_export import interview_finalize_payload
from ..django_client import DjangoClient
from ..prompts.interview import interview_instructions, interview_opening_reply_instructions

logger = logging.getLogger(__name__)


class InterviewVoiceAgent(Agent):
    """Realtime interview — single pipeline handles all speech."""

    def __init__(
        self,
        *,
        session_id: str,
        django: DjangoClient,
        context: dict[str, Any],
    ):
        self._session_id = session_id
        self._django = django
        self._context = {**context, "session_id": session_id}
        self._finalized = False
        self._started_at = time.time()

        super().__init__(
            instructions=interview_instructions(
                company=context.get("company", "the company"),
                role=context.get("role", "this role"),
                difficulty_tier=context.get("difficulty_tier", "EASY"),
                mock_mode=context.get("mock_mode", "RECRUITER_WARMUP"),
                job_description=context.get("job_description", ""),
                resume_text=context.get("resume_text", ""),
                resume_anchors=context.get("resume_anchors", []) or [],
            ),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=interview_opening_reply_instructions(
                company=self._context.get("company", "the company"),
                role=self._context.get("role", "this role"),
            ),
            allow_interruptions=False,
        )

    @function_tool
    async def end_interview(self, context: RunContext) -> str:
        """End the interview when the candidate wants to stop."""
        await self.finalize(ended_by="user")
        return "Thank you. The interview is ending and your analysis will be ready shortly."

    async def finalize(self, ended_by: str = "agent") -> None:
        if self._finalized:
            return
        self._finalized = True
        duration = int(time.time() - self._started_at)
        payload = interview_finalize_payload(
            self,
            context=self._context,
            ended_by=ended_by,
            duration_seconds=duration,
        )
        try:
            await self._django.finalize_interview(self._session_id, payload)
            logger.info("Interview %s finalized (%s)", self._session_id, ended_by)
        except Exception as exc:
            logger.exception("Interview finalize failed: %s", exc)

    async def handle_end_signal(self) -> None:
        await self.finalize(ended_by="user")

    async def handle_disconnect(self) -> None:
        await self.finalize(ended_by="disconnect")
