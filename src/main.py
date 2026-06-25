"""LiveKit voice agent worker entrypoint."""

from __future__ import annotations

import json
import logging

# Load .env / .env.local into os.environ before LiveKit reads LIVEKIT_*
from .config import settings
from . import config as _config  # noqa: F401

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli

from .agents.debate_agent import DebateVoiceAgent
from .agents.interview_agent import InterviewVoiceAgent
from .django_client import DjangoClient
from .session_factory import create_agent_session

logger = logging.getLogger(__name__)

server = AgentServer()


def _parse_room_metadata(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid room metadata JSON: %s", raw)
        return {}


@server.rtc_session(agent_name=settings.voice_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    metadata = _parse_room_metadata(ctx.room.metadata)
    mode = metadata.get("mode", "")
    logger.info(
        "Voice job started: room=%s mode=%s session_id=%s",
        ctx.room.name,
        mode,
        metadata.get("session_id", ""),
    )
    session_id = metadata.get("session_id", "")
    user_token = metadata.get("user_token", "")
    session_context = metadata.get("session_context") or {}

    if not session_id or not user_token:
        logger.error("Missing session_id or user_token in room metadata")
        return

    django = DjangoClient(user_token=user_token)
    agent_session = create_agent_session()

    if mode == "debate":
        agent = DebateVoiceAgent(
            session_id=session_id,
            django=django,
            context=session_context,
        )
    elif mode == "interview":
        agent = InterviewVoiceAgent(
            session_id=session_id,
            django=django,
            context=session_context,
        )
    else:
        logger.error("Unknown voice mode: %s", mode)
        return

    @ctx.room.on("data_received")
    def on_data(data: rtc.DataPacket) -> None:
        try:
            payload = json.loads(data.data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if payload.get("type") == "end_session":
            import asyncio

            asyncio.create_task(agent.handle_end_signal())

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        if participant.identity == ctx.room.local_participant.identity:
            return
        import asyncio

        asyncio.create_task(agent.handle_disconnect())

    try:
        await agent_session.start(agent=agent, room=ctx.room)
    finally:
        if not getattr(agent, "_finalized", False):
            await agent.finalize(ended_by="disconnect")


def run() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    run()
