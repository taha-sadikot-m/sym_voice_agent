"""Build LiveKit AgentSession with Deepgram STT/TTS and Gemini LLM."""

from __future__ import annotations

from livekit.agents import AgentSession
from livekit.plugins import deepgram, google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .config import settings


def create_agent_session() -> AgentSession:
    """Create a voice pipeline session (en-IN STT, Gemini 2.5 Flash, Deepgram TTS)."""
    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is required for the voice agent")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for the voice agent")

    return AgentSession(
        stt=deepgram.STT(
            model=settings.deepgram_stt_model,
            language=settings.deepgram_stt_language,
        ),
        llm=google.LLM(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
        ),
        tts=deepgram.TTS(
            model=settings.deepgram_tts_model,
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
