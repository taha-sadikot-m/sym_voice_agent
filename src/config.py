"""Environment configuration for the voice agent worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# voice-agent/ root — support both .env and .env.local (local overrides)
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    deepgram_api_key: str
    deepgram_stt_model: str
    deepgram_stt_language: str
    deepgram_tts_model: str
    gemini_api_key: str
    gemini_model: str
    django_api_url: str
    voice_agent_name: str

    @classmethod
    def load(cls) -> "Settings":
        gemini = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        return cls(
            livekit_url=_env("LIVEKIT_URL"),
            livekit_api_key=_env("LIVEKIT_API_KEY"),
            livekit_api_secret=_env("LIVEKIT_API_SECRET"),
            deepgram_api_key=_env("DEEPGRAM_API_KEY"),
            deepgram_stt_model=_env("DEEPGRAM_STT_MODEL", "nova-3"),
            deepgram_stt_language=_env("DEEPGRAM_STT_LANGUAGE", "en-IN"),
            deepgram_tts_model=_env("DEEPGRAM_TTS_MODEL", "aura-2-draco-en"),
            gemini_api_key=gemini,
            gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
            django_api_url=_env("DJANGO_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/"),
            voice_agent_name=_env("VOICE_AGENT_NAME", "sym-voice-agent"),
        )


settings = Settings.load()
