"""Debate AI service for voice agent (ported from Django, no ORM)."""

from __future__ import annotations

import json
import logging
from typing import Dict

from google import genai

from ..config import settings

logger = logging.getLogger(__name__)


class DebateAIService:
    """Gemini-powered debate opponent for realtime voice sessions."""

    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key
        self.model_name = f"models/{settings.gemini_model.lstrip('models/')}"
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def is_available(self) -> bool:
        return self.client is not None

    def generate_opening_statement(self, topic: str, position: str, difficulty: str = "MEDIUM") -> str:
        if not self.is_available():
            return self._fallback_opening(position)
        try:
            stance = "supporting" if position == "PRO" else "opposing"
            difficulty_instructions = {
                "EASY": "Use simple arguments with clear logic. Keep it conversational and friendly.",
                "MEDIUM": "Use well-reasoned arguments with some evidence. Be professional and engaging.",
                "HARD": "Use sophisticated arguments with statistical evidence and complex reasoning.",
            }
            prompt = f"""You are an expert debater in a {difficulty.lower()} difficulty debate match.
Generate a compelling opening statement (60-80 words) {stance} this topic:

"{topic}"

{difficulty_instructions.get(difficulty, difficulty_instructions['MEDIUM'])}

Use clear Indian English phrasing. No quotation marks."""
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text.strip()
        except Exception as exc:
            logger.error("Opening generation failed: %s", exc)
            return self._fallback_opening(position)

    def generate_response(
        self,
        topic: str,
        position: str,
        difficulty: str,
        conversation_history: list,
        user_last_message: str,
    ) -> str:
        if not self.is_available():
            return self._fallback_response()
        try:
            stance = "supporting" if position == "PRO" else "opposing"
            strategies = {
                "EASY": {"tone": "friendly", "tactics": "Basic logic and examples.", "length": "40-60 words"},
                "MEDIUM": {"tone": "professional", "tactics": "Logical reasoning with examples.", "length": "60-90 words"},
                "HARD": {"tone": "rigorous", "tactics": "Sophisticated argumentation and rhetoric.", "length": "80-120 words"},
            }
            strategy = strategies.get(difficulty, strategies["MEDIUM"])
            context = "\n".join(
                f"{'User' if msg['is_user'] else 'AI'}: {msg['content']}" for msg in conversation_history
            )
            message_count = len(conversation_history)
            debate_stage = "opening" if message_count <= 2 else "middle" if message_count <= 6 else "closing"
            prompt = f"""You are a specialized AI debate opponent {stance} this topic:

"{topic}"

Difficulty: {difficulty} | Position: {position} | Stage: {debate_stage}
Strategy: {strategy['tactics']} | Tone: {strategy['tone']}

CONVERSATION:
{context}

USER'S LATEST ARGUMENT:
{user_last_message}

Respond in {strategy['length']} using clear Indian English. No quotation marks."""
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text.strip()
        except Exception as exc:
            logger.error("Debate response failed: %s", exc)
            return self._fallback_response()

    def _fallback_opening(self, position: str) -> str:
        if position == "PRO":
            return "I'm ready to present compelling arguments in support of this motion. Let's begin."
        return "I'm prepared to challenge this motion with strong counter-arguments. Let's begin."

    def _fallback_response(self) -> str:
        return (
            "That's a thought-provoking argument. While I see your perspective, "
            "I'd like to present an alternative viewpoint for us to consider."
        )
