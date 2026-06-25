"""HTTP client for Django voice finalize APIs only."""

from __future__ import annotations

from typing import Any

import httpx

from .config import settings


class DjangoClient:
    """Finalize-only client — no per-turn conversation calls."""

    def __init__(self, user_token: str, timeout: float = 120.0):
        self._token = user_token
        self._timeout = timeout
        self._base = settings.django_api_url

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, json: dict | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base}{path}",
                headers=self._headers(),
                json=json or {},
            )
            response.raise_for_status()
            if response.status_code == 204:
                return {}
            return response.json()

    async def finalize_debate(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/voice/debate/{session_id}/finalize/", payload)

    async def finalize_interview(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/voice/interview/{session_id}/finalize/", payload)
