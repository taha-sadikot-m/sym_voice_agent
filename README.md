# SYM LiveKit Voice Agent



Standalone voice worker for **Debate with AI** and **Interview with AI**. One agent per room runs the full realtime conversation through a single **STT → Gemini LLM → TTS** pipeline. Django is not in the turn loop — only setup, token issuance, and end-of-session finalize + analysis.



## Architecture



```

Setup (Django) → Token + room metadata → LiveKit room

During session: Voice worker only (Deepgram STT → Gemini → Deepgram TTS)

End: Agent POST /voice/.../finalize/ (transcript from chat context) → Django → analysis

Frontend: Poll /voice/sessions/{mode}/{id}/status/ → load results from DB

```



| Phase | Django | Voice agent |

|-------|--------|-------------|

| Start | Create session shell, issue JWT with `RoomAgentDispatch` | Load config from room metadata |

| Conversation | **No HTTP** (per-turn APIs return 410 for voice) | Pipeline handles every spoken reply |

| End | Persist messages/state, run analysis | `POST /voice/debate|interview/{id}/finalize/` |



**Important:** Do not run two workers with the same `VOICE_AGENT_NAME` (e.g. local `python -m src.main dev` **and** a LiveKit Cloud deployment). That spawns duplicate agents in one room and you will hear overlapping voices.



Agent dispatch uses **only** the participant token `room_config` (`RoomAgentDispatch`). Django does not call `create_dispatch` separately.



## Prerequisites



- Python 3.10+ (3.11–3.12 recommended; 3.14 works but LangChain may warn about Pydantic v1)

- LiveKit Cloud, Deepgram, Gemini API keys

- Django backend with matching `LIVEKIT_*` and `VOICE_FINALIZE_SYNC=true` for local dev



## Python setup (shared repo `.venv`)



If you use one virtualenv at the repo root for both Django and the voice worker, install **both** dependency sets:



```bash

# From repo root with .venv activated

pip install -r backend/requirements.txt   # Django + livekit-api (tokens)

pip install -e voice-agent/               # livekit-agents + plugins (worker)

```



Backend needs `livekit-api`; the worker needs `livekit-agents` — they are different packages.



First-time worker setup also downloads VAD and turn-detector weights:



```bash

cd voice-agent

python -m livekit.agents download-files

```



## Local development (3 processes)



```bash

# Terminal 1 — Django

cd backend && python manage.py runserver



# Terminal 2 — Frontend

cd frontend && npm run dev



# Terminal 3 — Voice agent (only one worker for VOICE_AGENT_NAME)

cd voice-agent && cp .env.example .env.local

pip install -e .

python -m livekit.agents download-files

python -m src.main dev

```



Credentials can live in `.env` or `.env.local` (`.env.local` overrides).



## Environment



| Variable | Description |

|----------|-------------|

| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud |

| `VOICE_AGENT_NAME` | Must match backend (`sym-voice-agent`); worker registers with this name |

| `DEEPGRAM_API_KEY` | STT (`en-IN`) + TTS (`aura-2-draco-en`) |

| `GEMINI_API_KEY` | Gemini 2.5 Flash (all realtime replies) |

| `DJANGO_API_URL` | e.g. `http://127.0.0.1:8000/api/v1` |



Backend also needs `VOICE_FINALIZE_SYNC=true` in DEBUG (default) so analysis runs without Celery.



## Voice pipeline



- **STT:** Deepgram Nova-3, `en-IN`

- **LLM:** Gemini 2.5 Flash via `google.LLM` in `AgentSession` — the model **is** the debate opponent / interviewer

- **TTS:** Deepgram Aura-2 `aura-2-draco-en`

- **Turn handling:** Framework auto-generates one reply per user turn after STT; agents do **not** call `session.say()` or per-turn Django APIs

- **Opening:** `on_enter` → `session.generate_reply()` once (AI opens or user opens per debate config)



## Finalize flow



1. User clicks **End Session** → frontend sends LiveKit data `{ type: "end_session" }`

2. Agent exports `chat_ctx` → `POST /api/v1/voice/debate/{id}/finalize/` or `.../interview/{id}/finalize/`

3. Django persists transcript, enqueues analysis (`VOICE_FINALIZE_SYNC` or Celery)

4. Frontend polls `GET /api/v1/voice/sessions/{mode}/{id}/status/` until `COMPLETED`



## Troubleshooting



| Symptom | Likely cause |

|---------|----------------|

| Two voices / overlapping replies | Duplicate worker (local + cloud) or old code path with `session.say()` + pipeline LLM |

| Agent keeps talking after leave | Orphan `<audio>` elements — frontend hook removes them on disconnect; restart tab if needed |

| No agent joins | Worker not running, or `VOICE_AGENT_NAME` mismatch between backend and worker |

| No transcript after end | Finalize failed — check worker logs and Django `voice_finalize_status` in session metadata |



## Deploy



```bash

docker build -t sym-voice-agent .

docker run --env-file .env sym-voice-agent

```



Register **one** worker in LiveKit Cloud with entrypoint `python -m src.main start`. Disable or use a different agent name for local dev while cloud worker is active.


