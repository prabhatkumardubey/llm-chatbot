# llmchat: a provider-agnostic LLM chatbot

A small, production-minded chatbot service. One streaming chat API sits in front of **Claude, OpenAI, any OpenAI-compatible server (Ollama, vLLM, Groq) or an offline mock model**. Conversations are saved in SQLite, and a minimal web UI streams replies token by token. Optionally, answers can be **grounded in documents you add (RAG)** with cited sources.

Built as an engineering portfolio project, so the documentation is as important as the code:

| Document | Read it for |
| --- | --- |
| [docs/FUNCTIONAL_SPEC.md](docs/FUNCTIONAL_SPEC.md) | What the system does: requirements, user flows, API contract, traceability to tests |
| [docs/TECH_SPEC.md](docs/TECH_SPEC.md) | How and why: architecture, stack choices and alternatives, data model, algorithms, design patterns, SOLID, OOP, concurrency, security, limits |
| [docs/RAG_SPEC.md](docs/RAG_SPEC.md) | Phase 2: retrieval-augmented answers over your own documents, plus hybrid keyword + vector search (specs written before the code) |
| [docs/AUTH_SPEC.md](docs/AUTH_SPEC.md) | Phase 3: accounts, per-user data isolation, rate limiting, threat model (spec written before the code) |

## Quick start (no API key needed)
```bash
pip install -r requirements-dev.txt
uvicorn app.main:create_app --factory --reload      # http://localhost:8000
pytest -q
```
With no configuration the offline `mock` provider answers, so you can try everything immediately. Open the page, **register** an account, and chat. Authentication is on by default; for a purely personal run set `AUTH_REQUIRED=false` (everything then belongs to one built-in user). Never expose a server with authentication off.

## Use a real model
Copy `.env.example` to `.env`, set values in your shell or Docker, and pick the provider in the UI dropdown.
```bash
export ANTHROPIC_API_KEY=...        # enables "anthropic"
export OPENAI_API_KEY=...           # enables "openai"
export OLLAMA_MODEL=llama3.1        # enables "ollama" (local, OpenAI-compatible)
export DEFAULT_PROVIDER=anthropic
```
Model names are configurable (`ANTHROPIC_MODEL`, `OPENAI_MODEL`), because model names change often.

## API
| Method and path | Purpose |
| --- | --- |
| `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` | Accounts and sessions (bearer token). Every other data endpoint needs `Authorization: Bearer <token>` |
| `POST /api/chat` | Send a message, receive a Server-Sent Events stream (`meta`, `token`..., `done` or `error`) |
| `GET /api/conversations` | List conversations, newest first |
| `GET /api/conversations/{id}` | Load one conversation's messages |
| `DELETE /api/conversations/{id}` | Delete a conversation and its messages |
| `POST /api/documents`, `GET /api/documents`, `DELETE /api/documents/{id}` | Add, list and delete documents for retrieval (RAG). Send `"use_docs": true` to `/api/chat` to answer from them; a `sources` event lists the citations |
| `GET /api/providers` | Configured providers and the default |
| `GET /health` | Liveness check |

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/register -H 'Content-Type: application/json' -d '{"username":"demo","password":"a long password"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -N -X POST localhost:8000/api/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"message":"hello","provider":"mock"}'
```

## Docker
```bash
docker build -t llmchat . && docker run -p 8000:8000 -v llmchat-data:/data -e ANTHROPIC_API_KEY llmchat
```

## Layout
```
app/config.py      settings from environment        app/providers.py  LLM strategy classes + registry
app/context.py     context-window trimming          app/store.py      SQLite repository
app/rag.py         chunking, embeddings, vector search, prompt grounding
app/auth.py        users, scrypt password hashing, sessions     app/ratelimit.py  token-bucket rate limiter
app/main.py        FastAPI app factory and routes   static/index.html web UI (vanilla JS, SSE client)
tests/             pytest suite (mock provider)     docs/             specs (functional, technical, RAG, auth)
```

## Status
Implemented: streaming chat, multi-provider, persistence, context trimming, validation, error handling, UI, tests, Docker, CI. Phase 2 (RAG) is implemented with **hybrid retrieval** (keyword BM25 + vector search merged by reciprocal rank fusion), an offline word-overlap embedder by default and an optional OpenAI embedder; see `docs/RAG_SPEC.md` for its limits. Phase 3 adds registration/login, per-user data isolation and rate limiting (see `docs/AUTH_SPEC.md`; hand-built with scrypt and server-side sessions, and a managed identity provider is the usual production choice). Not implemented (see the roadmap in the tech spec): email verification, password reset, MFA, tool use, token accounting, observability.
