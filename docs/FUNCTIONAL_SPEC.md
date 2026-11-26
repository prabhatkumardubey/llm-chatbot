# Functional Specification

> **Provenance note:** the core implementation was written first and this specification was reconciled against the code afterwards, with a traceability table at the end. It is the contract for the current version and the baseline for phase 2.

## 1. Purpose and scope
`llmchat` lets a person hold multi-turn conversations with a large language model (LLM) through a web UI or an HTTP API, choosing among interchangeable model providers, with history saved between sessions.

**In scope (v0.1):** streamed chat, multi-turn context, persistent history, provider selection, configurable system prompt, input validation, error handling, a minimal UI.
**Out of scope (v0.1 core):** email verification, password reset, MFA, roles and organisations, file or image input, usage billing, mobile apps.

## 2. Actors
| Actor | Goal |
| --- | --- |
| End user | Ask questions and continue earlier conversations |
| Operator (developer) | Run the service, configure providers and limits, keep keys out of the code |
| Contributor | Add a provider or feature safely, with tests |

## 3. Functional requirements
| ID | Requirement | Acceptance criteria | Status |
| --- | --- | --- | --- |
| FR-1 | Streamed reply | Reply text arrives incrementally as `token` events, followed by `done` | Done |
| FR-2 | Multi-turn context | A follow-up with `conversation_id` includes earlier turns in the model prompt | Done |
| FR-3 | Persistent history | Conversations can be listed (newest first), loaded and deleted; they survive a restart | Done |
| FR-4 | Provider choice | Each request may name a provider; unconfigured or unknown names return 400 | Done |
| FR-5 | System prompt | A default comes from configuration; a request may override it (max 2000 chars) | Done |
| FR-6 | Context window management | The prompt never exceeds the configured token budget; the newest message is always kept; the oldest turns drop first | Done |
| FR-7 | Input validation | Messages must be 1 to 8000 characters; invalid input returns 422 | Done |
| FR-8 | Error behaviour | Unknown conversation returns 404; a failure mid-stream sends an `error` event and saves nothing for that exchange | Done |
| FR-9 | Web UI | New chat, conversation list, provider dropdown, streaming display, delete | Done |
| FR-10 | Health check | `GET /health` returns 200 `{"status":"ok"}` | Done |
| FR-11 | Offline mode | A mock provider works without keys, for demos and tests | Done |
| FR-12 | Authentication and per-user history | Users see only their own conversations and documents | Done (phase 3, see [AUTH_SPEC](AUTH_SPEC.md)) |
| FR-13 | Rate limiting | Chat requests per user per minute and login attempts per IP are limited; 429 with `Retry-After` | Done (usage/spend caps still planned) |
| FR-14 | Retrieval-augmented answers (RAG) | Answers can cite added documents | Done (phase 2, see [RAG_SPEC](RAG_SPEC.md)) |
| FR-15 | Tool/function calling | The model can call allow-listed tools in a bounded loop | Done (phase 4, see [TOOLS_SPEC](TOOLS_SPEC.md)) |
| FR-16 | Stop and regenerate | The user can cancel a reply or regenerate it | Planned |
| FR-17 | Token usage reporting | Input/output tokens are recorded per exchange | Planned |

## 4. Non-functional requirements
| ID | Requirement | How it is met today |
| --- | --- | --- |
| NFR-1 Testability | Core logic is testable without network or keys | Mock provider, injected store, 53 automated tests |
| NFR-2 Portability | Runs locally, in Docker, in CI | Standard library SQLite, Dockerfile, GitHub Actions |
| NFR-3 Security | No secrets in the repository; safe rendering; bounded input | Keys from environment only; UI uses `textContent`; length limits |
| NFR-4 Maintainability | Adding a provider touches one file | Provider interface plus registry (see tech spec) |
| NFR-5 Responsiveness | The first token reaches the user as soon as the model produces it | Streaming end to end; no buffering |
| NFR-6 Reliability | A failure never corrupts stored history | Atomic save of each exchange |
| NFR-7 Observability | Failures are diagnosable | Exceptions logged server-side; richer metrics are planned |

## 5. Key user flows
**Send a message (new conversation).** UI posts `{message, provider}` → server validates, picks the provider, builds the prompt from the trimmed history → streams `meta` (new conversation id), `token` events, then `done` → after a complete reply the user and assistant messages are saved together → UI refreshes the conversation list.

**Continue a conversation.** Same as above with `conversation_id`; the server loads past messages, appends the new one, trims to budget and streams.

**Switch provider mid-conversation.** The dropdown value is sent with each request, so the next turn can use a different model; history is plain text and works across providers.

**Failure.** Provider unreachable or erroring → the stream sends one `error` event ("The model request failed...") → nothing is saved, so the conversation never holds a half exchange; the user can retry.

**Delete.** `DELETE /api/conversations/{id}` removes the conversation and its messages (database cascade).

## 6. API contract
`POST /api/chat` body: `{"message": str (1-8000), "conversation_id": str|null, "provider": str|null, "system": str|null (≤2000)}`

Response `200 text/event-stream`, events in order:
```
event: meta   data: {"conversation_id": "...", "provider": "mock"}
event: token  data: {"text": "..."}          (repeated)
event: done   data: {}                        (or, on failure:)
event: error  data: {"message": "..."}
```
| Status | When |
| --- | --- |
| 200 | Stream started (failures after this point arrive as an `error` event, because the status line is already sent) |
| 400 | Provider unknown or not configured |
| 404 | `conversation_id` does not exist |
| 422 | Validation failed (empty or too long message, wrong types) |

Other endpoints are listed in the README. Conversation list items are `{id, title, created_at}`; the title is the first 60 characters of the first message.

## 7. UI behaviour
Sidebar with "New chat" and the conversation list (open, delete); header with a provider dropdown; scrolling message area; a text box where Enter sends and Shift+Enter inserts a newline. Replies appear as they stream. All message text is inserted as plain text, never as HTML.

## 8. Assumptions and constraints
Authentication is required by default (see AUTH_SPEC); one server process with a local SQLite file; the operator supplies provider keys; token counts are estimated, not exact; model quality, latency and cost belong to the chosen provider.

## 9. Traceability
| Requirement | Code | Test |
| --- | --- | --- |
| FR-1, FR-11 | `providers.py` (`MockProvider`), `main.py` (`events`) | `test_streams_tokens_and_persists` |
| FR-2, FR-3 | `store.py`, `main.py` | `test_second_turn_reuses_conversation`, `test_delete_conversation` |
| FR-4 | `providers.get_provider` | `test_unconfigured_provider_is_400` |
| FR-6 | `context.trim_history` | `test_trim_history_*` (2 tests) |
| FR-7 | `ChatRequest` (Pydantic) | `test_empty_message_rejected` |
| FR-8 | `main.chat` | `test_unknown_conversation_is_404`, `test_provider_failure_persists_nothing` |
| FR-9 | `static/index.html` | `test_index_page_served` (page loads; behaviour checked manually) |
| FR-5, FR-10 | `config.py`, `main.py` | Not covered by automated tests yet |
