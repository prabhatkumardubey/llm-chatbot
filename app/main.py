import json
import logging
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import auth
from .config import get_settings
from .context import trim_history
from .providers import ProviderError, available_providers, get_provider
from .rag import KnowledgeBase, build_system, make_embedder
from .ratelimit import RateLimiter
from .store import Store

log = logging.getLogger("llmchat")
STATIC = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    provider: str | None = None
    system: str | None = Field(default=None, max_length=2000)
    use_docs: bool = False


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200000)


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9_.-]{3,32}$")
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


def sse(event: str, data: dict) -> str:
    """One Server-Sent Event: named event, JSON payload, blank line terminator."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(store: Store | None = None, kb: KnowledgeBase | None = None) -> FastAPI:
    settings = get_settings()
    store = store or Store(settings.db_path)
    kb = kb or KnowledgeBase(store.path, make_embedder(settings), settings.rag_chunk_words, settings.rag_overlap_words)
    limiter = RateLimiter()
    if not settings.auth_required:
        auth.ensure_local_user(store.path)
    app = FastAPI(title="llmchat", version="0.3.0")

    def bearer(authorization: str | None) -> str | None:
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return None

    def current_user(authorization: str | None = Header(default=None)) -> dict:
        if not settings.auth_required:
            return auth.LOCAL_USER
        token = bearer(authorization)
        user = auth.user_for_token(store.path, token) if token else None
        if not user:
            raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
        return user

    def throttle(key: str, limit: int) -> None:
        allowed, wait = limiter.check(key, limit)
        if not allowed:
            raise HTTPException(429, "too many requests", headers={"Retry-After": str(wait)})

    def chat_user(user: dict = Depends(current_user)) -> dict:
        throttle(f"chat:{user['id']}", settings.rate_limit_chat_per_min)
        return user

    def auth_throttle(request: Request) -> None:
        throttle(f"auth:{request.client.host if request.client else 'unknown'}", settings.rate_limit_auth_per_min)

    def auth_enabled() -> None:
        if not settings.auth_required:
            raise HTTPException(400, "authentication is disabled")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/providers")
    def providers():
        return {"available": available_providers(), "default": settings.default_provider}

    @app.post("/api/auth/register", status_code=201, dependencies=[Depends(auth_throttle), Depends(auth_enabled)])
    def register(body: Credentials):
        if not settings.allow_registration:
            raise HTTPException(403, "registration is disabled")
        try:
            user = auth.create_user(store.path, body.username, body.password)
        except auth.UserExists:
            raise HTTPException(409, "username already taken")
        return {"token": auth.create_session(store.path, user["id"], settings.session_days), "user": user}

    @app.post("/api/auth/login", dependencies=[Depends(auth_throttle), Depends(auth_enabled)])
    def login(body: LoginIn):
        user = auth.authenticate(store.path, body.username, body.password)
        if not user:  # same message for unknown user and wrong password
            raise HTTPException(401, "invalid username or password", headers={"WWW-Authenticate": "Bearer"})
        return {"token": auth.create_session(store.path, user["id"], settings.session_days), "user": user}

    @app.post("/api/auth/logout")
    def logout(authorization: str | None = Header(default=None), user: dict = Depends(current_user)):
        token = bearer(authorization)
        if settings.auth_required and token:
            auth.revoke_token(store.path, token)
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: dict = Depends(current_user)):
        return user

    @app.get("/api/conversations")
    def list_conversations(user: dict = Depends(current_user)):
        return store.list(user["id"])

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str, user: dict = Depends(current_user)):
        if not store.exists(user["id"], cid):  # 404, not 403: do not reveal that the id exists
            raise HTTPException(404, "conversation not found")
        return {"id": cid, "messages": store.messages(user["id"], cid)}

    @app.delete("/api/conversations/{cid}")
    def delete_conversation(cid: str, user: dict = Depends(current_user)):
        if not store.delete(user["id"], cid):
            raise HTTPException(404, "conversation not found")
        return {"deleted": cid}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, user: dict = Depends(chat_user)):
        uid = user["id"]
        try:
            provider = get_provider(req.provider or settings.default_provider)
        except ProviderError as e:
            raise HTTPException(400, str(e))
        cid = req.conversation_id
        if cid:
            if not await run_in_threadpool(store.exists, uid, cid):
                raise HTTPException(404, "conversation not found")
            past = await run_in_threadpool(store.messages, uid, cid)
        else:
            cid, past = uuid.uuid4().hex, []
        # Nothing is saved yet: the exchange is persisted only after a complete reply (see below).
        history = trim_history([*past, {"role": "user", "content": req.message}], settings.max_context_tokens)
        system = req.system or settings.system_prompt
        sources: list[dict] = []
        if req.use_docs:
            try:
                sources = await run_in_threadpool(
                    kb.search, uid, req.message, settings.rag_top_k, settings.rag_min_score,
                    settings.rag_mode, settings.rag_min_bm25)
            except Exception:
                log.exception("retrieval failed")
                raise HTTPException(502, "document retrieval failed")
            system = build_system(system, sources)

        async def events():
            yield sse("meta", {"conversation_id": cid, "provider": provider.name})
            if req.use_docs:
                yield sse("sources", {"sources": [
                    {"n": s["n"], "document": s["document"], "snippet": s["text"][:200], "score": s["score"], "via": s["via"]}
                    for s in sources]})
            parts: list[str] = []
            try:
                async for tok in provider.stream(history, system, settings.max_output_tokens):
                    parts.append(tok)
                    yield sse("token", {"text": tok})
            except Exception:
                log.exception("provider %s failed", provider.name)
                yield sse("error", {"message": "The model request failed. Check the server logs."})
                return
            await run_in_threadpool(store.save_exchange, uid, cid, req.message[:60], req.message, "".join(parts))
            yield sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/documents")
    def add_document(doc: DocumentIn, user: dict = Depends(current_user)):
        try:
            return kb.add_document(user["id"], doc.title, doc.text)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception:
            log.exception("ingestion failed")
            raise HTTPException(502, "embedding failed; nothing was stored")

    @app.get("/api/documents")
    def list_documents(user: dict = Depends(current_user)):
        return kb.list_documents(user["id"])

    @app.delete("/api/documents/{did}")
    def delete_document(did: str, user: dict = Depends(current_user)):
        if not kb.delete_document(user["id"], did):
            raise HTTPException(404, "document not found")
        return {"deleted": did}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
