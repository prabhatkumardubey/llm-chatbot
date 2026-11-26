"""Settings come from environment variables and are read on every call, so tests can override them."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str
    default_provider: str
    system_prompt: str
    max_context_tokens: int
    max_output_tokens: int
    anthropic_api_key: str | None
    anthropic_model: str
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None
    ollama_model: str | None
    ollama_base_url: str
    rag_embedder: str
    rag_top_k: int
    rag_chunk_words: int
    rag_overlap_words: int
    rag_min_score: float
    openai_embed_model: str
    rag_mode: str
    rag_min_bm25: float
    auth_required: bool
    allow_registration: bool
    session_days: int
    rate_limit_chat_per_min: int
    rate_limit_auth_per_min: int
    rag_rerank: str
    rag_candidates: int
    tool_max_steps: int
    tool_timeout_s: float


def get_settings() -> Settings:
    e = os.getenv
    return Settings(
        db_path=e("CHAT_DB_PATH", "chat.db"),
        default_provider=e("DEFAULT_PROVIDER", "mock"),
        system_prompt=e("SYSTEM_PROMPT", "You are a helpful, concise assistant."),
        max_context_tokens=int(e("MAX_CONTEXT_TOKENS", "6000")),
        max_output_tokens=int(e("MAX_OUTPUT_TOKENS", "1024")),
        anthropic_api_key=e("ANTHROPIC_API_KEY"),
        anthropic_model=e("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        openai_api_key=e("OPENAI_API_KEY"),
        openai_model=e("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=e("OPENAI_BASE_URL"),
        ollama_model=e("OLLAMA_MODEL"),
        ollama_base_url=e("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        rag_embedder=e("RAG_EMBEDDER", "hash"),
        rag_top_k=int(e("RAG_TOP_K", "3")),
        rag_chunk_words=int(e("RAG_CHUNK_WORDS", "120")),
        rag_overlap_words=int(e("RAG_OVERLAP_WORDS", "20")),
        rag_min_score=float(e("RAG_MIN_SCORE", "0.1")),
        openai_embed_model=e("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        rag_mode=e("RAG_MODE", "hybrid"),
        rag_min_bm25=float(e("RAG_MIN_BM25", "0.3")),
        auth_required=e("AUTH_REQUIRED", "true").lower() != "false",
        allow_registration=e("ALLOW_REGISTRATION", "true").lower() != "false",
        session_days=int(e("SESSION_DAYS", "7")),
        rate_limit_chat_per_min=int(e("RATE_LIMIT_CHAT_PER_MIN", "20")),
        rate_limit_auth_per_min=int(e("RATE_LIMIT_AUTH_PER_MIN", "10")),
        rag_rerank=e("RAG_RERANK", "none"),
        rag_candidates=int(e("RAG_CANDIDATES", "20")),
        tool_max_steps=int(e("TOOL_MAX_STEPS", "4")),
        tool_timeout_s=float(e("TOOL_TIMEOUT_SECONDS", "5")),
    )
