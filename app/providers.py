"""LLM providers behind one interface (Strategy pattern). The rest of the app only knows LLMProvider."""
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from .config import get_settings


class ProviderError(ValueError):
    """Raised when a provider is unknown or not configured."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def stream(self, messages: list[dict], system: str, max_tokens: int) -> AsyncIterator[str]:
        """Yield the reply as text chunks. `messages` alternates user/assistant and starts with user."""


class MockProvider(LLMProvider):
    """Deterministic fake model: no API key, used for tests, CI and demos."""
    name = "mock"

    async def stream(self, messages, system, max_tokens):
        for word in f"[mock] You said: {messages[-1]['content']}".split(" "):
            await asyncio.sleep(0.01)
            yield word + " "


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str):
        from anthropic import AsyncAnthropic  # imported lazily so the SDK is optional

        self.client, self.model = AsyncAnthropic(api_key=api_key), model

    async def stream(self, messages, system, max_tokens):
        async with self.client.messages.stream(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        ) as s:
            async for text in s.text_stream:
                yield text


class OpenAIProvider(LLMProvider):
    """Also serves any OpenAI-compatible endpoint (Ollama, vLLM, Groq, ...) via base_url."""

    def __init__(self, name: str, model: str, api_key: str, base_url: str | None = None):
        from openai import AsyncOpenAI

        self.name, self.model = name, model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream(self, messages, system, max_tokens):
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


_cache: dict[str, LLMProvider] = {}


def available_providers() -> list[str]:
    s = get_settings()
    names = ["mock"]
    if s.anthropic_api_key:
        names.append("anthropic")
    if s.openai_api_key:
        names.append("openai")
    if s.ollama_model:
        names.append("ollama")
    return names


def get_provider(name: str) -> LLMProvider:
    if name not in available_providers():
        raise ProviderError(f"provider '{name}' is not configured; available: {available_providers()}")
    if name not in _cache:
        s = get_settings()
        if name == "mock":
            _cache[name] = MockProvider()
        elif name == "anthropic":
            _cache[name] = AnthropicProvider(s.anthropic_model, s.anthropic_api_key)
        elif name == "openai":
            _cache[name] = OpenAIProvider("openai", s.openai_model, s.openai_api_key, s.openai_base_url)
        else:
            _cache[name] = OpenAIProvider("ollama", s.ollama_model, "ollama", s.ollama_base_url)
    return _cache[name]
