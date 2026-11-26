"""LLM providers behind small interfaces (Strategy + Adapter). The rest of the app only knows these abstractions."""
import asyncio
import json
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


class ToolUser(ABC):
    """Optional capability (interface segregation): providers that can request tool calls."""

    @abstractmethod
    async def step(self, turns: list[dict], system: str, tools: list[dict], max_tokens: int) -> dict:
        """One non-streaming model turn. Turns are neutral dicts: user, assistant (maybe with tool_calls), tool.
        Returns {"text": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""


def to_anthropic(turns: list[dict]) -> list[dict]:
    """Neutral turns -> Claude messages (tool_use / tool_result content blocks)."""
    out: list[dict] = []
    for t in turns:
        if t["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": t["tool_call_id"], "content": t["content"]}
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) and prev["content"][0].get("type") == "tool_result":
                prev["content"].append(block)  # all results of one round go in a single user message
            else:
                out.append({"role": "user", "content": [block]})
        elif t["role"] == "assistant" and t.get("tool_calls"):
            blocks = [{"type": "text", "text": t["content"]}] if t.get("content") else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in t["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": t["role"], "content": t["content"]})
    return out


def to_openai(turns: list[dict], system: str) -> list[dict]:
    """Neutral turns -> OpenAI-style chat messages (tool_calls and role=tool)."""
    out: list[dict] = [{"role": "system", "content": system}]
    for t in turns:
        if t["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": t["tool_call_id"], "content": t["content"]})
        elif t["role"] == "assistant" and t.get("tool_calls"):
            out.append({"role": "assistant", "content": t.get("content") or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for c in t["tool_calls"]]})
        else:
            out.append({"role": t["role"], "content": t["content"]})
    return out


class MockProvider(LLMProvider, ToolUser):
    """Deterministic fake model: no API key, used for tests, CI and demos."""
    name = "mock"

    async def stream(self, messages, system, max_tokens):
        for word in f"[mock] You said: {messages[-1]['content']}".split(" "):
            await asyncio.sleep(0.01)
            yield word + " "

    async def step(self, turns, system, tools, max_tokens):
        last = turns[-1]
        if last["role"] == "tool":
            return {"text": f"[mock] Tool result: {last['content']}", "tool_calls": []}
        msg, low = last["content"], last["content"].lower()
        call = None
        if low.startswith("calc:"):
            call = ("calculator", {"expression": msg[5:].strip()})
        elif low.startswith("docs:"):
            call = ("search_documents", {"query": msg[5:].strip()})
        elif low.startswith("time"):
            call = ("get_current_time", {})
        if call and any(t["name"] == call[0] for t in tools):
            return {"text": "", "tool_calls": [{"id": f"call_{len(turns)}", "name": call[0], "arguments": call[1]}]}
        return {"text": f"[mock] You said: {msg}", "tool_calls": []}


class AnthropicProvider(LLMProvider, ToolUser):
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

    async def step(self, turns, system, tools, max_tokens):
        resp = await self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=to_anthropic(turns),
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["schema"]} for t in tools])
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input)} for b in resp.content if b.type == "tool_use"]
        return {"text": text, "tool_calls": calls}


class OpenAIProvider(LLMProvider, ToolUser):
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

    async def step(self, turns, system, tools, max_tokens):
        resp = await self.client.chat.completions.create(
            model=self.model, messages=to_openai(turns, system), max_tokens=max_tokens,
            tools=[{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["schema"]}}
                   for t in tools])
        msg, calls = resp.choices[0].message, []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except ValueError:  # the model produced invalid JSON; argument validation will report it
                args = {}
            calls.append({"id": c.id, "name": c.function.name, "arguments": args})
        return {"text": msg.content or "", "tool_calls": calls}


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
