import asyncio
import json

import pytest
from conftest import signup

from app.providers import LLMProvider, ToolUser, to_anthropic, to_openai
from app.tools import default_tools, execute, safe_eval


def parse(body):
    out = []
    for block in body.strip().split("\n\n"):
        ev, data = block.split("\n", 1)
        out.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def chat(client, message, **kw):
    r = client.post("/api/chat", json={"message": message, "provider": "mock", "use_tools": True, **kw})
    assert r.status_code == 200, r.text
    return parse(r.text)


def test_calculator_accepts_arithmetic():
    assert safe_eval("2+3*4") == 14 and safe_eval("(1+2)**3") == 27 and safe_eval("-4 + 10 / 4") == -1.5


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "open('f')", "1 if 1 else 2", "a+1", "9**9**9", "2 +", "1" * 300])
def test_calculator_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        safe_eval(bad)


def test_unknown_tool_and_bad_arguments():
    tools = default_tools()
    run = lambda name, args: asyncio.run(execute(tools, name, args, None, 5))
    assert run("rm_rf", {}).startswith("error: unknown tool")
    assert "missing required argument" in run("calculator", {})
    assert "must be a string" in run("calculator", {"expression": 5})
    assert "unknown argument" in run("calculator", {"expression": "1", "x": 1})
    assert run("get_current_time", {"timezone": "UTC"}).endswith("+00:00")


def test_tool_loop_through_the_api(client):
    events = chat(client, "calc: 2+3*4")
    names = [e for e, _ in events]
    assert names[:3] == ["meta", "tool_call", "tool_result"] and names[-1] == "done"
    assert events[1][1]["name"] == "calculator" and events[2][1]["output"] == "14"
    answer = "".join(d["text"] for e, d in events if e == "token")
    assert answer == "[mock] Tool result: 14"
    cid = events[0][1]["conversation_id"]
    msgs = client.get(f"/api/conversations/{cid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["content"] == answer  # only the final answer is saved


def test_tool_error_is_returned_to_the_model(client):
    events = chat(client, "calc: 1/0")
    assert [d["output"] for e, d in events if e == "tool_result"][0].startswith("error")
    assert events[-1][0] == "done"


def test_search_documents_tool_is_user_scoped(client, make_client):
    client.post("/api/documents", json={"title": "France", "text": "The capital of France is Paris."})
    alice = [d["output"] for e, d in chat(client, "docs: capital of France") if e == "tool_result"][0]
    assert "Paris" in alice
    bob = make_client()
    signup(bob, "bob")
    assert [d["output"] for e, d in chat(bob, "docs: capital of France") if e == "tool_result"][0] == "no matching documents"


class Looper(LLMProvider, ToolUser):
    name = "looper"

    async def stream(self, messages, system, max_tokens):
        yield ""

    async def step(self, turns, system, tools, max_tokens):
        return {"text": "", "tool_calls": [{"id": f"c{len(turns)}", "name": "calculator", "arguments": {"expression": "1+1"}}]}


def test_tool_loop_stops_at_step_limit(monkeypatch, make_client):
    monkeypatch.setenv("TOOL_MAX_STEPS", "2")
    monkeypatch.setattr("app.main.get_provider", lambda name: Looper())
    c = make_client()
    signup(c)
    events = chat(c, "go")
    assert [e for e, _ in events].count("tool_call") == 2
    assert "limit" in "".join(d["text"] for e, d in events if e == "token")


class PlainProvider(LLMProvider):
    name = "plain"

    async def stream(self, messages, system, max_tokens):
        yield "x"


def test_provider_without_tools_is_400(monkeypatch, client):
    monkeypatch.setattr("app.main.get_provider", lambda name: PlainProvider())
    r = client.post("/api/chat", json={"message": "hi", "use_tools": True})
    assert r.status_code == 400


def test_message_conversion_for_vendors():
    turns = [
        {"role": "user", "content": "calc"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "calculator", "arguments": {"expression": "1+1"}},
            {"id": "c2", "name": "calculator", "arguments": {"expression": "2+2"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": "2"},
        {"role": "tool", "tool_call_id": "c2", "name": "calculator", "content": "4"},
    ]
    a = to_anthropic(turns)
    assert a[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "1+1"}}
    assert len(a) == 3 and [b["tool_use_id"] for b in a[2]["content"]] == ["c1", "c2"]  # one user message holds all results
    o = to_openai(turns, "sys")
    assert o[0] == {"role": "system", "content": "sys"} and o[2]["content"] is None
    assert o[2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    assert o[3] == {"role": "tool", "tool_call_id": "c1", "content": "2"}
