import json

from app.context import trim_history
from app.providers import LLMProvider


def parse(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        ev, data = block.split("\n", 1)
        out.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def chat(client, message, **kw):
    r = client.post("/api/chat", json={"message": message, "provider": "mock", **kw})
    assert r.status_code == 200
    return parse(r.text)


def test_streams_tokens_and_persists(client):
    events = chat(client, "hello")
    assert events[0][0] == "meta" and events[-1][0] == "done"
    assert "".join(d["text"] for e, d in events if e == "token") == "[mock] You said: hello "
    cid = events[0][1]["conversation_id"]
    msgs = client.get(f"/api/conversations/{cid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_second_turn_reuses_conversation(client):
    cid = chat(client, "one")[0][1]["conversation_id"]
    chat(client, "two", conversation_id=cid)
    assert len(client.get(f"/api/conversations/{cid}").json()["messages"]) == 4
    assert len(client.get("/api/conversations").json()) == 1


def test_unconfigured_provider_is_400(client):
    r = client.post("/api/chat", json={"message": "hi", "provider": "anthropic"})
    assert r.status_code == 400


def test_unknown_conversation_is_404(client):
    r = client.post("/api/chat", json={"message": "hi", "provider": "mock", "conversation_id": "nope"})
    assert r.status_code == 404


def test_empty_message_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_delete_conversation(client):
    cid = chat(client, "bye")[0][1]["conversation_id"]
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_trim_history_keeps_recent_turns_starting_with_user():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 40} for i in range(9)]
    assert trim_history(msgs, 35) == msgs[-3:]


def test_trim_history_always_keeps_latest_message():
    msgs = [{"role": "user", "content": "y" * 4000}]
    assert trim_history(msgs, 10) == msgs


def test_index_page_served(client):
    assert client.get("/").status_code == 200


def test_provider_failure_persists_nothing(client, monkeypatch):
    class Boom(LLMProvider):
        name = "boom"

        async def stream(self, messages, system, max_tokens):
            raise RuntimeError("provider down")
            yield ""

    monkeypatch.setattr("app.main.get_provider", lambda name: Boom())
    events = parse(client.post("/api/chat", json={"message": "hi"}).text)
    assert events[-1][0] == "error"
    assert client.get("/api/conversations").json() == []
