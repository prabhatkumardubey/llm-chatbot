import asyncio
import json

from conftest import signup

from app.providers import LLMProvider
from app.rerank import LexicalReranker, LLMReranker, _parse_order


def cand(n, doc, text, score):
    return {"n": n, "document": doc, "text": text, "score": score, "via": ["vector"]}


class Says(LLMProvider):
    name = "says"

    def __init__(self, text):
        self.text = text

    async def stream(self, messages, system, max_tokens):
        yield self.text


class Boom(LLMProvider):
    name = "boom"

    async def stream(self, messages, system, max_tokens):
        raise RuntimeError("down")
        yield ""


CANDS = [cand(1, "A", "alpha text", 0.5), cand(2, "B", "beta text", 0.4), cand(3, "C", "gamma text", 0.3)]


def test_lexical_reranker_prefers_adjacent_phrase():
    cands = [cand(1, "A", "France has many cities and the capital region is large", 0.5),
             cand(2, "B", "The capital of France is Paris.", 0.4)]
    out = asyncio.run(LexicalReranker().rerank("capital of France", cands, 1, None))
    assert len(out) == 1 and out[0]["document"] == "B" and out[0]["n"] == 1 and out[0]["retrieval_score"] == 0.4


def test_llm_reranker_reorders():
    out = asyncio.run(LLMReranker().rerank("q", CANDS, 2, Says("Order: [3, 1, 2]")))
    assert [c["document"] for c in out] == ["C", "A"] and [c["n"] for c in out] == [1, 2]


def test_llm_reranker_fails_open():
    for provider in (Says("no idea"), Boom()):
        out = asyncio.run(LLMReranker().rerank("q", CANDS, 3, provider))
        assert [c["document"] for c in out] == ["A", "B", "C"]


def test_parse_order_is_strict():
    assert _parse_order("[2, 2, 9, 1]", 3) == [1, 0, 2]
    assert _parse_order("hello", 3) == [0, 1, 2] and _parse_order("[1, 2", 3) == [0, 1, 2] and _parse_order("[]", 3) == [0, 1, 2]


def test_rerank_in_chat_flow(monkeypatch, make_client):
    monkeypatch.setenv("RAG_RERANK", "lexical")
    c = make_client()
    signup(c)
    c.post("/api/documents", json={"title": "France", "text": "The capital of France is Paris. Paris has the Eiffel Tower."})
    c.post("/api/documents", json={"title": "Plants", "text": "Photosynthesis converts sunlight and water into glucose."})
    r = c.post("/api/chat", json={"message": "What is the capital of France?", "provider": "mock", "use_docs": True})
    sources = json.loads(r.text.split("\n\n")[1].split("\n", 1)[1].removeprefix("data: "))["sources"]
    assert sources[0]["document"] == "France" and sources[0]["retrieval_score"] is not None
