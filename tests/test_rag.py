import json

from app import auth
from app.rag import HashEmbedder, KnowledgeBase, chunk_words

DOCS = [
    ("France", "The capital of France is Paris. Paris is known for the Eiffel Tower and the river Seine."),
    ("Plants", "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen inside chloroplasts."),
]


def kb_with_user(db_path):
    kb = KnowledgeBase(db_path, HashEmbedder())
    return kb, auth.create_user(db_path, "alice", "correct horse battery")["id"]


def parse(body):
    out = []
    for block in body.strip().split("\n\n"):
        ev, data = block.split("\n", 1)
        out.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def add_docs(client):
    for title, text in DOCS:
        assert client.post("/api/documents", json={"title": title, "text": text}).status_code == 200


def test_chunking_overlap():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_words(words, 120, 20)
    assert len(chunks) == 3
    assert chunks[1].split()[:20] == chunks[0].split()[-20:]
    assert chunk_words("short text", 120, 20) == ["short text"]
    assert chunk_words("   ", 120, 20) == []


def test_hash_embedder_deterministic_and_normalised():
    a, b = HashEmbedder().embed(["same text here", "same text here"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_search_ranks_relevant_document(db_path):
    kb, uid = kb_with_user(db_path)
    for title, text in DOCS:
        kb.add_document(uid, title, text)
    hits = kb.search(uid, "What is the capital of France?", 3, 0.1)
    assert hits and hits[0]["document"] == "France"


def test_min_score_filters_irrelevant(db_path):
    kb, uid = kb_with_user(db_path)
    kb.add_document(uid, *DOCS[0])
    assert kb.search(uid, "quantum chromodynamics gluons", 3, 0.1) == []


def test_document_api_add_list_delete(client):
    add_docs(client)
    docs = client.get("/api/documents").json()
    assert sorted(d["title"] for d in docs) == ["France", "Plants"] and all(d["chunks"] == 1 for d in docs)
    assert client.delete(f"/api/documents/{docs[0]['id']}").status_code == 200
    assert len(client.get("/api/documents").json()) == 1
    assert client.delete(f"/api/documents/{docs[0]['id']}").status_code == 404


def test_empty_document_rejected(client):
    assert client.post("/api/documents", json={"title": "x", "text": "   "}).status_code == 422


def test_chat_with_docs_emits_sources(client):
    add_docs(client)
    r = client.post("/api/chat", json={"message": "What is the capital of France?", "provider": "mock", "use_docs": True})
    events = parse(r.text)
    assert [e for e, _ in events][:2] == ["meta", "sources"]
    assert events[1][1]["sources"][0]["document"] == "France"


def test_chat_without_docs_has_no_sources(client):
    add_docs(client)
    events = parse(client.post("/api/chat", json={"message": "capital of France", "provider": "mock"}).text)
    assert "sources" not in [e for e, _ in events]
