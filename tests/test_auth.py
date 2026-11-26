import json

from conftest import PASSWORD, signup

from app import auth
from app.ratelimit import RateLimiter
from app.store import Store


def chat(client, message="hello", **kw):
    return client.post("/api/chat", json={"message": message, "provider": "mock", **kw})


def cid_of(response):
    block = response.text.split("\n\n")[0]
    return json.loads(block.split("\n", 1)[1].removeprefix("data: "))["conversation_id"]


def test_unauthenticated_requests_are_401(make_client):
    c = make_client()
    assert c.get("/api/conversations").status_code == 401
    assert c.get("/api/documents").status_code == 401
    assert c.post("/api/documents", json={"title": "t", "text": "x"}).status_code == 401
    assert chat(c).status_code == 401
    assert c.get("/health").status_code == 200 and c.get("/api/providers").status_code == 200


def test_register_login_me(make_client):
    c = make_client()
    signup(c, "alice")
    assert c.get("/api/auth/me").json()["username"] == "alice"
    assert c.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code == 200
    bad = c.post("/api/auth/login", json={"username": "alice", "password": "wrong password"})
    ghost = c.post("/api/auth/login", json={"username": "nobody", "password": PASSWORD})
    assert bad.status_code == ghost.status_code == 401 and bad.json() == ghost.json()
    assert c.post("/api/auth/register", json={"username": "ALICE", "password": PASSWORD}).status_code == 409
    assert c.post("/api/auth/register", json={"username": "bob", "password": "short"}).status_code == 422
    assert c.post("/api/auth/register", json={"username": "a b", "password": PASSWORD}).status_code == 422


def test_logout_revokes_token(client):
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_expired_session_rejected(db_path):
    Store(db_path)
    user = auth.create_user(db_path, "carol", PASSWORD)
    token = auth.create_session(db_path, user["id"], days=-1)
    assert auth.user_for_token(db_path, token) is None


def test_conversations_are_isolated(client, make_client):
    cid = cid_of(chat(client))
    bob = make_client()
    signup(bob, "bob")
    assert bob.get("/api/conversations").json() == []
    assert bob.get(f"/api/conversations/{cid}").status_code == 404
    assert bob.delete(f"/api/conversations/{cid}").status_code == 404
    assert chat(bob, conversation_id=cid).status_code == 404
    assert len(client.get(f"/api/conversations/{cid}").json()["messages"]) == 2


def test_documents_and_retrieval_are_isolated(client, make_client):
    did = client.post("/api/documents", json={"title": "France", "text": "The capital of France is Paris."}).json()["id"]
    bob = make_client()
    signup(bob, "bob")
    assert bob.get("/api/documents").json() == []
    assert bob.delete(f"/api/documents/{did}").status_code == 404
    r = chat(bob, "capital of France", use_docs=True)
    sources = json.loads(r.text.split("\n\n")[1].split("\n", 1)[1].removeprefix("data: "))["sources"]
    assert sources == []
    assert len(client.get("/api/documents").json()) == 1


def test_password_hashing():
    h1, h2 = auth.hash_password("pw12345678"), auth.hash_password("pw12345678")
    assert h1 != h2 and "pw12345678" not in h1
    assert auth.verify_password("pw12345678", h1) and not auth.verify_password("other", h1)
    assert not auth.verify_password("x", "garbage")


def test_token_bucket():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    assert rl.check("k", 2) == (True, 0) and rl.check("k", 2) == (True, 0)
    allowed, wait = rl.check("k", 2)
    assert not allowed and wait == 30  # 2 per minute refills one token every 30 seconds
    now[0] = 30
    assert rl.check("k", 2)[0]
    assert rl.check("other", 2)[0] and rl.check("k", 0) == (True, 0)


def test_chat_rate_limit(monkeypatch, make_client):
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MIN", "2")
    c = make_client()
    signup(c)
    assert [chat(c).status_code for _ in range(3)] == [200, 200, 429]
    assert "retry-after" in chat(c).headers


def test_login_rate_limit(monkeypatch, make_client):
    monkeypatch.setenv("RATE_LIMIT_AUTH_PER_MIN", "4")
    c = make_client()
    signup(c)  # uses one attempt
    codes = [c.post("/api/auth/login", json={"username": "alice", "password": "wrong"}).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]


def test_auth_can_be_disabled(monkeypatch, make_client):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    c = make_client()
    assert chat(c).status_code == 200
    assert len(c.get("/api/conversations").json()) == 1
    assert c.get("/api/auth/me").json()["username"] == "local"
