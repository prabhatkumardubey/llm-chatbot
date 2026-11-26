from app import auth
from app.rag import HashEmbedder, KnowledgeBase, _bm25, rrf


def setup(db_path, **kw):
    kb = KnowledgeBase(db_path, HashEmbedder(), **kw)
    return kb, auth.create_user(db_path, "alice", "correct horse battery")["id"]


def test_bm25_prefers_rare_term_match():
    scores = _bm25("zyxwv", ["alpha beta", "alpha gamma delta", "unique zyxwv"])
    assert scores[2] > 0 and scores[0] == scores[1] == 0


def test_rrf_rewards_agreement():
    assert rrf([[0, 1, 2], [2, 0, 1]]) == [0, 2, 1]


def test_hybrid_finds_what_vector_search_misses(db_path):
    kb, uid = setup(db_path, chunk_size=500)
    kb.add_document(uid, "Long", " ".join(f"f{i}" for i in range(300)) + " zyxwv123")
    kb.add_document(uid, "Other", "completely different short text about gardening")
    assert kb.search(uid, "zyxwv123", 3, 0.2, mode="vector") == []  # the rare word is diluted in a long chunk
    hits = kb.search(uid, "zyxwv123", 3, 0.2, mode="hybrid", min_bm25=0.1)
    assert hits[0]["document"] == "Long" and hits[0]["via"] == ["bm25"]


def test_vector_mode_matches_phase2(db_path):
    kb, uid = setup(db_path)
    kb.add_document(uid, "France", "The capital of France is Paris. Paris is known for the Eiffel Tower.")
    kb.add_document(uid, "Plants", "Photosynthesis converts sunlight, water and carbon dioxide into glucose.")
    hits = kb.search(uid, "What is the capital of France?", 3, 0.1, mode="vector")
    assert hits[0]["document"] == "France" and hits[0]["via"] == ["vector"]
