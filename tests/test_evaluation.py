import asyncio

from app import evaluation as ev
from app.rag import HashEmbedder


def harness(tmp_path):
    corpus, questions = ev.load_dataset()
    kb, uid = ev.build(corpus, HashEmbedder(), str(tmp_path))
    return kb, uid, questions


def test_dataset_is_well_formed():
    corpus, qs = ev.load_dataset()
    titles = [d["title"] for d in corpus]
    assert len(titles) == len(set(titles)) and len([q["id"] for q in qs]) == len({q["id"] for q in qs})
    for q in qs:
        assert q["type"] in ev.TYPES and set(q["relevant"]) <= set(titles)
        assert (q["type"] == "unanswerable") == (q["relevant"] == [])
        assert q["type"] != "multi" or len(q["relevant"]) >= 2
    assert {q["type"] for q in qs} == set(ev.TYPES)


def test_metric_functions():
    assert ev.first_rank(["a", "b", "c"], {"c"}) == 3 and ev.first_rank(["a"], {"z"}) is None
    assert ev.metrics_for(["a", "b", "c"], {"b", "z"}, 3) == {"hit": 1.0, "recall": 0.5, "rr": 0.5}
    assert ev.metrics_for(["a", "b", "c"], {"c"}, 2)["hit"] == 0.0  # found, but outside the top k


def test_bootstrap_ci_is_deterministic_and_sane():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    ci = ev.bootstrap_ci(vals)
    assert ci == ev.bootstrap_ci(vals) and 0 <= ci[0] <= sum(vals) / len(vals) <= ci[1] <= 1
    assert ev.bootstrap_ci([1.0] * 5) == (1.0, 1.0)


def test_harness_runs_all_configs(tmp_path):
    kb, uid, qs = harness(tmp_path)
    res = asyncio.run(ev.evaluate(kb, uid, qs))
    assert list(res) == ["vector", "bm25", "hybrid", "hybrid+lexical"]
    for r in res.values():
        assert len(r["rows"]) == len(qs) and r["summary"]["n_answerable"] == 24 and 0 <= r["summary"]["mrr"]["mean"] <= 1
    assert res["bm25"]["summary"]["hit"]["mean"] >= 0.8  # regression guard for the offline baseline, not a quality claim
    assert "| hybrid |" in ev.format_report(res, 3, "hash-256")


def test_sweep_trades_hits_for_abstention(tmp_path):
    kb, uid, qs = harness(tmp_path)
    low, high = asyncio.run(ev.sweep(kb, uid, qs, "vector", "min_score", [0.1, 0.4]))
    assert low[1] > high[1] and high[2] >= low[2]  # a stricter threshold loses hits and abstains more
