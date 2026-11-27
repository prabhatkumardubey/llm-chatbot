"""Retrieval evaluation harness. See docs/EVAL_SPEC.md. Run: python -m app.evaluation"""
import argparse
import asyncio
import json
import random
import statistics
import tempfile
from pathlib import Path

from . import auth
from .config import get_settings
from .providers import get_provider
from .rag import KnowledgeBase, make_embedder
from .rerank import make_reranker

DATA = Path(__file__).resolve().parent.parent / "eval" / "data"
TYPES = ("exact", "keyword", "paraphrase", "multi", "unanswerable")
DEFAULT_CONFIGS = [("vector", "vector", "none"), ("bm25", "bm25", "none"),
                   ("hybrid", "hybrid", "none"), ("hybrid+lexical", "hybrid", "lexical")]


def load_dataset(data_dir: Path = DATA) -> tuple[list[dict], list[dict]]:
    return json.loads((data_dir / "corpus.json").read_text()), json.loads((data_dir / "questions.json").read_text())


def first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i for i, d in enumerate(ranked, 1) if d in relevant), None)


def metrics_for(ranked: list[str], relevant: set[str], k: int) -> dict:
    top = ranked[:k]
    r = first_rank(top, relevant)
    return {"hit": float(r is not None), "recall": len(set(top) & relevant) / len(relevant), "rr": 1 / r if r else 0.0}


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def build(corpus: list[dict], embedder, workdir: str) -> tuple[KnowledgeBase, str]:
    db = str(Path(workdir) / "eval.db")
    kb = KnowledgeBase(db, embedder)
    uid = auth.create_user(db, "eval", "evaluation-password")["id"]
    for d in corpus:
        kb.add_document(uid, d["title"], d["text"])
    return kb, uid


def _summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    stat = lambda key: {"mean": statistics.fmean(r[key] for r in ans),
                        "ci": bootstrap_ci([r[key] for r in ans])}
    by_type = {t: statistics.fmean(r["rr"] for r in ans if r["type"] == t) for t in TYPES if t != "unanswerable"}
    return {"n_answerable": len(ans), "hit": stat("hit"), "recall": stat("recall"), "mrr": stat("rr"),
            "mrr_by_type": by_type, "n_unanswerable": len(unans),
            "abstention": statistics.fmean(float(r["abstained"]) for r in unans) if unans else None}


async def evaluate(kb, uid, questions, configs=DEFAULT_CONFIGS, k=3, candidates=10, min_score=0.1, min_bm25=0.3, provider=None) -> dict:
    out = {}
    for name, mode, rr in configs:
        reranker, rows = make_reranker(rr), []
        for q in questions:
            pool = k if rr == "none" else candidates
            found = kb.search(uid, q["question"], pool, min_score, mode, min_bm25)
            sources = await reranker.rerank(q["question"], found, k, provider)
            ranked = list(dict.fromkeys(s["document"] for s in sources))  # documents in rank order
            row = {"id": q["id"], "type": q["type"], "ranked": ranked, "abstained": not sources}
            row.update(metrics_for(ranked, set(q["relevant"]), k) if q["relevant"] else {"hit": 0.0, "recall": 0.0, "rr": 0.0})
            rows.append(row)
        out[name] = {"summary": _summarise(rows), "rows": rows}
    return out


async def sweep(kb, uid, questions, mode: str, param: str, values: list[float], k: int = 3) -> list[tuple[float, float, float]]:
    """Exploratory: answerable hit rate vs unanswerable abstention as one threshold changes."""
    rows = []
    for v in values:
        kw = {"min_bm25": v} if param == "min_bm25" else {"min_score": v}
        s = next(iter((await evaluate(kb, uid, questions, [(mode, mode, "none")], k, **kw)).values()))["summary"]
        rows.append((v, s["hit"]["mean"], s["abstention"]))
    return rows


def format_sweep(title: str, param: str, rows) -> str:
    return "\n".join([title, "", f"| {param} | Hit rate (answerable) | Abstains (unanswerable) |", "|---|---|---|"] +
                     [f"| {v} | {h:.2f} | {a:.0%} |" for v, h, a in rows])


def format_report(results: dict, k: int, embedder: str) -> str:
    f = lambda s: f"{s['mean']:.2f} [{s['ci'][0]:.2f}-{s['ci'][1]:.2f}]"
    n = next(iter(results.values()))["summary"]
    lines = [f"Retrieval evaluation (embedder: {embedder}, k={k}, {n['n_answerable']} answerable and {n['n_unanswerable']} unanswerable questions)", "",
             f"| Config | Hit@{k} | Recall@{k} | MRR@{k} | Abstains on unanswerable |", "|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["summary"]
        lines.append(f"| {name} | {f(s['hit'])} | {f(s['recall'])} | {f(s['mrr'])} | {s['abstention']:.0%} |")
    lines += ["", f"MRR@{k} by question type", "", "| Config | " + " | ".join(t for t in TYPES if t != "unanswerable") + " |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['summary']['mrr_by_type'][t]:.2f}" for t in TYPES if t != "unanswerable") + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality on the bundled question set.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--candidates", type=int, default=10)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--min-bm25", type=float, default=None)
    p.add_argument("--embedder", choices=["hash", "openai"], default=None)
    p.add_argument("--llm-rerank", metavar="PROVIDER", help="also run hybrid+llm using this configured provider")
    p.add_argument("--sweep", action="store_true", help="exploratory threshold sweep for bm25 and vector")
    p.add_argument("--json", metavar="PATH")
    a = p.parse_args(argv)
    s = get_settings()
    if a.embedder:
        s = type(s)(**{**s.__dict__, "rag_embedder": a.embedder})
    embedder = make_embedder(s)
    corpus, questions = load_dataset()
    configs, provider = list(DEFAULT_CONFIGS), None
    if a.llm_rerank:
        provider = get_provider(a.llm_rerank)
        configs.append(("hybrid+llm", "hybrid", "llm"))
    with tempfile.TemporaryDirectory() as tmp:
        kb, uid = build(corpus, embedder, tmp)
        results = asyncio.run(evaluate(kb, uid, questions, configs, a.k, a.candidates,
                                       s.rag_min_score if a.min_score is None else a.min_score,
                                       s.rag_min_bm25 if a.min_bm25 is None else a.min_bm25, provider))
        print(format_report(results, a.k, embedder.name))
        if a.sweep:
            print("\nExploratory threshold sweeps (do not tune defaults on this small set)\n")
            print(format_sweep("BM25 only", "min_bm25", asyncio.run(sweep(kb, uid, questions, "bm25", "min_bm25", [0.3, 0.6, 1.0, 1.5, 2.0, 3.0], a.k))))
            print()
            print(format_sweep("Vector only", "min_score", asyncio.run(sweep(kb, uid, questions, "vector", "min_score", [0.1, 0.2, 0.3, 0.4], a.k))))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
