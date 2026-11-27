# Evaluation Results (offline embedder)

Produced by `python -m app.evaluation --sweep` using the offline `hash-256` embedder, k = 3, default thresholds (`RAG_MIN_SCORE=0.1`, `RAG_MIN_BM25=0.3`), 24 answerable and 4 unanswerable questions. Deterministic: rerunning gives identical numbers. Read [EVAL_SPEC.md](EVAL_SPEC.md) first for the dataset, metrics and method risks. Brackets are 95% bootstrap confidence intervals.

## Main comparison
| Config | Hit@3 | Recall@3 | MRR@3 | Abstains on unanswerable |
|---|---|---|---|---|
| vector | 0.62 [0.42-0.79] | 0.60 [0.42-0.79] | 0.58 [0.40-0.77] | 25% |
| bm25 | 0.88 [0.71-1.00] | 0.85 [0.71-0.98] | 0.85 [0.69-0.97] | 25% |
| hybrid | 0.88 [0.71-1.00] | 0.85 [0.71-0.98] | 0.76 [0.60-0.90] | 0% |
| hybrid+lexical | 0.88 [0.71-1.00] | 0.85 [0.71-0.98] | 0.81 [0.65-0.94] | 0% |

## MRR@3 by question type
| Config | exact | keyword | paraphrase | multi |
|---|---|---|---|---|
| vector | 0.75 | 0.73 | 0.14 | 1.00 |
| bm25 | 1.00 | 0.92 | 0.62 | 1.00 |
| hybrid | 0.75 | 0.85 | 0.55 | 1.00 |
| hybrid+lexical | 1.00 | 0.88 | 0.55 | 1.00 |

## Exploratory threshold sweeps (not used to tune defaults)
| min_bm25 | Hit rate (answerable) | Abstains (unanswerable) |
|---|---|---|
| 0.3 | 0.88 | 25% |
| 1.0 | 0.88 | 25% |
| 2.0 | 0.83 | 25% |
| 3.0 | 0.79 | 100% |

| min_score (vector) | Hit rate | Abstains |
|---|---|---|
| 0.1 | 0.62 | 25% |
| 0.2 | 0.46 | 100% |
| 0.4 | 0.08 | 100% |

## What the numbers do and do not say
1. **Keyword search was the strongest configuration here, and hybrid did not beat it** (MRR 0.76 vs 0.85). The intervals overlap, so with 24 questions this is not a reliable difference. A plausible cause: the offline embedder only matches hashed words, so it is a noisy second signal and fusing it dilutes BM25. **This says nothing about hybrid search with a real embedding model**, which has not been tested and is the comparison that matters (`--embedder openai`).
2. **Vector search with the offline embedder is weak on every type**, and near zero on paraphrases (0.14), exactly as expected for a word-matching embedder.
3. **The "paraphrase" category is not a pure test of meaning.** BM25 scored 0.62 on it because several paraphrased questions still share words with the answer ("customer", "phone", "minute"). That is a flaw in my dataset, to be fixed by writing harder paraphrases.
4. **Abstention is poor at the default thresholds** (0 to 25%). The unanswerable questions share ordinary words with documents (for example "company"), and a single common-word match clears the BM25 threshold. The sweep shows a trade-off, not a free fix: BM25 at 3.0 abstained on all 4 but lost hits (0.88 to 0.79); vector at 0.2 abstained on all 4 but lost nearly half its hits. With only 4 unanswerable questions this is anecdotal, and absolute BM25 scores depend on corpus size, so a fixed threshold will not transfer. A relative or calibrated threshold is the better direction. **The defaults were deliberately not changed**, to avoid overfitting this question set.
5. **Lexical re-ranking** raised hybrid MRR from 0.76 to 0.81, within the noise.

## Next steps
Run with a real embedder and an LLM re-ranker; add more and harder questions (including paraphrases with no shared words) and a corpus with multi-chunk documents; use a paired bootstrap to test differences between configurations; evaluate answers (faithfulness and citation accuracy), not only retrieval.
