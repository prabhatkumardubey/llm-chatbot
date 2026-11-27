# Evaluation Specification: Measuring Retrieval Quality

> Written before the harness. Section 10 links each requirement to its implementation and tests.

## 1. Purpose
Earlier phases added vector search, keyword search, hybrid fusion and re-ranking, but never measured whether any of it helps. This phase adds a repeatable harness that scores retrieval configurations on a labelled question set, so changes can be compared with numbers and regressions are caught.

## 2. Scope
**Measured:** retrieval only: does the system put the right document in the top results, and does it stay quiet when no document is relevant? **Not measured:** the model's final answer (faithfulness, correctness, citation accuracy). That needs human review or a judge model and is listed as future work. Retrieval quality is a ceiling on answer quality, which is why it is measured first.

## 3. Requirements
| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| E-1 | Labelled dataset | A corpus of documents and questions, each answerable question labelled with its relevant document(s); some questions deliberately unanswerable |
| E-2 | Question types | Exact-term, keyword, paraphrase (little word overlap), multi-document and unanswerable questions, reported separately |
| E-3 | Standard metrics | Hit rate, recall and mean reciprocal rank at k for answerable questions; abstention rate for unanswerable ones |
| E-4 | Config comparison | The same questions run through vector, BM25, hybrid and hybrid plus re-ranking, on the production code path |
| E-5 | Uncertainty | 95% bootstrap confidence intervals, because the dataset is small |
| E-6 | Reproducible | Deterministic with the offline embedder; one command; machine-readable JSON output |
| E-7 | Model-agnostic | Runs with the offline embedder by default and with a real embedder or LLM re-ranker via configuration |
| E-9 | Threshold sweep (exploratory) | For one ranker, show answerable hit rate and unanswerable abstention across several thresholds, to expose the trade-off; never used to change defaults on this dataset |
| E-8 | Dataset checks | Automated tests reject broken labels (unknown document ids, missing types) |

## 4. Dataset design
A **fictional company handbook** (14 short documents written for this project; no real policies or data). Documents deliberately overlap on vocabulary (two expense-related documents, three about sign-in and phones) so retrieval can be confused. 28 questions:
| Type | Count | What it tests |
| --- | --- | --- |
| exact | 2 | Rare literal terms (an error code, an HTTP header) |
| keyword | 13 | Natural questions that share words with the answer |
| paraphrase | 7 | Same meaning, almost no shared words ("new fathers take time off" vs "secondary caregivers leave") |
| multi | 2 | Two documents are both relevant |
| unanswerable | 4 | Nothing relevant exists; the right behaviour is to return nothing |

**Labelling:** document-level, binary relevance, decided by the question author. A retrieved chunk counts as relevant if its document is. Documents are short enough to be one chunk each, so chunking is not exercised here.

## 5. Metrics (k = 3 by default, the production `RAG_TOP_K`)
- **Hit@k:** 1 if any relevant document appears in the top k results.
- **Recall@k:** share of the question's relevant documents found in the top k.
- **MRR@k:** 1 / rank of the first relevant result (0 if none): rewards putting the right document first.
- **Abstention rate:** over unanswerable questions, the share where the system returns no results. Higher is better; the complement is the false-answer rate.
The pipeline evaluated is exactly what the chat feature uses: search, optional re-rank, keep the top k chunks.

## 6. Configurations compared
`vector` (cosine only), `bm25` (keywords only), `hybrid` (rank fusion), `hybrid+lexical` (re-rank with the word-overlap heuristic), and optionally `hybrid+llm` (re-rank with a chat model; costs model calls, so opt-in).

## 7. Statistics
With about 24 answerable questions, one question flips a metric by about 4 points. Each mean carries a 95% percentile **bootstrap confidence interval** (1,000 resamples, fixed seed): overlapping intervals mean the configurations are not distinguishable on this data. Results are never presented as a ranking without them.

## 8. Method risks (state these when presenting results)
- **Author bias:** the same person wrote the system and the questions; real users phrase things differently.
- **Small and synthetic:** 28 questions on 14 short fictional documents say little about a large, messy corpus.
- **Threshold tuning:** abstention depends on the score thresholds. Defaults are reported untuned; tuning them on this question set would overfit it.
- **Offline embedder:** the default embedder matches words, not meaning, so paraphrase questions are expected to be weak. This measures the harness and the baseline; the interesting comparison needs a real embedding model, run with `--embedder openai`.
- **Document-level labels** ignore whether the *right passage* was found inside a long document.

## 9. Usage
`python -m app.evaluation` prints a Markdown report; `--json results.json` saves it; `--sweep` prints the threshold trade-off; `--k`, `--min-score`, `--min-bm25`, `--embedder openai`, `--llm-rerank anthropic` change the setup. Results from the offline embedder are recorded in [EVAL_RESULTS.md](EVAL_RESULTS.md).

## 10. Traceability
| Requirement | Code | Tests |
| --- | --- | --- |
| E-1, E-2, E-8 | `eval/data/*.json`, `evaluation.load_dataset` | `test_dataset_is_well_formed` |
| E-3 | `evaluation.metrics_for`, `evaluate` | `test_metric_functions` |
| E-4 | `evaluation.evaluate`, `DEFAULT_CONFIGS` | `test_harness_runs_all_configs` |
| E-5 | `evaluation.bootstrap_ci` | `test_bootstrap_ci_is_deterministic_and_sane` |
| E-9 | `evaluation.sweep` | `test_sweep_trades_hits_for_abstention` |
| E-6, E-7 | `evaluation.main` | `test_harness_runs_all_configs` (offline embedder); real embedders untested without keys |
