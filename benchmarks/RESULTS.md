# HireFlow search benchmark results

- Corpus: 50 resumes in `data/resumes/`, 9 hand-labeled queries
- Pinecone vector search: live
- LLM re-ranking: live (Gemini)
- Total run time: 305.9s

Relevance judgments are hand-curated from job-title matching across the corpus (see `QUERIES` in `benchmarks/benchmark.py`), not inferred from the resumes' own text.

## BM25-only vs. hybrid search (RRF-fused BM25 + Pinecone vector)

| metric | BM25-only | Hybrid | Lift |
|---|---|---|---|
| mrr | 0.8333 | 0.8889 | +0.0556 |
| map | 0.3672 | 0.3204 | -0.0469 |
| precision@3 | 0.6296 | 0.5926 | -0.0370 |
| precision@5 | 0.4889 | 0.5111 | +0.0222 |
| precision@10 | 0.4111 | 0.3889 | -0.0222 |
| recall@3 | 0.2417 | 0.2357 | -0.0060 |
| recall@5 | 0.2933 | 0.3206 | +0.0273 |
| recall@10 | 0.4797 | 0.4342 | -0.0455 |
| ndcg@3 | 0.6521 | 0.6677 | +0.0157 |
| ndcg@5 | 0.5911 | 0.6315 | +0.0404 |
| ndcg@10 | 0.6132 | 0.6130 | -0.0002 |

## Re-ranker (averaged across all queries)

| metric | value |
|---|---|
| avg fit_score_correlation | 0.2104 |
| avg ndcg_uplift | -0.0101 |

| query | fit_score_correlation | ndcg_uplift |
|---|---|---|
| Senior Accountant w/ QuickBooks & tax compliance | 0.4082 | +0.0000 |
| Staff Accountant w/ general ledger & month-end close | 0.0000 | -0.1252 |
| Controller w/ financial reporting & leadership | -0.1863 | -0.1499 |
| Tax Manager w/ corporate tax & audit defense | 0.0000 | +0.0000 |
| Senior Financial Analyst w/ forecasting & variance analysis | 0.6667 | +0.1931 |
| Financial Analyst w/ budget planning | 0.3536 | +0.0417 |
| VP of Finance w/ strategic leadership | 0.0589 | -0.0274 |
| Entry-level / Junior Accountant | 0.5923 | -0.0235 |
| Accounting Assistant w/ AP & reconciliation | 0.0000 | +0.0000 |

## Methodology notes

- `n=9` queries is small; per-query numbers are noisy (see the per-query table above) — the averaged/lift numbers are the meaningful ones, not any single query's score.
- Relevance grades are 0/1/2 (not relevant / adjacent role / exact role match), assigned by hand from the corpus's actual job-title distribution.
- Re-run with `python benchmarks/benchmark.py` (requires `PINECONE_API_KEY` and `GOOGLE_API_KEY` in `.env` for the full pipeline).
