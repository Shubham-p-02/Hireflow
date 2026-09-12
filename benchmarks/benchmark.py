"""Labeled multi-query benchmark for HireFlow's search & re-ranking pipeline.

Unlike core/evaluator.py's evaluate_ranking_quality/evaluate_reranker_quality
(which infer relevance from a skill-overlap heuristic on the fly — useful for
ad hoc checks with no labeled data), this benchmark uses relevance judgments
hand-curated from the actual job-title distribution across the 50 resumes in
data/resumes/ (see QUERIES below) — a real labeled ground truth rather than a
heuristic derived from the same text being searched.

It runs every query through three configurations and reports the lift between
them:
  1. BM25 only
  2. Hybrid (BM25 + Pinecone vector search, RRF-fused)
  3. Hybrid + LLM re-ranking

Requires PINECONE_API_KEY (for the hybrid step) and GOOGLE_API_KEY (for
re-ranking) in .env; degrades gracefully to BM25-only / rule-based
re-ranking if either is unset, though the results won't reflect the full
pipeline in that case.

Run: python benchmarks/benchmark.py
Writes a Markdown report to benchmarks/RESULTS.md.
"""
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion import load_resumes
from core.hybrid_indexer import HybridIndexer
from core.re_ranker import ReRanker
from core.ir_metrics import QueryJudgment, RankingMetrics, evaluate_ranking
from core.rerank_eval import evaluate_reranker
from utils.schemas import SearchQuery, Resume
from utils.utils import load_pdf

# Gemini's free tier caps at 15 requests/minute; re_rank_candidates() fires
# one LLM call per candidate back-to-back with no pacing, which blows past
# that within a couple of queries. Space individual calls out instead of
# just pacing between queries.
RERANK_CALL_DELAY_SECONDS = 4.5

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "resumes")
RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"

# ---------------------------------------------------------------------------
# Labeled benchmark queries.
#
# `relevant_titles` maps a job title (the second non-empty line of these
# generated resumes — see `_extract_title`) to a relevance grade: 2 = strong
# match for the query's target role, 1 = an adjacent/partial match, absent =
# not relevant. These grades were assigned by hand after reviewing the
# actual title distribution across all 50 resumes — this is a real (if
# small) labeled ground truth, not derived automatically from the resumes'
# own text.
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label": "Senior Accountant w/ QuickBooks & tax compliance",
        "query": "Senior Accountant with QuickBooks and tax compliance experience",
        "job_title": "Senior Accountant",
        "required_skills": ["QuickBooks", "Tax Compliance", "Excel", "Audit"],
        "relevant_titles": {"Senior Accountant": 2, "Accountant": 1},
    },
    {
        "label": "Staff Accountant w/ general ledger & month-end close",
        "query": "Staff Accountant with general ledger reconciliation and month-end close experience",
        "job_title": "Staff Accountant",
        "required_skills": ["General Ledger", "Month-End Closing", "Bank Reconciliation"],
        "relevant_titles": {"Staff Accountant": 2, "Accounting Assistant": 1},
    },
    {
        "label": "Controller w/ financial reporting & leadership",
        "query": "Controller overseeing financial reporting, budgeting, and team leadership",
        "job_title": "Controller",
        "required_skills": ["Financial Reporting", "Budgeting", "Team Leadership"],
        "relevant_titles": {"Controller": 2, "VP": 1, "Senior Accountant": 1},
    },
    {
        "label": "Tax Manager w/ corporate tax & audit defense",
        "query": "Tax Manager with corporate tax compliance and audit defense experience",
        "job_title": "Tax Manager",
        "required_skills": ["Corporate Tax", "Audit Defense", "Tax Strategy"],
        "relevant_titles": {"Tax Manager": 2, "Senior Accountant": 1},
    },
    {
        "label": "Senior Financial Analyst w/ forecasting & variance analysis",
        "query": "Senior Financial Analyst with forecasting, variance analysis, and financial modeling",
        "job_title": "Senior Financial Analyst",
        "required_skills": ["Forecasting", "Variance Analysis", "Financial Modeling"],
        "relevant_titles": {"Senior Financial Analyst": 2, "Financial Analyst": 1},
    },
    {
        "label": "Financial Analyst w/ budget planning",
        "query": "Financial Analyst with budget planning and financial modeling experience",
        "job_title": "Financial Analyst",
        "required_skills": ["Budget Planning", "Financial Modeling"],
        "relevant_titles": {"Financial Analyst": 2, "Senior Financial Analyst": 1},
    },
    {
        "label": "VP of Finance w/ strategic leadership",
        "query": "VP of Finance with strategic planning and executive financial leadership experience",
        "job_title": "VP",
        "required_skills": ["Strategic Planning", "Financial Management"],
        "relevant_titles": {"VP": 2, "Controller": 1},
    },
    {
        "label": "Entry-level / Junior Accountant",
        "query": "Entry-level accounting graduate seeking a junior accountant role",
        "job_title": "Junior Accountant",
        "required_skills": ["Accounting", "Excel"],
        "relevant_titles": {"Recent Graduate": 2, "Junior Accountant": 2, "Accounting Assistant": 1},
    },
    {
        "label": "Accounting Assistant w/ AP & reconciliation",
        "query": "Accounting Assistant with accounts payable and bank reconciliation experience",
        "job_title": "Accounting Assistant",
        "required_skills": ["Accounts Payable", "Bank Reconciliation"],
        "relevant_titles": {"Accounting Assistant": 2, "Staff Accountant": 1},
    },
]


def _extract_title(resume_path: str) -> str:
    """Second non-empty line of these generated resumes is the job title
    (e.g. name line, then 'VP' / 'Staff Accountant' / ...). Specific to this
    dataset's fixed template — not a general resume parser."""
    text = load_pdf(resume_path) or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[1] if len(lines) > 1 else ""


def build_relevance(titles: Dict[str, str], relevant_titles: Dict[str, float]) -> Dict[str, float]:
    return {cid: relevant_titles.get(title, 0.0) for cid, title in titles.items()}


def bm25_only_ranking(indexer: HybridIndexer, query: str, top_k: int) -> List[str]:
    """Rank candidates using BM25 alone (bypassing Pinecone) — the ablation baseline."""
    query_tokens = query.lower().split()
    bm25_scores = [float(s) for s in indexer.bm25_resumes.get_scores(query_tokens)]
    ranked = indexer.combine_results(bm25_scores, [], top_k)
    return [c["candidate_id"] for c in ranked]


def paced_rerank(reranker: ReRanker, candidates: List[Dict], jd: SearchQuery):
    """Same conversion/sort logic as ReRanker.re_rank_candidates(), but with
    a delay between each LLM call so a multi-query benchmark run doesn't
    blow through Gemini's free-tier rate limit partway through."""
    evaluations = []
    for i, candidate in enumerate(candidates):
        resume = Resume(
            candidate_id=candidate.get("candidate_id", "unknown"),
            name=candidate.get("name", "Unknown"),
            text=candidate.get("page_content") or candidate.get("text") or "",
            skills=candidate.get("skills", []),
            experience=candidate.get("experience", None),
        )
        evaluations.append(reranker.evaluate_candidate(resume, jd))
        if i < len(candidates) - 1:
            time.sleep(RERANK_CALL_DELAY_SECONDS)
    evaluations.sort(key=lambda x: x.fit_score, reverse=True)
    return evaluations


def _lift(hybrid: RankingMetrics, bm25: RankingMetrics) -> Dict[str, float]:
    h, b = hybrid.to_dict(), bm25.to_dict()
    return {k: h[k] - b[k] for k in h if k != "num_queries"}


def main():
    t0 = time.time()
    resumes = load_resumes(DATA_DIR, parse=False)
    print(f"Loaded {len(resumes)} resumes")

    titles = {doc.metadata["candidate_id"]: _extract_title(doc.metadata["source"]) for doc in resumes}

    indexer = HybridIndexer()
    indexer.index_resumes(resumes)
    pinecone_ready = indexer.vector_store.is_ready()
    print(f"Indexed. Pinecone ready: {pinecone_ready}")

    reranker = ReRanker()
    llm_available = reranker.is_available()
    print(f"Re-ranker LLM available: {llm_available}")

    bm25_judgments, hybrid_judgments = [], []
    rerank_results = []

    for q in QUERIES:
        relevance = build_relevance(titles, q["relevant_titles"])
        n_relevant = sum(1 for grade in relevance.values() if grade > 0)

        bm25_ids = bm25_only_ranking(indexer, q["query"], top_k=10)
        bm25_judgments.append(QueryJudgment(query=q["query"], ranked_ids=bm25_ids, relevance=relevance))

        hybrid_candidates = indexer.search_resumes(q["query"], top_k=10)
        hybrid_ids = [c["candidate_id"] for c in hybrid_candidates]
        hybrid_judgments.append(QueryJudgment(query=q["query"], ranked_ids=hybrid_ids, relevance=relevance))

        top5 = hybrid_candidates[:5]
        jd = SearchQuery(title=q["job_title"], text=q["query"], required_skills=q["required_skills"])
        evaluations = paced_rerank(reranker, top5, jd)
        fit_scores_by_id = {e.candidate_id: float(e.fit_score) for e in evaluations}
        pre_ids = [c["candidate_id"] for c in top5]
        post_ids = [e.candidate_id for e in evaluations]
        rr = evaluate_reranker(pre_ids, post_ids, fit_scores_by_id, relevance, k=5)
        rerank_results.append(rr)

        print(f"  [{q['label']}] {n_relevant} labeled-relevant candidates | "
              f"bm25 top1={bm25_ids[0] if bm25_ids else None} | hybrid top1={hybrid_ids[0] if hybrid_ids else None}")
        time.sleep(RERANK_CALL_DELAY_SECONDS)  # extra headroom between queries too

    bm25_metrics = evaluate_ranking(bm25_judgments, k_values=(3, 5, 10))
    hybrid_metrics = evaluate_ranking(hybrid_judgments, k_values=(3, 5, 10))
    lift = _lift(hybrid_metrics, bm25_metrics)
    avg_uplift = sum(r.ndcg_uplift for r in rerank_results) / len(rerank_results)
    avg_corr = sum(r.fit_score_correlation for r in rerank_results) / len(rerank_results)

    print("\n" + "=" * 70)
    print(f"BM25-only:  {bm25_metrics.to_dict()}")
    print(f"Hybrid:     {hybrid_metrics.to_dict()}")
    print(f"Lift:       {lift}")
    print(f"Re-ranker avg fit_score_correlation: {avg_corr:.4f}, avg ndcg_uplift: {avg_uplift:+.4f}")
    print(f"Total time: {time.time()-t0:.1f}s")

    _write_report(
        resumes, pinecone_ready, llm_available, bm25_metrics, hybrid_metrics, lift,
        avg_corr, avg_uplift, rerank_results, time.time() - t0,
    )
    print(f"\nReport written to {RESULTS_PATH}")


def _write_report(resumes, pinecone_ready, llm_available, bm25_metrics, hybrid_metrics,
                   lift, avg_corr, avg_uplift, rerank_results, elapsed):
    lines = [
        "# HireFlow search benchmark results",
        "",
        f"- Corpus: {len(resumes)} resumes in `data/resumes/`, {len(QUERIES)} hand-labeled queries",
        f"- Pinecone vector search: {'live' if pinecone_ready else 'unavailable — falling back to BM25-only'}",
        f"- LLM re-ranking: {'live (Gemini)' if llm_available else 'unavailable — rule-based fallback'}",
        f"- Total run time: {elapsed:.1f}s",
        "",
        "Relevance judgments are hand-curated from job-title matching across the corpus "
        "(see `QUERIES` in `benchmarks/benchmark.py`), not inferred from the resumes' own text.",
        "",
        "## BM25-only vs. hybrid search (RRF-fused BM25 + Pinecone vector)",
        "",
        "| metric | BM25-only | Hybrid | Lift |",
        "|---|---|---|---|",
    ]
    b, h = bm25_metrics.to_dict(), hybrid_metrics.to_dict()
    for k in b:
        if k == "num_queries":
            continue
        lines.append(f"| {k} | {b[k]:.4f} | {h[k]:.4f} | {lift[k]:+.4f} |")

    lines += [
        "",
        "## Re-ranker (averaged across all queries)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| avg fit_score_correlation | {avg_corr:.4f} |",
        f"| avg ndcg_uplift | {avg_uplift:+.4f} |",
        "",
        "| query | fit_score_correlation | ndcg_uplift |",
        "|---|---|---|",
    ]
    for q, rr in zip(QUERIES, rerank_results):
        lines.append(f"| {q['label']} | {rr.fit_score_correlation:.4f} | {rr.ndcg_uplift:+.4f} |")

    lines += [
        "",
        "## Methodology notes",
        "",
        "- `n=9` queries is small; per-query numbers are noisy (see the per-query table above) — "
        "the averaged/lift numbers are the meaningful ones, not any single query's score.",
        "- Relevance grades are 0/1/2 (not relevant / adjacent role / exact role match), assigned "
        "by hand from the corpus's actual job-title distribution.",
        "- Re-run with `python benchmarks/benchmark.py` (requires `PINECONE_API_KEY` and "
        "`GOOGLE_API_KEY` in `.env` for the full pipeline).",
    ]
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
