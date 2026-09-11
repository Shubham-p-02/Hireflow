"""Unit tests for the IR-metrics evaluation methods on core/evaluator.py's
RAGEvaluator — no LLM or Pinecone calls required. GOOGLE_API_KEY is patched
to empty so evaluate_reranker_quality's internal ReRanker() falls back to
its rule-based path, matching the pattern used in tests/test_re_ranker.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.evaluator import RAGEvaluator, _relevance_grade, _resolve_skills
from core.ir_metrics import RankingMetrics
from core.rerank_eval import RerankerMetrics


class _FakeHybridIndexer:
    """Minimal stand-in for HybridIndexer: fixed search results + full corpus metadata."""

    def __init__(self, search_results, all_metadata, resume_texts=None):
        self._search_results = search_results
        self.resume_metadata = all_metadata
        self.resume_texts = resume_texts or ["" for _ in all_metadata]

    def search_resumes(self, query, top_k=5):
        return self._search_results[:top_k]


ALL_METADATA = [
    {"candidate_id": "c1", "skills": ["Python", "SQL", "AWS"]},
    {"candidate_id": "c2", "skills": ["Java", "Spring"]},
    {"candidate_id": "c3", "skills": ["Python", "AWS"]},
    {"candidate_id": "c4", "skills": ["Kotlin"]},
]

# Ranked as the hybrid search would return them: irrelevant candidate first.
SEARCH_RESULTS = [
    {"candidate_id": "c2", "name": "Bob", "skills": ["Java", "Spring"], "text": "Java engineer"},
    {"candidate_id": "c1", "name": "Alice", "skills": ["Python", "SQL", "AWS"], "text": "Python dev"},
    {"candidate_id": "c3", "name": "Carol", "skills": ["Python", "AWS"], "text": "Python dev 2"},
]


def _make_evaluator() -> RAGEvaluator:
    indexer = _FakeHybridIndexer(SEARCH_RESULTS, ALL_METADATA)
    return RAGEvaluator(vector_store=None, hybrid_indexer=indexer)


# ---------------------------------------------------------------------------
# _resolve_skills
# ---------------------------------------------------------------------------

class TestResolveSkills:
    def test_prefers_structured_skills_when_present(self):
        skills = _resolve_skills(["Python"], "no python here", ["Python", "SQL"])
        assert skills == ["Python"]

    def test_falls_back_to_text_when_structured_skills_empty(self):
        skills = _resolve_skills([], "Senior Python developer with SQL experience", ["Python", "SQL", "AWS"])
        assert set(skills) == {"Python", "SQL"}

    def test_text_fallback_finds_nothing_when_absent(self):
        skills = _resolve_skills([], "Java developer with Spring Boot", ["Python", "SQL"])
        assert skills == []


# ---------------------------------------------------------------------------
# _relevance_grade
# ---------------------------------------------------------------------------

class TestRelevanceGrade:
    def test_strong_match_scores_two(self):
        assert _relevance_grade(["Python", "SQL", "AWS"], ["Python", "SQL", "AWS"]) == 2.0

    def test_partial_match_scores_one(self):
        assert _relevance_grade(["Python", "SQL"], ["Python", "SQL", "AWS"]) == 1.0

    def test_no_match_scores_zero(self):
        assert _relevance_grade(["Kotlin"], ["Python", "SQL", "AWS"]) == 0.0

    def test_no_expected_skills_scores_zero(self):
        assert _relevance_grade(["Python"], []) == 0.0


# ---------------------------------------------------------------------------
# evaluate_ranking_quality
# ---------------------------------------------------------------------------

class TestEvaluateRankingQuality:
    def test_returns_ranking_metrics(self):
        evaluator = _make_evaluator()
        metrics = evaluator.evaluate_ranking_quality(
            "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=3, k_values=(1, 3)
        )
        assert isinstance(metrics, RankingMetrics)
        assert metrics.num_queries == 1

    def test_recall_uses_full_corpus_not_just_retrieved(self):
        # top_k=1 only retrieves c2 (irrelevant), even though c1 and c3
        # (both relevant) exist elsewhere in the indexed corpus.
        evaluator = _make_evaluator()
        metrics = evaluator.evaluate_ranking_quality(
            "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=1, k_values=(1,)
        )
        assert metrics.recall_at_k[1] == 0.0

    def test_recall_is_perfect_when_all_relevant_are_retrieved(self):
        evaluator = _make_evaluator()
        metrics = evaluator.evaluate_ranking_quality(
            "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=3, k_values=(3,)
        )
        assert metrics.recall_at_k[3] == 1.0

    def test_no_expected_skills_yields_zero_metrics(self):
        evaluator = _make_evaluator()
        metrics = evaluator.evaluate_ranking_quality(
            "python developer", expected_skills=[], top_k=3, k_values=(3,)
        )
        assert metrics.recall_at_k[3] == 0.0
        assert metrics.mrr == 0.0

    def test_falls_back_to_text_when_no_structured_skills_indexed(self):
        # Simulates indexing without GOOGLE_API_KEY: every candidate's
        # structured `skills` metadata is empty, but relevance should still
        # be inferred from the raw resume text.
        unparsed_metadata = [{"candidate_id": m["candidate_id"], "skills": []} for m in ALL_METADATA]
        resume_texts = [
            "python sql aws developer",
            "java spring developer",
            "python aws engineer",
            "kotlin android developer",
        ]
        indexer = _FakeHybridIndexer(SEARCH_RESULTS, unparsed_metadata, resume_texts)
        evaluator = RAGEvaluator(vector_store=None, hybrid_indexer=indexer)

        metrics = evaluator.evaluate_ranking_quality(
            "python developer", expected_skills=["python", "sql", "aws"], top_k=3, k_values=(3,)
        )
        assert metrics.recall_at_k[3] == 1.0


# ---------------------------------------------------------------------------
# evaluate_reranker_quality
# ---------------------------------------------------------------------------

class TestEvaluateRerankerQuality:
    def test_returns_reranker_metrics(self):
        with patch("core.re_ranker.GOOGLE_API_KEY", ""):
            evaluator = _make_evaluator()
            metrics = evaluator.evaluate_reranker_quality(
                "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=3
            )
        assert isinstance(metrics, RerankerMetrics)

    def test_rule_based_reranker_improves_or_maintains_ndcg(self):
        # The hybrid search ranks the irrelevant candidate (c2) first; the
        # rule-based re-ranker (skill-overlap scoring) should not make the
        # ordering worse relative to relevance.
        with patch("core.re_ranker.GOOGLE_API_KEY", ""):
            evaluator = _make_evaluator()
            metrics = evaluator.evaluate_reranker_quality(
                "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=3
            )
        assert metrics.ndcg_after >= metrics.ndcg_before

    def test_relevance_judgment_and_reranker_both_use_text_fallback(self):
        # Neither candidate has structured `skills` metadata (unparsed
        # resumes), but c1's text mentions all 3 required skills and c2's
        # mentions none. Both the relevance judgment (evaluator) and the
        # rule-based re-ranker's own scoring (re_ranker.py) now fall back to
        # text matching, so the re-ranker should promote c1 above its
        # pre-rerank position — a positive NDCG uplift.
        indexer = _FakeHybridIndexer(
            search_results=[
                {"candidate_id": "c2", "name": "Bob", "skills": [], "text": "java spring developer"},
                {"candidate_id": "c1", "name": "Alice", "skills": [], "text": "python sql aws developer"},
            ],
            all_metadata=ALL_METADATA,
        )
        with patch("core.re_ranker.GOOGLE_API_KEY", ""):
            evaluator = RAGEvaluator(vector_store=None, hybrid_indexer=indexer)
            metrics = evaluator.evaluate_reranker_quality(
                "python developer", expected_skills=["Python", "SQL", "AWS"], top_k=2
            )
        assert metrics.ndcg_before > 0.0
        assert metrics.ndcg_uplift > 0.0
        assert metrics.fit_score_correlation > 0.0
