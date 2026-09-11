"""Unit tests for core/rerank_eval.py — pure functions, no external dependencies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rerank_eval import spearman_correlation, evaluate_reranker, RerankerMetrics


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    def test_perfect_positive_correlation(self):
        assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_constant_sequence_is_undefined_returns_zero(self):
        assert spearman_correlation([5, 5, 5], [1, 2, 3]) == 0.0

    def test_fewer_than_two_points_returns_zero(self):
        assert spearman_correlation([1], [1]) == 0.0
        assert spearman_correlation([], []) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert spearman_correlation([1, 2], [1, 2, 3]) == 0.0

    def test_handles_ties_via_average_rank(self):
        # x has a tie at the first two positions; should not raise and should
        # still land strongly positive since y is monotonically increasing.
        corr = spearman_correlation([1, 1, 2, 3], [10, 11, 20, 30])
        assert corr > 0.9

    def test_weakly_correlated_sequence_scores_near_zero(self):
        corr = spearman_correlation([1, 2, 3, 4, 5], [3, 5, 1, 4, 2])
        assert -0.5 < corr < 0.5


# ---------------------------------------------------------------------------
# evaluate_reranker
# ---------------------------------------------------------------------------

class TestEvaluateReranker:
    def test_promoting_relevant_candidate_increases_ndcg(self):
        pre_ids = ["c1", "c2", "c3", "c4"]
        post_ids = ["c3", "c1", "c4", "c2"]  # c3 (most relevant) promoted to top
        fit_scores = {"c1": 70.0, "c2": 40.0, "c3": 90.0, "c4": 55.0}
        relevance = {"c1": 1.0, "c2": 0.0, "c3": 2.0, "c4": 0.0}

        metrics = evaluate_reranker(pre_ids, post_ids, fit_scores, relevance, k=4)
        assert isinstance(metrics, RerankerMetrics)
        assert metrics.ndcg_after > metrics.ndcg_before
        assert metrics.ndcg_uplift > 0

    def test_identical_pre_and_post_order_has_zero_uplift(self):
        ids = ["c1", "c2", "c3"]
        fit_scores = {"c1": 80.0, "c2": 60.0, "c3": 40.0}
        relevance = {"c1": 2.0, "c2": 1.0, "c3": 0.0}

        metrics = evaluate_reranker(ids, ids, fit_scores, relevance, k=3)
        assert metrics.ndcg_uplift == 0.0
        assert metrics.ndcg_before == metrics.ndcg_after

    def test_demoting_relevant_candidate_decreases_ndcg(self):
        pre_ids = ["c1", "c2", "c3"]
        post_ids = ["c3", "c2", "c1"]  # c1 (most relevant) demoted to bottom
        fit_scores = {"c1": 30.0, "c2": 50.0, "c3": 80.0}
        relevance = {"c1": 2.0, "c2": 1.0, "c3": 0.0}

        metrics = evaluate_reranker(pre_ids, post_ids, fit_scores, relevance, k=3)
        assert metrics.ndcg_uplift < 0

    def test_fit_score_tracking_relevance_gives_high_correlation(self):
        ids = ["c1", "c2", "c3", "c4"]
        fit_scores = {"c1": 90.0, "c2": 70.0, "c3": 50.0, "c4": 30.0}
        relevance = {"c1": 2.0, "c2": 1.0, "c3": 1.0, "c4": 0.0}

        metrics = evaluate_reranker(ids, ids, fit_scores, relevance, k=4)
        assert metrics.fit_score_correlation > 0.8

    def test_candidates_missing_fit_score_are_excluded_from_correlation(self):
        pre_ids = ["c1", "c2", "c3"]
        post_ids = ["c1", "c2", "c3"]
        fit_scores = {"c1": 80.0, "c3": 20.0}  # c2 has no fit score
        relevance = {"c1": 2.0, "c2": 1.0, "c3": 0.0}

        # Should not raise despite the missing entry.
        metrics = evaluate_reranker(pre_ids, post_ids, fit_scores, relevance, k=3)
        assert isinstance(metrics.fit_score_correlation, float)

    def test_to_dict_has_expected_keys(self):
        ids = ["c1"]
        metrics = evaluate_reranker(ids, ids, {"c1": 50.0}, {"c1": 1.0}, k=1)
        d = metrics.to_dict()
        assert set(d.keys()) == {
            "fit_score_correlation", "ndcg_before", "ndcg_after", "ndcg_uplift"
        }
