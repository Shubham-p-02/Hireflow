"""Unit tests for core/ir_metrics.py — pure functions, no external dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ir_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    QueryJudgment,
    RankingMetrics,
    evaluate_ranking,
)


RANKED = ["c1", "c2", "c3", "c4", "c5"]
RELEVANT = {"c1", "c3", "c5"}
GRADES = {"c1": 2.0, "c3": 1.0, "c5": 2.0}


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["c1", "c2"], {"c1", "c2"}, 2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["c1", "c2"], {"c9"}, 2) == 0.0

    def test_partial_match(self):
        assert precision_at_k(RANKED, RELEVANT, 3) == 2 / 3

    def test_k_zero_returns_zero(self):
        assert precision_at_k(RANKED, RELEVANT, 0) == 0.0

    def test_empty_ranked_list(self):
        assert precision_at_k([], RELEVANT, 5) == 0.0

    def test_k_larger_than_list_uses_full_list(self):
        assert precision_at_k(RANKED, RELEVANT, 100) == 3 / 5


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_finds_all_relevant(self):
        assert recall_at_k(RANKED, RELEVANT, 5) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(RANKED, RELEVANT, 1) == 1 / 3

    def test_no_relevant_items_returns_zero(self):
        assert recall_at_k(RANKED, set(), 5) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(RANKED, RELEVANT, 0) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank(RANKED, RELEVANT) == 1.0

    def test_second_result_relevant(self):
        assert reciprocal_rank(["c2", "c1"], {"c1"}) == 0.5

    def test_no_relevant_result_returns_zero(self):
        assert reciprocal_rank(RANKED, {"c9"}) == 0.0

    def test_empty_ranked_list_returns_zero(self):
        assert reciprocal_rank([], RELEVANT) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_perfect_ranking(self):
        assert average_precision(["c1", "c3", "c5"], RELEVANT) == 1.0

    def test_no_relevant_hits(self):
        assert average_precision(RANKED, {"c9"}) == 0.0

    def test_no_relevant_items_returns_zero(self):
        assert average_precision(RANKED, set()) == 0.0

    def test_worst_case_ordering_scores_lower_than_best_case(self):
        best = average_precision(["c1", "c3", "c5", "c2", "c4"], RELEVANT)
        worst = average_precision(["c2", "c4", "c1", "c3", "c5"], RELEVANT)
        assert best > worst


# ---------------------------------------------------------------------------
# dcg_at_k / ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_ideal_order_scores_one(self):
        ideal = ["c1", "c5", "c3", "c2", "c4"]  # sorted by grade: 2, 2, 1, 0, 0
        assert ndcg_at_k(ideal, GRADES, 5) == 1.0

    def test_worse_order_scores_less_than_one(self):
        worse = ["c2", "c4", "c3", "c1", "c5"]
        assert 0.0 < ndcg_at_k(worse, GRADES, 5) < 1.0

    def test_no_relevance_returns_zero(self):
        assert ndcg_at_k(RANKED, {}, 5) == 0.0

    def test_dcg_rewards_earlier_relevant_hits(self):
        early = dcg_at_k(["c1", "c2"], GRADES, 2)
        late = dcg_at_k(["c2", "c1"], GRADES, 2)
        assert early > late


# ---------------------------------------------------------------------------
# QueryJudgment
# ---------------------------------------------------------------------------

class TestQueryJudgment:
    def test_relevant_ids_excludes_zero_grade(self):
        judgment = QueryJudgment(query="q", ranked_ids=RANKED, relevance={"c1": 2.0, "c2": 0.0})
        assert judgment.relevant_ids == ["c1"]

    def test_relevant_ids_empty_when_no_relevance(self):
        judgment = QueryJudgment(query="q", ranked_ids=RANKED)
        assert judgment.relevant_ids == []


# ---------------------------------------------------------------------------
# evaluate_ranking (aggregator)
# ---------------------------------------------------------------------------

class TestEvaluateRanking:
    def test_empty_judgments_returns_zeroed_metrics(self):
        metrics = evaluate_ranking([], k_values=(1, 5))
        assert isinstance(metrics, RankingMetrics)
        assert metrics.num_queries == 0
        assert metrics.mrr == 0.0
        assert metrics.precision_at_k == {1: 0.0, 5: 0.0}

    def test_single_perfect_query(self):
        # Ideal order: both grade-2 items (c1, c5) before the grade-1 item (c3).
        judgment = QueryJudgment(query="q", ranked_ids=["c1", "c5", "c3"], relevance=GRADES)
        metrics = evaluate_ranking([judgment], k_values=(3,))
        assert metrics.num_queries == 1
        assert metrics.precision_at_k[3] == 1.0
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.mrr == 1.0
        assert metrics.map == 1.0
        assert metrics.ndcg_at_k[3] == 1.0

    def test_averages_across_multiple_queries(self):
        perfect = QueryJudgment(query="q1", ranked_ids=["c1"], relevance={"c1": 1.0})
        miss = QueryJudgment(query="q2", ranked_ids=["c2"], relevance={"c1": 1.0})
        metrics = evaluate_ranking([perfect, miss], k_values=(1,))
        assert metrics.precision_at_k[1] == 0.5
        assert metrics.mrr == 0.5

    def test_to_dict_has_expected_keys(self):
        judgment = QueryJudgment(query="q", ranked_ids=["c1"], relevance={"c1": 1.0})
        metrics = evaluate_ranking([judgment], k_values=(1,))
        d = metrics.to_dict()
        assert "precision@1" in d
        assert "recall@1" in d
        assert "ndcg@1" in d
        assert "mrr" in d
        assert "map" in d
