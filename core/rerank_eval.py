"""Evaluate whether the re-ranker (core/re_ranker.py) actually improves
candidate ordering, using relevance judgments (e.g. skill-overlap grades
from `core.evaluator._relevance_grade`) as ground truth.

Two signals are combined:
- `fit_score_correlation` — Spearman correlation between the re-ranker's
  fit_score and the true relevance grade. High correlation means the
  re-ranker's scores track what actually matters.
- `ndcg_uplift` — NDCG@k of the post-rerank order minus NDCG@k of the
  pre-rerank (hybrid search) order. Positive means re-ranking moved more
  relevant candidates higher; negative means it made the ranking worse.
"""

import sys
sys.path.append(".")
from dataclasses import dataclass
from typing import Dict, List, Sequence

from core.ir_metrics import ndcg_at_k


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between two equal-length sequences, in [-1, 1].

    Ranks are averaged on ties. Returns 0.0 when there are fewer than 2
    points or either sequence is constant (correlation is undefined).
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0

    def _ranks(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for idx in range(i, j + 1):
                ranks[order[idx]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(list(x))
    ry = _ranks(list(y))

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


@dataclass
class RerankerMetrics:
    """Quality signal for one re-ranking run, scored against relevance judgments."""
    fit_score_correlation: float
    ndcg_before: float
    ndcg_after: float
    ndcg_uplift: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "fit_score_correlation": self.fit_score_correlation,
            "ndcg_before": self.ndcg_before,
            "ndcg_after": self.ndcg_after,
            "ndcg_uplift": self.ndcg_uplift,
        }


def evaluate_reranker(
    pre_rerank_ids: List[str],
    post_rerank_ids: List[str],
    fit_scores_by_id: Dict[str, float],
    relevance: Dict[str, float],
    k: int = 5,
) -> RerankerMetrics:
    """Score a re-ranking run against relevance judgments.

    Args:
        pre_rerank_ids: candidate IDs in hybrid-search order (before re-ranking).
        post_rerank_ids: candidate IDs in the re-ranker's output order.
        fit_scores_by_id: candidate_id -> fit_score (0-100) assigned by the re-ranker.
        relevance: candidate_id -> relevance grade (ground truth); 0 = not relevant.
        k: cutoff used for the before/after NDCG comparison.
    """
    scored_ids = [cid for cid in pre_rerank_ids if cid in fit_scores_by_id]
    fit_scores = [fit_scores_by_id[cid] for cid in scored_ids]
    grades = [relevance.get(cid, 0.0) for cid in scored_ids]
    correlation = spearman_correlation(fit_scores, grades)

    ndcg_before = ndcg_at_k(pre_rerank_ids, relevance, k)
    ndcg_after = ndcg_at_k(post_rerank_ids, relevance, k)

    return RerankerMetrics(
        fit_score_correlation=correlation,
        ndcg_before=ndcg_before,
        ndcg_after=ndcg_after,
        ndcg_uplift=ndcg_after - ndcg_before,
    )


if __name__ == "__main__":
    print("=== spearman_correlation ===")
    print("perfect positive:", round(spearman_correlation([1, 2, 3], [10, 20, 30]), 4))
    print("perfect negative:", round(spearman_correlation([1, 2, 3], [30, 20, 10]), 4))
    print("constant (undefined):", round(spearman_correlation([1, 1, 1], [1, 2, 3]), 4))

    print("\n=== evaluate_reranker ===")
    pre_ids = ["c1", "c2", "c3", "c4"]
    post_ids = ["c3", "c1", "c4", "c2"]  # re-ranker promoted c3
    fit_scores = {"c1": 70.0, "c2": 40.0, "c3": 90.0, "c4": 55.0}
    relevance = {"c1": 1.0, "c2": 0.0, "c3": 2.0, "c4": 0.0}

    metrics = evaluate_reranker(pre_ids, post_ids, fit_scores, relevance, k=4)
    for key, val in metrics.to_dict().items():
        print(f"  {key}: {round(val, 4)}")
