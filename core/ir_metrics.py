"""Information-retrieval quality metrics for evaluating ranked search results.

Pure, dependency-free functions — Precision@K, Recall@K, MRR, NDCG@K, MAP —
for scoring a ranked list of candidate IDs against relevance judgments. These
are the standard metrics for evaluating a ranking system (as opposed to
RAGAS-style QA metrics like faithfulness/answer_relevancy, which assume a
generated-answer/RAG use case). No LLM calls are required, so they run fast
enough to use as a regression check.

`evaluate_ranking` aggregates these metrics across a labeled benchmark of
multiple queries (a list of `QueryJudgment`).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k ranked results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = list(ranked_ids)[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k ranked results."""
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    top_k = list(ranked_ids)[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    relevant = set(relevant_ids)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = None) -> float:
    """Average precision: mean of precision@rank at each relevant hit, over all relevant items."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    candidates = list(ranked_ids)[:k] if k else list(ranked_ids)
    hits = 0
    precision_sum = 0.0
    for rank, cid in enumerate(candidates, start=1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def dcg_at_k(ranked_ids: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    """Discounted cumulative gain of the top-k ranked results."""
    top_k = list(ranked_ids)[:k]
    dcg = 0.0
    for rank, cid in enumerate(top_k, start=1):
        rel = relevance.get(cid, 0.0)
        if rel:
            dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    """DCG@k normalized by the ideal (best-possible) DCG@k, in [0, 1]."""
    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_order) if rel)
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class QueryJudgment:
    """One labeled benchmark query: a ranked result list plus relevance judgments.

    `relevance` maps candidate_id -> relevance grade (0 means "not relevant";
    higher is more relevant, e.g. 0/1/2 for none/partial/strong match).
    """
    query: str
    ranked_ids: List[str]
    relevance: Dict[str, float] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> List[str]:
        return [cid for cid, grade in self.relevance.items() if grade > 0]


@dataclass
class RankingMetrics:
    """Metrics averaged across all queries in a benchmark run."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    map: float
    num_queries: int

    def to_dict(self) -> Dict[str, float]:
        out = {"mrr": self.mrr, "map": self.map, "num_queries": self.num_queries}
        for k, v in self.precision_at_k.items():
            out[f"precision@{k}"] = v
        for k, v in self.recall_at_k.items():
            out[f"recall@{k}"] = v
        for k, v in self.ndcg_at_k.items():
            out[f"ndcg@{k}"] = v
        return out


def evaluate_ranking(judgments: List[QueryJudgment], k_values: Sequence[int] = (3, 5, 10)) -> RankingMetrics:
    """Compute Precision/Recall/NDCG@k, MRR, and MAP averaged over a set of queries."""
    if not judgments:
        return RankingMetrics(
            precision_at_k={k: 0.0 for k in k_values},
            recall_at_k={k: 0.0 for k in k_values},
            ndcg_at_k={k: 0.0 for k in k_values},
            mrr=0.0,
            map=0.0,
            num_queries=0,
        )

    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    ap_sum = 0.0

    for judgment in judgments:
        relevant = judgment.relevant_ids
        for k in k_values:
            precision_sums[k] += precision_at_k(judgment.ranked_ids, relevant, k)
            recall_sums[k] += recall_at_k(judgment.ranked_ids, relevant, k)
            ndcg_sums[k] += ndcg_at_k(judgment.ranked_ids, judgment.relevance, k)
        rr_sum += reciprocal_rank(judgment.ranked_ids, relevant)
        ap_sum += average_precision(judgment.ranked_ids, relevant)

    n = len(judgments)
    return RankingMetrics(
        precision_at_k={k: v / n for k, v in precision_sums.items()},
        recall_at_k={k: v / n for k, v in recall_sums.items()},
        ndcg_at_k={k: v / n for k, v in ndcg_sums.items()},
        mrr=rr_sum / n,
        map=ap_sum / n,
        num_queries=n,
    )


if __name__ == "__main__":
    print("=== single-query metrics ===")
    ranked = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c5"}
    relevance_grades = {"c1": 2.0, "c3": 1.0, "c5": 2.0}

    print("precision@3:", round(precision_at_k(ranked, relevant, 3), 4))
    print("recall@3:", round(recall_at_k(ranked, relevant, 3), 4))
    print("MRR:", round(reciprocal_rank(ranked, relevant), 4))
    print("AP:", round(average_precision(ranked, relevant), 4))
    print("NDCG@5:", round(ndcg_at_k(ranked, relevance_grades, 5), 4))

    print("\n=== evaluate_ranking (2-query benchmark) ===")
    judgments = [
        QueryJudgment(query="python developer", ranked_ids=ranked, relevance=relevance_grades),
        QueryJudgment(query="java engineer", ranked_ids=["c2", "c4", "c1"], relevance={"c1": 1.0}),
    ]
    metrics = evaluate_ranking(judgments, k_values=(3, 5))
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v}")
