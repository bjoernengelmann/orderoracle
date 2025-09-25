from __future__ import annotations

from typing import Dict, List, Sequence
import math

from ..data.models import Collection, Qrel


def _build_qrel_lookup(qrels: Sequence[Qrel]) -> Dict[str, Dict[str, int]]:
    lookup: Dict[str, Dict[str, int]] = {}
    for q in qrels:
        lookup.setdefault(q.topic_id, {})[q.doc_id] = int(q.relevance)
    return lookup


def _avg_precision_for_topic(ranking: Sequence[str], rel_lookup: Dict[str, int], k: int = 100) -> float:
    if not ranking:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranking[:k], start=1):
        if rel_lookup.get(doc_id, 0) > 0:
            hits += 1
            sum_prec += hits / i
    return 0.0 if hits == 0 else sum_prec / hits


def _dcg_at_k(gains: List[float], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        # positions are 1-based in DCG: discount by log2(i+2)
        dcg += (2**g - 1) / (math.log2(i + 2))
    return dcg


def _ndcg_for_topic(ranking: Sequence[str], rel_lookup: Dict[str, int], k: int = 10) -> float:
    if not ranking:
        return 0.0
    gains = [float(rel_lookup.get(doc_id, 0)) for doc_id in ranking]
    ideal_gains = sorted(gains, reverse=True)
    if not any(ideal_gains):
        return 0.0
    dcg = _dcg_at_k(gains, k)
    idcg = _dcg_at_k(ideal_gains, k)
    return 0.0 if idcg <= 0 else dcg / idcg


def mean_average_precision_from_ranking(collection: Collection, ranking_by_topic: Dict[str, Sequence[str]], k: int = 100) -> float:
    qrels_by_topic = _build_qrel_lookup(collection.qrels)
    ap_vals: List[float] = []
    for topic_id, ranking in ranking_by_topic.items():
        rel_lookup = qrels_by_topic.get(topic_id, {})
        ap_vals.append(_avg_precision_for_topic(ranking, rel_lookup, k=k))
    return sum(ap_vals) / len(ap_vals) if ap_vals else 0.0


