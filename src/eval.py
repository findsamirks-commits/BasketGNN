"""Evaluation Harness for Next-Basket Recommendations.

Computes standard RecSys metrics:
- Recall@K: Fraction of ground-truth basket items in top-K
- NDCG@K: Position-discounted ranking gain
- HitRate@K: Fraction of evaluation sessions with >=1 hit in top-K
- MAP@K: Mean Average Precision at rank K
- Coverage@K: Fraction of catalog items ever recommended (anti-popularity metric)
"""

import math
from typing import Dict, List, Set, Tuple
import numpy as np
import torch


def compute_recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Computes Recall@K: |Recommended@K ∩ GroundTruth| / |GroundTruth|."""
    if not ground_truth:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & ground_truth)
    return hits / float(len(ground_truth))


def compute_hit_rate_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Computes HitRate@K (1.0 if at least 1 item in ground truth appears in top-K, else 0.0)."""
    if not ground_truth:
        return 0.0
    top_k = recommended[:k]
    return 1.0 if len(set(top_k) & ground_truth) > 0 else 0.0


def compute_ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K (NDCG@K)."""
    if not ground_truth:
        return 0.0
    top_k = recommended[:k]

    dcg = 0.0
    for rank, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(rank + 2)  # rank is 0-indexed, so rank+2 = 2, 3, 4...

    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_map_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Computes Mean Average Precision at rank K (MAP@K)."""
    if not ground_truth:
        return 0.0
    top_k = recommended[:k]

    hits = 0
    sum_precisions = 0.0
    for rank, item in enumerate(top_k):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / float(rank + 1)

    max_possible_hits = min(len(ground_truth), k)
    return sum_precisions / float(max_possible_hits) if max_possible_hits > 0 else 0.0


def evaluate_recommendations(
    user_predictions: Dict[int, List[int]],
    user_ground_truth: Dict[int, Set[int]],
    catalog_size: int,
    k_list: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Evaluates top-K recommendations across all evaluation users."""
    metrics: Dict[str, float] = {}

    for k in k_list:
        recalls = []
        ndcgs = []
        hit_rates = []
        maps = []
        recommended_catalog_items = set()

        for u, true_items in user_ground_truth.items():
            if not true_items:
                continue
            recs = user_predictions.get(u, [])
            recalls.append(compute_recall_at_k(recs, true_items, k))
            ndcgs.append(compute_ndcg_at_k(recs, true_items, k))
            hit_rates.append(compute_hit_rate_at_k(recs, true_items, k))
            maps.append(compute_map_at_k(recs, true_items, k))
            recommended_catalog_items.update(recs[:k])

        metrics[f"recall_{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"ndcg_{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        metrics[f"hit_rate_{k}"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        metrics[f"map_{k}"] = float(np.mean(maps)) if maps else 0.0
        metrics[f"coverage_{k}"] = len(recommended_catalog_items) / float(max(1, catalog_size))

    return metrics
