"""Unit Tests for BasketGNN Evaluation Metrics, LightGCN Propagation, and Bridge Governance."""

import math
import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.crosssell_bridge import CrossSellBridge
from src.eval import (
    compute_hit_rate_at_k,
    compute_map_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_recommendations,
)
from src.graph import build_bipartite_adjacency
from src.model import BasketAwareHead, LightGCN


def test_recall_at_k_correctness():
    """Validates Recall@K fraction."""
    rec = [10, 20, 30, 40, 50]
    gt = {20, 40, 99}  # 2 hits out of 3 ground truth items in top 5
    assert compute_recall_at_k(rec, gt, k=5) == pytest.approx(2.0 / 3.0)
    assert compute_recall_at_k(rec, gt, k=2) == pytest.approx(1.0 / 3.0)
    assert compute_recall_at_k(rec, set(), k=5) == 0.0


def test_hit_rate_at_k_correctness():
    """Validates binary HitRate@K."""
    rec = [1, 2, 3, 4, 5]
    assert compute_hit_rate_at_k(rec, {3}, k=5) == 1.0
    assert compute_hit_rate_at_k(rec, {9}, k=5) == 0.0
    assert compute_hit_rate_at_k(rec, {5}, k=3) == 0.0


def test_ndcg_at_k_discounting():
    """Validates that a hit at rank 1 gives higher NDCG than a hit at rank 5."""
    gt = {100}
    recs_rank_1 = [100, 2, 3, 4, 5]
    recs_rank_5 = [1, 2, 3, 4, 100]

    ndcg_1 = compute_ndcg_at_k(recs_rank_1, gt, k=5)
    ndcg_5 = compute_ndcg_at_k(recs_rank_5, gt, k=5)

    assert ndcg_1 == pytest.approx(1.0)  # Ideal DCG at rank 1
    assert ndcg_5 == pytest.approx(1.0 / math.log2(6))  # Discounted at rank 5
    assert ndcg_1 > ndcg_5


def test_map_at_k_correctness():
    """Validates Mean Average Precision."""
    rec = [1, 2, 3, 4]
    gt = {1, 3}  # Hit at rank 1 (P=1/1=1.0) and rank 3 (P=2/3=0.667)
    # Average precision = (1.0 + 0.6667) / 2 = 0.8333
    expected_map = (1.0 + (2.0 / 3.0)) / 2.0
    assert compute_map_at_k(rec, gt, k=4) == pytest.approx(expected_map, rel=1e-3)


def test_lightgcn_bipartite_propagation():
    """Verifies LightGCN forward pass produces valid user and item embeddings."""
    num_users = 4
    num_items = 5
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    sparse_adj = build_bipartite_adjacency(num_users, num_items, edges)

    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    users, items = model(sparse_adj)

    assert users.shape == (num_users, 16)
    assert items.shape == (num_items, 16)
    assert not torch.isnan(users).any()
    assert not torch.isnan(items).any()


def test_basket_aware_head_prediction():
    """Verifies that basket completion masks out cart items and accounts for cart context."""
    head = BasketAwareHead(basket_alpha=0.5)
    user_emb = torch.randn(1, 16)
    cart_emb = torch.randn(2, 16)
    all_items = torch.randn(10, 16)

    scores = head.predict_basket_completion(
        user_emb, cart_emb, all_items, basket_item_ids=[2, 3]
    )

    assert scores.shape == (10,)
    assert scores[2].item() < -1e8, "Item 2 in cart must be masked with negative infinity"
    assert scores[3].item() < -1e8, "Item 3 in cart must be masked with negative infinity"


def test_crosssell_bridge_shelf_vision_oos_masking():
    """Validates that Shelf-Vision OOS items are completely penalized and excluded from top-K."""
    bridge = CrossSellBridge(oos_penalty=-100.0)
    raw_scores = torch.tensor([5.0, 4.0, 3.0, 2.0])
    oos_items = {0}  # Top-scoring item is out of stock on shelf

    adj_scores, _ = bridge.apply_operational_signals(raw_scores, oos_item_ids=oos_items)
    assert adj_scores[0].item() < -90.0

    id_to_name = {0: "Amul Milk", 1: "Bread", 2: "Eggs", 3: "Butter"}
    id_to_cat = {i: "Grocery" for i in range(4)}
    id_to_price = {i: 50.0 for i in range(4)}

    recs = bridge.rank_and_explain(adj_scores, id_to_name, id_to_cat, id_to_price, top_k=3)
    rec_names = [r.sku_name for r in recs]

    assert "Amul Milk" not in rec_names, "OOS item must not appear in recommendations"
    assert rec_names[0] == "Bread"


def test_crosssell_bridge_fresh_grade_fefo_boosting():
    """Validates that near-expiry items receive FEFO priority rank promotion."""
    bridge = CrossSellBridge(fefo_boost_weight=2.0)
    # Item 1 has lower score (3.0) than Item 0 (4.0)
    raw_scores = torch.tensor([4.0, 3.0, 2.0])
    fefo_items = {1: 2.0}  # Item 1 has 2.0 days shelf life

    adj_scores, boosted = bridge.apply_operational_signals(raw_scores, fefo_item_ids=fefo_items)
    assert 1 in boosted
    # With boost, Item 1 should overtake Item 0
    assert adj_scores[1] > adj_scores[0]
