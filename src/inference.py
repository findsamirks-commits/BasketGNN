"""Production Inference Pipeline for BasketGNN.

Serves two primary retail quick-commerce query modes:
1. User-only Query (Next-basket prediction): user_id -> top-K recommended SKUs
2. Basket-completion Query: user_id + {active cart item IDs} -> top-K complementary cross-sells

Integrates Shelf-Vision OOS masks and Fresh-Grade FEFO clearance priority signals.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.crosssell_bridge import CrossSellBridge, RecommendedSKU
from src.data import RETAIL_CATALOG, load_dataset
from src.graph import build_bipartite_adjacency
from src.model import BasketAwareHead, LightGCN


class BasketGNNPredictor:
    """Inference engine for real-time quick-commerce recommendations."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        data_dir: str = "data/processed",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.data = load_dataset(data_dir=data_dir)
        self.num_users = self.data["num_users"]
        self.num_items = self.data["num_items"]
        self.item_id_to_name = self.data["item_id_to_name"]
        self.item_id_to_category = self.data["item_id_to_category"]
        self.item_id_to_price = self.data["item_id_to_price"]

        # Default model architecture
        embedding_dim = 64
        num_layers = 3

        # Load weights if available
        state_dict = None
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    embedding_dim = checkpoint.get("embedding_dim", 64)
                    num_layers = checkpoint.get("num_layers", 3)
                else:
                    state_dict = checkpoint
            except Exception as e:
                print(f"[Warning] Failed loading weights from {weights_path}: {e}")

        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        ).to(self.device)

        if state_dict:
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Loaded BasketGNN checkpoint from {weights_path}")

        self.model.eval()

        # Build graph adjacency
        train_pairs = self.data["train_pairs"]
        user_item_edges = torch.tensor(
            [[p[0] for p in train_pairs], [p[1] for p in train_pairs]], dtype=torch.long
        )
        self.sparse_adj = build_bipartite_adjacency(
            self.num_users, self.num_items, user_item_edges, device=self.device
        )

        # Precompute propagated representations
        with torch.no_grad():
            self.final_users, self.final_items = self.model(self.sparse_adj, apply_dropout=False)

        self.head = BasketAwareHead(basket_alpha=0.45)
        self.bridge = CrossSellBridge(fefo_boost_weight=0.35, oos_penalty=-100.0)

    def recommend(
        self,
        user_id: int,
        basket_item_ids: Optional[List[int]] = None,
        top_k: int = 10,
        oos_skus: Optional[Set[int]] = None,
        fefo_skus: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """Generates ranked SKU recommendations for a user or active cart."""
        start_t = time.perf_counter()
        u = int(user_id) % self.num_users
        user_vec = self.final_users[u].unsqueeze(0)

        query_mode = "basket_completion" if basket_item_ids and len(basket_item_ids) > 0 else "next_basket"
        cart_names = []

        if query_mode == "basket_completion":
            valid_basket = [int(i) for i in basket_item_ids if 0 <= int(i) < self.num_items]
            cart_names = [self.item_id_to_name.get(i, f"SKU #{i}") for i in valid_basket]
            basket_item_tensors = self.final_items[valid_basket] if valid_basket else torch.empty(0, self.model.embedding_dim, device=self.device)
            raw_scores = self.head.predict_basket_completion(
                user_vec, basket_item_tensors, self.final_items, basket_item_ids=valid_basket
            )
        else:
            raw_scores = self.head.predict_next_basket(user_vec, self.final_items).squeeze(0)

        # Apply Shelf-Vision OOS masks & Fresh-Grade FEFO boosts
        adjusted_scores, boosted_ids = self.bridge.apply_operational_signals(
            raw_scores, oos_item_ids=oos_skus, fefo_item_ids=fefo_skus
        )

        # Rank and attach commercial rationales
        recommendations = self.bridge.rank_and_explain(
            adjusted_scores=adjusted_scores,
            item_id_to_name=self.item_id_to_name,
            item_id_to_category=self.item_id_to_category,
            item_id_to_price=self.item_id_to_price,
            top_k=top_k,
            cart_item_names=cart_names,
            fefo_item_ids=fefo_skus,
            boosted_ids=boosted_ids,
        )

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "user_id": u,
            "query_mode": query_mode,
            "current_basket": cart_names if query_mode == "basket_completion" else [],
            "recommendations": [rec.model_dump() for rec in recommendations],
            "inference_ms": round(latency_ms, 1),
            "filters_applied": {
                "shelf_vision_oos_masked": len(oos_skus) if oos_skus else 0,
                "fresh_grade_fefo_boosted": len(boosted_ids),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="BasketGNN Quick-Commerce Inference")
    parser.add_argument("--user", type=int, default=42, help="User ID to recommend for")
    parser.add_argument(
        "--basket", nargs="*", type=int, default=None, help="Active cart SKU IDs (e.g., --basket 0 1)"
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of items to recommend")
    parser.add_argument("--weights", type=str, default="runs/basketgnn/best.pt", help="Model weights path")
    args = parser.parse_args()

    predictor = BasketGNNPredictor(weights_path=args.weights if os.path.exists(args.weights) else None)

    # Demo Shelf-Vision and Fresh-Grade signals
    demo_oos = {2}  # Amul Butter currently out of stock on shelf
    demo_fefo = {10: 2.5, 11: 3.0}  # Tomato & Banana near peak freshness (2.5d and 3.0d)

    result = predictor.recommend(
        user_id=args.user,
        basket_item_ids=args.basket,
        top_k=args.topk,
        oos_skus=demo_oos,
        fefo_skus=demo_fefo,
    )

    print("\n--- BASKETGNN RECOMMENDATION REPORT ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
