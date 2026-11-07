"""Cross-Sell Merchandising Bridge: Chains RecSys with CV Shelf Audit & Produce Quality.

Key capabilities:
1. Zero Ghost-Inventory Filtering: Ingests OOS mask from Shelf-Vision to eliminate stockout recommendations.
2. FEFO Clearance Acceleration: Ingests Fresh-Grade SELL_FIRST signals to boost near-peak perishables.
3. Commercial Explainability: Generates human-understandable merchandising rationales
   (e.g., 'co-purchase with Bread', 'FEFO priority: 3.5d left', 'user profile preference').
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from pydantic import BaseModel, Field


class RecommendedSKU(BaseModel):
    """Structured SKU recommendation with merchandising justification."""

    rank: int = Field(description="Display rank (1 to K)")
    item_id: int = Field(description="Internal item index")
    sku_name: str = Field(description="Retail SKU name (e.g., Amul Butter 500g)")
    category: str = Field(description="Merchandising category (Dairy, Bakery, F&V)")
    score: float = Field(description="Final calibrated recommendation score")
    reason: str = Field(description="Merchandising rationale explaining why SKU was recommended")
    is_fefo_boosted: bool = Field(default=False, description="Whether item received FEFO priority boost")
    base_price_inr: float = Field(default=50.0, description="Retail selling price in INR")


class CrossSellBridge:
    """Applies retail business constraints and operational signals to raw GNN logits."""

    def __init__(
        self,
        fefo_boost_weight: float = 0.35,
        oos_penalty: float = -100.0,
        enable_oos_filtering: bool = True,
        enable_fefo_boosting: bool = True,
    ):
        self.fefo_boost_weight = fefo_boost_weight
        self.oos_penalty = oos_penalty
        self.enable_oos_filtering = enable_oos_filtering
        self.enable_fefo_boosting = enable_fefo_boosting

    def apply_operational_signals(
        self,
        raw_scores: torch.Tensor,
        oos_item_ids: Optional[Set[int]] = None,
        fefo_item_ids: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, Set[int]]:
        """Applies Shelf-Vision OOS masks and Fresh-Grade FEFO priority boosts to raw logits.

        Args:
            raw_scores: 1D Tensor of raw candidate scores [num_items]
            oos_item_ids: Set of item IDs currently out of stock on dark-store shelves
            fefo_item_ids: Dict mapping perishable item IDs to remaining days-to-spoil

        Returns:
            adjusted_scores: 1D Tensor of adjusted scores
            boosted_ids: Set of items that received FEFO boost
        """
        scores = raw_scores.clone()
        boosted_ids = set()

        # 1. Hard OOS Filtering (Shelf-Vision integration)
        if self.enable_oos_filtering and oos_item_ids:
            for item_id in oos_item_ids:
                if 0 <= item_id < scores.size(0):
                    scores[item_id] += self.oos_penalty

        # 2. FEFO Clearance Boost (Fresh-Grade integration)
        if self.enable_fefo_boosting and fefo_item_ids:
            for item_id, days in fefo_item_ids.items():
                if 0 <= item_id < scores.size(0):
                    # Higher boost if remaining shelf-life is within the critical 2-4 day window
                    boost = self.fefo_boost_weight * (1.0 + (1.0 / max(1.0, days)))
                    scores[item_id] += boost
                    boosted_ids.add(item_id)

        return scores, boosted_ids

    def rank_and_explain(
        self,
        adjusted_scores: torch.Tensor,
        item_id_to_name: Dict[int, str],
        item_id_to_category: Dict[int, str],
        item_id_to_price: Dict[int, float],
        top_k: int = 10,
        cart_item_names: Optional[List[str]] = None,
        fefo_item_ids: Optional[Dict[int, float]] = None,
        boosted_ids: Optional[Set[int]] = None,
    ) -> List[RecommendedSKU]:
        """Sorts top-K items and attaches merchandising explanations."""
        values, indices = torch.topk(adjusted_scores, k=min(top_k, adjusted_scores.size(0)))
        recommendations = []
        boosted_set = boosted_ids or set()
        fefo_map = fefo_item_ids or {}

        for rank_idx, (score_val, item_idx_t) in enumerate(zip(values, indices)):
            item_id = int(item_idx_t.item())
            score = round(float(score_val.item()), 2)

            # Skip items filtered out by OOS penalty
            if score < -50.0:
                continue

            sku_name = item_id_to_name.get(item_id, f"SKU #{item_id}")
            cat = item_id_to_category.get(item_id, "General Grocery")
            price = item_id_to_price.get(item_id, 50.0)

            # Determine merchandising rationale
            is_fefo = item_id in boosted_set
            if is_fefo and item_id in fefo_map:
                days = fefo_map[item_id]
                reason = f"FEFO priority clearance ({days:.1f}d remaining shelf-life)"
            elif cart_item_names and len(cart_item_names) > 0:
                # Find affinity with cart
                anchor = cart_item_names[0]
                for c_name in cart_item_names:
                    if ("Milk" in c_name and "Egg" in sku_name) or ("Bread" in c_name and "Butter" in sku_name):
                        anchor = c_name
                        break
                reason = f"Co-purchase attachment with {anchor}"
            else:
                reason = "User recurring consumption pattern"

            recommendations.append(
                RecommendedSKU(
                    rank=rank_idx + 1,
                    item_id=item_id,
                    sku_name=sku_name,
                    category=cat,
                    score=score,
                    reason=reason,
                    is_fefo_boosted=is_fefo,
                    base_price_inr=price,
                )
            )

        return recommendations
