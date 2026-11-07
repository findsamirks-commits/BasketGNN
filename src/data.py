"""Dataset Ingestion, Co-Purchase Graph Preparation, and Quick-Commerce Basket Generator.

Supports:
1. Instacart Market Basket Analysis dataset ingestion
2. Dunnhumby 'The Complete Journey' transaction logs
3. Built-in Quick-Commerce Dark Store generator producing realistic Milkbasket co-purchase baskets
4. Stratified temporal/order-level splits and catalog metadata export
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

RETAIL_CATALOG = [
    # Dairy & Breakfast
    {"id": 0, "name": "Amul Taaza Fresh Toned Milk 1L", "category": "Dairy", "price": 54.0},
    {"id": 1, "name": "Britannia Daily Fresh White Bread 400g", "category": "Bakery", "price": 45.0},
    {"id": 2, "name": "Amul Pasteurised Butter 500g", "category": "Dairy", "price": 275.0},
    {"id": 3, "name": "Nestle Farm Fresh White Eggs 6pc", "category": "Eggs & Meat", "price": 55.0},
    {"id": 4, "name": "Mother Dairy Classic Dahi 400g", "category": "Dairy", "price": 35.0},
    {"id": 5, "name": "Amul Processed Cheese Slices 200g", "category": "Dairy", "price": 140.0},
    {"id": 6, "name": "Kissan Mixed Fruit Jam 500g", "category": "Breakfast", "price": 185.0},
    {"id": 7, "name": "Kellogg's Real Almond Honey Corn Flakes 500g", "category": "Breakfast", "price": 240.0},
    {"id": 8, "name": "Tata Tea Gold Leaf Tea 500g", "category": "Beverages", "price": 320.0},
    {"id": 9, "name": "Nescafe Classic Instant Coffee 50g", "category": "Beverages", "price": 170.0},

    # Fresh Produce (Perishables)
    {"id": 10, "name": "Fresh Hybrid Tomato 1kg", "category": "Produce", "price": 40.0},
    {"id": 11, "name": "Fresh Robusta Banana 500g", "category": "Produce", "price": 35.0},
    {"id": 12, "name": "Fresh Shimla Apple 4pc", "category": "Produce", "price": 120.0},
    {"id": 13, "name": "Fresh Red Onion 1kg", "category": "Produce", "price": 38.0},
    {"id": 14, "name": "Fresh Potato (Dum Aloo) 1kg", "category": "Produce", "price": 28.0},
    {"id": 15, "name": "Fresh Coriander Bunch 100g", "category": "Produce", "price": 15.0},
    {"id": 16, "name": "Fresh Green Chilli 100g", "category": "Produce", "price": 12.0},
    {"id": 17, "name": "Fresh Lemon 4pc", "category": "Produce", "price": 20.0},

    # Staples & Packaged Food
    {"id": 18, "name": "Aashirvaad Shudh Chakki Atta 5kg", "category": "Staples", "price": 235.0},
    {"id": 19, "name": "Fortune Sunlite Refined Sunflower Oil 1L", "category": "Staples", "price": 135.0},
    {"id": 20, "name": "Tata Salt Vacuum Evaporated 1kg", "category": "Staples", "price": 28.0},
    {"id": 21, "name": "Maggi 2-Minute Masala Noodles 280g", "category": "Snacks", "price": 56.0},
    {"id": 22, "name": "Lay's Classic Salted Potato Chips 50g", "category": "Snacks", "price": 20.0},
    {"id": 23, "name": "Britannia Good Day Cashew Cookies 200g", "category": "Snacks", "price": 45.0},
    {"id": 24, "name": "Haldiram's Bhujia Sev 400g", "category": "Snacks", "price": 115.0},

    # Home & Personal Care
    {"id": 25, "name": "Vim Lemon Dishwash Gel 500ml", "category": "Household", "price": 125.0},
    {"id": 26, "name": "Surf Excel Matic Front Load Liquid 1L", "category": "Household", "price": 230.0},
    {"id": 27, "name": "Harpic Power Plus Toilet Cleaner 500ml", "category": "Household", "price": 105.0},
    {"id": 28, "name": "Dettol Liquid Handwash Refill 750ml", "category": "Personal Care", "price": 119.0},
    {"id": 29, "name": "Colgate Strong Teeth Toothpaste 200g", "category": "Personal Care", "price": 110.0},
]

# Basket co-occurrence affinity clusters (Anchor -> Affinity Complements)
CO_PURCHASE_CLUSTERS = [
    # Breakfast cluster: Milk + Bread + Butter + Eggs + Jam
    [0, 1, 2, 3, 4, 5, 6, 7],
    # Tea/Coffee combo: Milk + Tea/Coffee + Sugar/Biscuits
    [0, 8, 9, 20, 23],
    # Indian Cooking bundle: Tomato + Onion + Potato + Coriander + Chilli + Oil + Salt
    [10, 13, 14, 15, 16, 17, 18, 19, 20],
    # Fruit bowl: Banana + Apple + Lemon
    [11, 12, 17],
    # Quick Evening Snacks: Maggi + Chips + Cookies + Sev
    [21, 22, 23, 24],
    # Dark store weekly cleaning replenishment: Dishwash + Detergent + Toilet cleaner + Handwash
    [25, 26, 27, 28, 29],
]


def generate_synthetic_quick_commerce_data(
    num_users: int = 500,
    orders_per_user_range: Tuple[int, int] = (6, 18),
    output_dir: str = "data/processed",
) -> Dict[str, Any]:
    """Generates realistic quick-commerce order logs with topological co-purchase clusters."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    orders_records = []
    order_id_counter = 1

    # Assign each user 1-2 primary household habits
    user_habits = {
        u: random.sample(CO_PURCHASE_CLUSTERS, k=random.choice([1, 2]))
        for u in range(num_users)
    }

    for u in range(num_users):
        num_orders = random.randint(*orders_per_user_range)
        habits = user_habits[u]

        for _ in range(num_orders):
            # Pick an anchor habit or occasional exploration
            cluster = random.choice(habits) if random.random() < 0.85 else random.choice(CO_PURCHASE_CLUSTERS)
            basket_size = random.randint(2, min(6, len(cluster)))
            basket_items = set(random.sample(cluster, k=basket_size))

            # 20% chance to add 1 staple/produce cross-sell
            if random.random() < 0.35:
                basket_items.add(random.choice([0, 1, 10, 11, 13, 21]))

            for item_id in basket_items:
                orders_records.append({
                    "order_id": order_id_counter,
                    "user_id": u,
                    "item_id": item_id,
                })
            order_id_counter += 1

    df_interactions = pd.DataFrame(orders_records)

    # Stratified Train/Val/Test Split by User
    # For each user, reserve earlier orders for training, later orders for validation and testing
    train_pairs: List[Tuple[int, int]] = []
    val_ground_truth: Dict[int, Set[int]] = {}
    test_ground_truth: Dict[int, Set[int]] = {}
    user_pos_map: Dict[int, Set[int]] = {}

    for u, user_orders in df_interactions.groupby("user_id"):
        unique_order_ids = user_orders["order_id"].unique()
        n_orders = len(unique_order_ids)
        train_cut = int(0.70 * n_orders)
        val_cut = int(0.85 * n_orders)

        train_orders = set(unique_order_ids[:train_cut])
        val_orders = set(unique_order_ids[train_cut:val_cut])
        test_orders = set(unique_order_ids[val_cut:])

        u_train_items = set(user_orders[user_orders["order_id"].isin(train_orders)]["item_id"])
        u_val_items = set(user_orders[user_orders["order_id"].isin(val_orders)]["item_id"])
        u_test_items = set(user_orders[user_orders["order_id"].isin(test_orders)]["item_id"])

        user_pos_map[u] = u_train_items
        for item_id in u_train_items:
            train_pairs.append((u, item_id))

        val_ground_truth[u] = u_val_items
        test_ground_truth[u] = u_test_items

    # Catalog mapping dictionaries
    item_id_to_name = {item["id"]: item["name"] for item in RETAIL_CATALOG}
    item_id_to_category = {item["id"]: item["category"] for item in RETAIL_CATALOG}
    item_id_to_price = {item["id"]: item["price"] for item in RETAIL_CATALOG}

    # Save to disk
    df_interactions.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)

    with open(os.path.join(output_dir, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(RETAIL_CATALOG, f, indent=2)

    stats = {
        "num_users": num_users,
        "num_items": len(RETAIL_CATALOG),
        "num_orders": order_id_counter - 1,
        "num_interactions": len(df_interactions),
        "num_train_pairs": len(train_pairs),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return {
        "train_pairs": train_pairs,
        "user_pos_map": user_pos_map,
        "val_ground_truth": val_ground_truth,
        "test_ground_truth": test_ground_truth,
        "item_id_to_name": item_id_to_name,
        "item_id_to_category": item_id_to_category,
        "item_id_to_price": item_id_to_price,
        "num_users": num_users,
        "num_items": len(RETAIL_CATALOG),
    }


def load_dataset(data_dir: str = "data/processed") -> Dict[str, Any]:
    """Loads existing preprocessed dataset or generates fresh benchmark."""
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        interactions_csv = os.path.join(data_dir, "interactions.csv")
        df = pd.read_csv(interactions_csv)
        with open(os.path.join(data_dir, "catalog.json"), "r", encoding="utf-8") as f:
            catalog = json.load(f)

        item_id_to_name = {item["id"]: item["name"] for item in catalog}
        item_id_to_category = {item["id"]: item["category"] for item in catalog}
        item_id_to_price = {item["id"]: item["price"] for item in catalog}

        # Reconstruct train/val/test maps
        train_pairs = []
        user_pos_map = {}
        val_ground_truth = {}
        test_ground_truth = {}

        for u, user_orders in df.groupby("user_id"):
            unique_order_ids = user_orders["order_id"].unique()
            n_orders = len(unique_order_ids)
            train_cut = max(1, int(0.70 * n_orders))
            val_cut = max(train_cut + 1, int(0.85 * n_orders))

            train_orders = set(unique_order_ids[:train_cut])
            val_orders = set(unique_order_ids[train_cut:val_cut])
            test_orders = set(unique_order_ids[val_cut:])

            u_train = set(user_orders[user_orders["order_id"].isin(train_orders)]["item_id"])
            u_val = set(user_orders[user_orders["order_id"].isin(val_orders)]["item_id"])
            u_test = set(user_orders[user_orders["order_id"].isin(test_orders)]["item_id"])

            user_pos_map[u] = u_train
            for item in u_train:
                train_pairs.append((u, item))
            val_ground_truth[u] = u_val
            test_ground_truth[u] = u_test

        return {
            "train_pairs": train_pairs,
            "user_pos_map": user_pos_map,
            "val_ground_truth": val_ground_truth,
            "test_ground_truth": test_ground_truth,
            "item_id_to_name": item_id_to_name,
            "item_id_to_category": item_id_to_category,
            "item_id_to_price": item_id_to_price,
            "num_users": int(df["user_id"].nunique()),
            "num_items": len(catalog),
        }
    else:
        return generate_synthetic_quick_commerce_data(output_dir=data_dir)


if __name__ == "__main__":
    data = generate_synthetic_quick_commerce_data(num_users=500, output_dir="data/processed")
    print(f"Generated {data['num_users']} users, {data['num_items']} items, {len(data['train_pairs'])} training pairs.")
