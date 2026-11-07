"""Training Pipeline for BasketGNN (LightGCN on User-Item Co-Purchase Graph).

Trains LightGCN embeddings via BPR ranking loss, evaluates on validation baskets
using Recall@10, NDCG@10, HitRate@10, and saves early-best checkpoint to runs/basketgnn/best.pt.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data import load_dataset
from src.eval import evaluate_recommendations
from src.graph import build_bipartite_adjacency
from src.model import BPRLoss, LightGCN
from src.sampler import create_bpr_dataloader


def train(config_path: str = "configs/basketgnn.yaml") -> Dict[str, Any]:
    """Runs complete LightGCN training and validation loop."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "model": {"embedding_dim": 64, "num_layers": 3, "edge_dropout": 0.1},
            "training": {
                "epochs": 15,
                "batch_size": 1024,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping_patience": 5,
            },
            "data": {"processed_dir": "data/processed"},
        }

    save_dir = "runs/basketgnn"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🛒 Initializing BasketGNN Training on [{device}]")

    # 1. Load Data
    data_dir = cfg["data"].get("processed_dir", "data/processed")
    data = load_dataset(data_dir=data_dir)

    num_users = data["num_users"]
    num_items = data["num_items"]
    train_pairs = data["train_pairs"]
    user_pos_map = data["user_pos_map"]
    val_ground_truth = data["val_ground_truth"]
    test_ground_truth = data["test_ground_truth"]

    print(
        f"Graph Nodes: {num_users} Users | {num_items} Items | {len(train_pairs)} Train Interactions"
    )

    # 2. Build Bipartite Adjacency
    user_item_edges = torch.tensor(
        [[pair[0] for pair in train_pairs], [pair[1] for pair in train_pairs]], dtype=torch.long
    )
    sparse_adj = build_bipartite_adjacency(num_users, num_items, user_item_edges, device=device)

    # 3. Create Model & Optimizer
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=m_cfg.get("embedding_dim", 64),
        num_layers=m_cfg.get("num_layers", 3),
        edge_dropout_rate=m_cfg.get("edge_dropout", 0.1),
    ).to(device)

    criterion = BPRLoss(reg_lambda=t_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("learning_rate", 1e-3))

    train_loader = create_bpr_dataloader(
        num_users,
        num_items,
        train_pairs,
        user_pos_map,
        batch_size=t_cfg.get("batch_size", 1024),
        shuffle=True,
    )

    epochs = t_cfg.get("epochs", 15)
    patience = t_cfg.get("early_stopping_patience", 5)
    best_ndcg = -1.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(save_dir, "best.pt")

    print("-" * 75)
    history = []

    for epoch in range(epochs):
        start_t = time.perf_counter()
        model.train()
        total_bpr_loss = 0.0
        total_reg_loss = 0.0

        for batch_users, batch_pos, batch_neg in train_loader:
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()

            # Multi-hop propagation with edge dropout
            all_users, all_items = model(sparse_adj, apply_dropout=True)

            u_emb = all_users[batch_users]
            pos_emb = all_items[batch_pos]
            neg_emb = all_items[batch_neg]

            u_ego = model.user_embedding(batch_users)
            pos_ego = model.item_embedding(batch_pos)
            neg_ego = model.item_embedding(batch_neg)

            loss, bpr, reg = criterion(u_emb, pos_emb, neg_emb, u_ego, pos_ego, neg_ego)
            loss.backward()
            optimizer.step()

            total_bpr_loss += bpr.item()
            total_reg_loss += reg.item()

        avg_bpr = total_bpr_loss / max(1, len(train_loader))
        elapsed = time.perf_counter() - start_t

        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            final_users, final_items = model(sparse_adj, apply_dropout=False)
            val_preds = {}

            # Score all items for validation users
            for u in val_ground_truth.keys():
                u_vec = final_users[u].unsqueeze(0)
                scores = torch.matmul(u_vec, final_items.t()).squeeze(0)
                # Mask out items already bought in training
                if u in user_pos_map:
                    scores[list(user_pos_map[u])] = -1e9
                top_items = torch.topk(scores, k=20).indices.cpu().numpy().tolist()
                val_preds[u] = top_items

            metrics = evaluate_recommendations(
                val_preds, val_ground_truth, catalog_size=num_items, k_list=[5, 10, 20]
            )

        val_recall_10 = metrics["recall_10"]
        val_ndcg_10 = metrics["ndcg_10"]
        val_hit_10 = metrics["hit_rate_10"]

        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] "
            f"BPR Loss: {avg_bpr:.4f} | "
            f"Val Recall@10: {val_recall_10:.4f} | "
            f"Val NDCG@10: {val_ndcg_10:.4f} | "
            f"HitRate@10: {val_hit_10*100:.1f}% ({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch + 1,
            "bpr_loss": round(avg_bpr, 4),
            "recall_10": round(val_recall_10, 4),
            "ndcg_10": round(val_ndcg_10, 4),
            "hit_rate_10": round(val_hit_10, 4),
        })

        if val_ndcg_10 > best_ndcg:
            best_ndcg = val_ndcg_10
            patience_counter = 0

            # Save model checkpoint + metadata
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_users": num_users,
                    "num_items": num_items,
                    "embedding_dim": m_cfg.get("embedding_dim", 64),
                    "num_layers": m_cfg.get("num_layers", 3),
                    "best_ndcg_10": best_ndcg,
                },
                best_checkpoint_path,
            )
            print(f"   ⭐ New best checkpoint saved! (NDCG@10: {best_ndcg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   [Early stopping triggered after {patience} epochs without improvement]")
                break

    print("-" * 75)
    print(f"✅ Training completed! Best model saved to: {best_checkpoint_path}")
    return {"best_ndcg_10": best_ndcg, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BasketGNN LightGCN Model")
    parser.add_argument("--config", type=str, default="configs/basketgnn.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(config_path=args.config)
