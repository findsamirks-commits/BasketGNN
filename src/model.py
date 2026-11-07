"""LightGCN Model and Basket-Aware Dual Prediction Head.

Implements:
1. LightGCN (He et al., SIGIR 2020): Pure neighbor aggregation over bipartite graph.
2. BasketAwareHead: Serves both next-basket prediction (user query) and basket completion (user + cart query).
3. BPRLoss: Bayesian Personalized Ranking loss with L2 regularization.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph import edge_dropout


class LightGCN(nn.Module):
    """Light Graph Convolutional Network (He et al., SIGIR 2020).

    Discards feature transformations and non-linearities, keeping only linear
    neighborhood aggregations on the bipartite user-item graph.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        layer_weights: Optional[List[float]] = None,
        edge_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_dropout_rate = edge_dropout_rate

        # Layer combination weights (default: uniform 1 / (K + 1))
        if layer_weights is None:
            self.alpha = [1.0 / (num_layers + 1)] * (num_layers + 1)
        else:
            self.alpha = layer_weights

        # Learnable initial 0-th layer representations
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initializes embeddings with truncated normal distribution (std=0.01)."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def get_ego_embeddings(self) -> torch.Tensor:
        """Returns stacked initial embeddings [N_u + N_i, embedding_dim]."""
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(
        self, sparse_adj: torch.Tensor, apply_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs K-layer graph propagation.

        Args:
            sparse_adj: Symmetric normalized adjacency tensor [N_u + N_i, N_u + N_i]
            apply_dropout: Whether to apply edge dropout (typically True during training)

        Returns:
            all_users: Final multi-hop user representations [N_u, d]
            all_items: Final multi-hop item representations [N_i, d]
        """
        adj = edge_dropout(sparse_adj, self.edge_dropout_rate) if apply_dropout else sparse_adj

        all_emb = self.get_ego_embeddings()
        embeddings_list = [all_emb]

        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embeddings_list.append(all_emb)

        # Multi-layer weighted sum
        final_embeddings = torch.zeros_like(all_emb)
        for k in range(self.num_layers + 1):
            final_embeddings = final_embeddings + self.alpha[k] * embeddings_list[k]

        all_users, all_items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return all_users, all_items


class BasketAwareHead(nn.Module):
    """Dual-mode prediction head supporting user-only and basket-completion queries."""

    def __init__(self, basket_alpha: float = 0.45):
        super().__init__()
        self.basket_alpha = basket_alpha

    def predict_next_basket(
        self, user_emb: torch.Tensor, all_item_embs: torch.Tensor
    ) -> torch.Tensor:
        """User-only query (Homepage / App Push / Next Order).

        score(u, i) = user_emb * item_emb^T
        """
        # [batch_size, num_items]
        return torch.matmul(user_emb, all_item_embs.t())

    def predict_basket_completion(
        self,
        user_emb: torch.Tensor,
        basket_item_embs: torch.Tensor,
        all_item_embs: torch.Tensor,
        basket_item_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Basket-completion query (Checkout cross-sell / Attachment lift).

        query = user_emb + alpha * mean(basket_item_embs)
        score(query, i) = query * item_emb^T
        """
        if basket_item_embs.size(0) > 0:
            cart_context = torch.mean(basket_item_embs, dim=0, keepdim=True)
            query_emb = user_emb + (self.basket_alpha * cart_context)
        else:
            query_emb = user_emb

        scores = torch.matmul(query_emb, all_item_embs.t()).squeeze(0)

        # Mask items already present in the user's active cart with large negative score
        if basket_item_ids:
            scores = scores.clone()
            scores[basket_item_ids] = -1e9

        return scores


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss with L2 Regularization."""

    def __init__(self, reg_lambda: float = 1e-4):
        super().__init__()
        self.reg_lambda = reg_lambda

    def forward(
        self,
        user_embs: torch.Tensor,
        pos_item_embs: torch.Tensor,
        neg_item_embs: torch.Tensor,
        u_ego: torch.Tensor,
        pos_ego: torch.Tensor,
        neg_ego: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes BPR loss over triplets (u, pos_i, neg_j).

        L = -sum log(sigmoid(x_u,pos - x_u,neg)) + lambda * ||theta_0||^2
        """
        pos_scores = torch.sum(user_embs * pos_item_embs, dim=-1)
        neg_scores = torch.sum(user_embs * neg_item_embs, dim=-1)

        diff = pos_scores - neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(diff))

        # L2 regularize initial 0-th layer embeddings
        reg_loss = (
            torch.norm(u_ego, p=2).pow(2)
            + torch.norm(pos_ego, p=2).pow(2)
            + torch.norm(neg_ego, p=2).pow(2)
        ) / (2.0 * user_embs.size(0))

        total_loss = bpr_loss + (self.reg_lambda * reg_loss)
        return total_loss, bpr_loss, reg_loss
