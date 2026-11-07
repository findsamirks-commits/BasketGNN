"""Graph Construction & Symmetric-Normalized Adjacency for LightGCN.

Builds the bipartite user-item interaction graph and converts it to a
sparse symmetric normalized adjacency matrix: A_tilde = D^(-1/2) * A * D^(-1/2)
Implemented in pure PyTorch sparse tensors (no PyG / C++ compile dependencies).
"""

from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def build_bipartite_adjacency(
    num_users: int,
    num_items: int,
    user_item_edges: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Constructs symmetric-normalized bipartite adjacency tensor for LightGCN.

    A = [[0, R], [R^T, 0]] of dimension (N_u + N_i) x (N_u + N_i)
    A_tilde = D^(-1/2) * A * D^(-1/2)

    Args:
        num_users: Number of unique users (N_u)
        num_items: Number of unique items (N_i)
        user_item_edges: LongTensor of shape [2, E] where row 0 is user_id, row 1 is item_id
        device: Target torch device

    Returns:
        torch.sparse_coo_tensor of shape (N_u + N_i, N_u + N_i)
    """
    total_nodes = num_users + num_items
    u_idx = user_item_edges[0]
    i_idx = user_item_edges[1] + num_users  # Shift item indices by N_u

    # Undirected edges: (u -> i) and (i -> u)
    row = torch.cat([u_idx, i_idx], dim=0)
    col = torch.cat([i_idx, u_idx], dim=0)
    indices = torch.stack([row, col], dim=0)

    # Convert to scipy coo for robust degree calculation
    row_np = row.cpu().numpy()
    col_np = col.cpu().numpy()
    data_np = np.ones(len(row_np), dtype=np.float32)

    adj_mat = sp.coo_matrix((data_np, (row_np, col_np)), shape=(total_nodes, total_nodes))

    # Degree normalization: D^(-1/2)
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    pos_mask = degrees > 0.0
    d_inv_sqrt[pos_mask] = np.power(degrees[pos_mask], -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()

    # Convert back to PyTorch sparse COO tensor
    norm_indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
    norm_values = torch.from_numpy(norm_adj.data.astype(np.float32))

    sparse_tensor = torch.sparse_coo_tensor(
        norm_indices, norm_values, torch.Size([total_nodes, total_nodes]), device=device
    )
    return sparse_tensor.coalesce()


def edge_dropout(sparse_adj: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
    """Randomly drops edges from sparse adjacency matrix for regularization."""
    if drop_rate <= 0.0:
        return sparse_adj

    indices = sparse_adj.indices()
    values = sparse_adj.values()
    num_edges = values.size(0)

    mask = torch.rand(num_edges, device=values.device) >= drop_rate
    dropped_indices = indices[:, mask]
    # Scale values to maintain expectation
    dropped_values = values[mask] / (1.0 - drop_rate)

    dropped_adj = torch.sparse_coo_tensor(
        dropped_indices, dropped_values, sparse_adj.shape, device=sparse_adj.device
    )
    return dropped_adj.coalesce()
