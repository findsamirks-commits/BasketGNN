"""BPR Negative Sampler for Pairwise Ranking Loss.

Constructs training batches of (user, pos_item, neg_item) triplets
where neg_item is uniformly sampled from non-interacted catalog SKUs.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BPRDataset(Dataset):
    """Pairwise BPR training dataset."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_pairs: List[Tuple[int, int]],
        user_pos_map: Dict[int, Set[int]],
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.train_pairs = train_pairs
        self.user_pos_map = user_pos_map

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        u, pos_i = self.train_pairs[idx]
        pos_set = self.user_pos_map.get(u, set())

        # Uniform negative sampling
        neg_j = np.random.randint(0, self.num_items)
        while neg_j in pos_set:
            neg_j = np.random.randint(0, self.num_items)

        return u, pos_i, neg_j


def create_bpr_dataloader(
    num_users: int,
    num_items: int,
    train_pairs: List[Tuple[int, int]],
    user_pos_map: Dict[int, Set[int]],
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Builds a PyTorch DataLoader yielding (users, pos_items, neg_items) batches."""
    dataset = BPRDataset(num_users, num_items, train_pairs, user_pos_map)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: (
            torch.tensor([x[0] for x in batch], dtype=torch.long),
            torch.tensor([x[1] for x in batch], dtype=torch.long),
            torch.tensor([x[2] for x in batch], dtype=torch.long),
        ),
    )
