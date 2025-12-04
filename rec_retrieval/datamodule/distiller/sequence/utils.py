from collections import Counter

import torch
from torch.utils.data import Dataset, Subset

from ....types import *

__all__ = [
    "sample_popular",
    "split_sequences",
]


def sample_popular(
    test_sequence: list[Sequence],
    num_sequences: int,
):
    counter = Counter()

    for sequence in test_sequence:
        counter.update(sequence)

    # Get the most common items
    most_common_items = counter.most_common(num_sequences)

    # Create a list of the most common items
    popular_items = [item for item, _ in most_common_items]

    return popular_items


def split_sequences(
    score_dataset: Dataset,
    valid_ratio: float | None = None,
):
    if valid_ratio is None:
        score_train_dataset = score_dataset
        score_valid_dataset = None
    else:
        indices = torch.randperm(len(score_dataset))
        train_indices = indices[: int(len(score_dataset) * (1 - valid_ratio))]
        valid_indices = indices[int(len(score_dataset) * (1 - valid_ratio)) :]
        score_train_dataset = Subset(score_dataset, train_indices)
        score_valid_dataset = Subset(score_dataset, valid_indices)

    return score_train_dataset, score_valid_dataset
