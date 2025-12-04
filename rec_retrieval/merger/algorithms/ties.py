from typing import List

import torch

from ..types import FlattenedModel, FlattenedModel2D


def _compute_sparse_updates(
    base_model: FlattenedModel, models: List[FlattenedModel], density: float, weights: List[float] = None
):
    """
    Computes sparse updates for each model relative to the base model.
    """
    numel = base_model.numel()
    topk_count = int(density * numel)
    sparse_updates = []

    for i, model in enumerate(models):
        update = model - base_model
        if weights:
            update *= weights[i]

        mask_indices = torch.topk(update.abs(), topk_count, largest=True).indices
        sparse_update = torch.zeros_like(update)
        sparse_update[mask_indices] = update[mask_indices]
        sparse_updates.append(sparse_update)

    return torch.stack(sparse_updates, dim=0)  # shape = (M, numel)


def _compute_final_sign(sparse_updates: torch.Tensor, base_model: FlattenedModel):
    """
    Computes the final sign decision for each parameter index based on sparse updates.
    """
    pos_sum = torch.sum(torch.where(sparse_updates > 0, sparse_updates, torch.zeros_like(sparse_updates)), dim=0)
    neg_sum = torch.sum(torch.where(sparse_updates < 0, sparse_updates, torch.zeros_like(sparse_updates)), dim=0)

    conflict_mask = (pos_sum != 0) & (neg_sum != 0)
    final_sign = torch.zeros_like(base_model)

    final_sign[conflict_mask] = torch.where(
        torch.abs(pos_sum[conflict_mask]) >= torch.abs(neg_sum[conflict_mask]),
        torch.ones_like(pos_sum[conflict_mask]),
        -1 * torch.ones_like(pos_sum[conflict_mask]),
    )

    no_conflict_mask = ~conflict_mask
    final_sign[no_conflict_mask] = torch.sign(pos_sum[no_conflict_mask] + neg_sum[no_conflict_mask])

    final_sign[final_sign == 0] = 1  # Default zero values to +1

    return final_sign


def get_ties_vectors(
    base_model: FlattenedModel, models: List[FlattenedModel], density: float, **__
) -> FlattenedModel2D:
    sparse_updates = _compute_sparse_updates(base_model, models, density)
    final_sign = _compute_final_sign(sparse_updates, base_model)

    selected_entries = torch.where(
        final_sign.unsqueeze(0) > 0,
        torch.where(sparse_updates > 0, sparse_updates, torch.zeros_like(sparse_updates)),
        torch.where(sparse_updates < 0, sparse_updates, torch.zeros_like(sparse_updates)),
    )

    # Disjoint mean
    nonzero_count = torch.count_nonzero(selected_entries, dim=0).unsqueeze(0)
    selected_entries.div_(nonzero_count)
    selected_entries.nan_to_num_(0.0)

    return selected_entries


def merge_ties(
    base_model: FlattenedModel, models: List[FlattenedModel], weights: List[float], density: float, **__
) -> FlattenedModel:
    assert len(models) == len(weights), "Number of models and weights should match."

    sparse_updates = _compute_sparse_updates(base_model, models, density, weights)
    final_delta = torch.sum(sparse_updates, dim=0)

    return base_model + final_delta
