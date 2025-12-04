from typing import List

import torch

from ..types import FlattenedModel, FlattenedModel2D


def get_localize_and_stitch_vectors(
    base_model: FlattenedModel, models: List[FlattenedModel], density: float = 0.05, **__
) -> FlattenedModel2D:
    """
    Dataless Localize-and-Stitch (top-k%) masked task vectors.
    - Localization: for each task vector τ_i = θ_ft^(i) - θ_pre, take top-k% by |τ_i|.
    - Stitching prep: build processed masks γ'_i where each active index is scaled by
      1 / (#tasks active at that index), then return γ'_i ⊙ τ_i for all i.

    Args:
        base_model: Flattened pretrained weights θ_pre (shape: (d,))
        models: List of flattened fine-tuned weights θ_ft^(i)
        density: k% (0~1) used for top-k selection; paper often uses 0.05 in dataless

    Returns:
        Tensor of shape (N, d): per-task masked updates (γ'_i ⊙ τ_i)
    """
    assert len(models) > 0, "models must be non-empty."
    # Task vectors τ_i
    updates = torch.stack([m - base_model for m in models], dim=0)  # (N, d)
    N, d = updates.shape

    # Handle k (top-k elements by magnitude) for dataless localization
    k = int(density * d)
    if k <= 0:
        # No positions selected -> all-zero masked vectors
        return torch.zeros_like(updates)

    # Binary masks γ_i: top-k% by |τ_i|
    abs_updates = updates.abs()
    topk_idx = torch.topk(abs_updates, k, dim=1, largest=True).indices  # (N, k)
    masks = torch.zeros_like(updates)
    masks.scatter_(1, topk_idx, 1.0)  # set selected indices to 1

    # Overlap resolution (stitching weight): γ'_i[k] = γ_i[k] / sum_j γ_j[k]
    overlap_counts = masks.sum(dim=0)  # (d,)
    denom = overlap_counts.clamp(min=1.0)  # avoid division by zero
    processed_masks = masks / denom  # broadcast (N, d) / (d,)

    # Return γ'_i ⊙ τ_i
    masked_updates = processed_masks * updates  # (N, d)
    return masked_updates


def merge_localize_and_stitch(
    base_model: FlattenedModel,
    models: List[FlattenedModel],
    weights: List[float],
    density: float = 0.05,
    **__,
) -> FlattenedModel:
    """
    Merge according to Localize-and-Stitch:
      θ_merged = θ_pre + sum_i ( w_i * (γ'_i ⊙ τ_i) )
    Note: 논문은 별도 scaling α 없이 합산을 사용하지만, 인터페이스 호환을 위해 weights를 허용합니다.

    Args:
        base_model: θ_pre
        models: list of θ_ft^(i)
        weights: per-task weights (optional scaling; set all 1.0 to mimic paper)
        density: top-k% for dataless localization

    Returns:
        θ_merged (FlattenedModel)
    """
    assert len(models) == len(weights), "Number of models and weights should match."

    vectors = get_localize_and_stitch_vectors(base_model, models, density=density)  # (N, d)
    if len(weights):
        w = torch.as_tensor(weights, dtype=vectors.dtype, device=vectors.device).view(-1, 1)  # (N, 1)
        vectors = vectors * w

    delta = vectors.sum(dim=0)
    return base_model + delta
