from typing import List

import torch

from ..types import FlattenedModel, FlattenedModel2D


def get_task_vectors(base_model: FlattenedModel, models: List[FlattenedModel], **__) -> FlattenedModel2D:
    models = [m - base_model for m in models]
    return torch.stack(models)


def merge_task_vector(
    base_model: FlattenedModel, models: List[FlattenedModel], weights: List[float], **__
) -> FlattenedModel:
    """
    Merges multiple models by adding scaled differences to a base model.
    Conceptually: merged = base_model + sum_i( weights[i] * models[i] ).

    Args:
        base_model (FlattenedModel): Base model parameters.
        models (List[FlattenedModel]): List of flattened model parameters.
        weights (List[float]): Weights for merging.

    Returns:
        FlattenedModel: The merged model parameters.
    """
    assert len(models) == len(weights), "Number of models and weights should match."

    merged = base_model.clone()
    for i, m in enumerate(models):
        merged += weights[i] * (m - base_model)

    return merged
