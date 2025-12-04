from typing import List

from torch.nn.functional import dropout

from ..types import FlattenedModel


def merge_dare(
    base_model: FlattenedModel, models: List[FlattenedModel], weights: List[float], density: float, **__
) -> FlattenedModel:
    """
    DARE merging for multiple models.

    Args:
        base_model (FlattenedModel): Base model parameters.
        models (List[FlattenedModel]): List of flattened model parameters.
        weights (List[float]): Weights per model.
        density (float): Fraction of elements to keep in each model.

    Returns:
        FlattenedModel: The merged model parameters.
    """
    assert len(models) == len(weights), "Number of models and weights should match."

    merged = base_model.clone()
    for i, m in enumerate(models):
        update = weights[i] * (m - base_model)

        update = dropout(update, p=density, training=True)
        merged += update

    return merged
