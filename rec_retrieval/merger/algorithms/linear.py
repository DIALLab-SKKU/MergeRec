from typing import List

import torch

from ..types import *


def merge_linear(models: List[FlattenedModel], weights: List[float], **__) -> FlattenedModel:
    """
    Merges multiple models by a simple linear combination:
        merged = sum_i( weights[i] * models[i] ).
    The base_model is not used in linear merging (kept for API compatibility).

    Args:
        models (List[FlattenedModel]): List of flattened model parameters.
        weights (List[float]): Weights used in the linear combination.

    Returns:
        FlattenedModel: The merged model parameters.
    """
    assert len(models) == len(weights), "Number of models and weights should match."

    merged = torch.zeros_like(models[0])
    for i, m in enumerate(models):
        merged += weights[i] * m

    return merged
