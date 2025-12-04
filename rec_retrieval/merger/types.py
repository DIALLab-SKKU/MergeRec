from pathlib import Path
from typing import Dict, Union

import torch

__all__ = [
    "PathStr",
    "FlattenedModel",
    "FlattenedModel2D",
    "StateDict",
    "ShapeDict",
]


type PathStr = Union[str, Path]
type FlattenedModel = torch.Tensor  # (d) where d is the number of parameters in the model, flattened into a 1D tensor
type FlattenedModel2D = torch.Tensor  # (N, d) where N is the number of models and d is the number of parameters in each model, flattened into a 2D tensor
type StateDict = Dict[str, torch.Tensor]
type ShapeDict = Dict[str, torch.Size]
