from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from torch import nn

T = TypeVar("T", bound=nn.Module)


class TaskVectorMergingModuleBase[T](nn.Module, ABC):
    def __init__(
        self,
        base_model_tensor: torch.Tensor,  # (num_model_parameters,)
        task_vectors_tensor: torch.Tensor,  # (num_task_vectors, num_model_parameters)
        model_without_params: T,
        shape_dict: dict[str, torch.Size],  # Sorted dict of parameter names to shapes
        disable_softmax: bool = False,
    ):
        super().__init__()

        self.model = model_without_params
        self.shape_dict = shape_dict
        self.disable_softmax = disable_softmax

        # Frozen base/task_vectors
        self.base_model_tensor = nn.Parameter(base_model_tensor, requires_grad=False)  # requires_grad=False
        self.task_vectors_tensor = nn.Parameter(task_vectors_tensor, requires_grad=False)  # requires_grad=False

        self.global_weights = nn.ParameterDict()
        self.global_biases = nn.ParameterDict()
        self.per_weights = nn.ParameterDict()

    def trainable_parameters(
        self, freeze_global_weight: bool = False, freeze_global_bias: bool = False, freeze_per_weight: bool = False
    ):
        params = []
        if not freeze_global_weight:
            params.extend(self.global_weights.parameters())
        if not freeze_global_bias:
            params.extend(self.global_biases.parameters())
        if not freeze_per_weight:
            params.extend(self.per_weights.parameters())

        return params

    def serialize_weights(self):
        return {
            "global_weights": {k: v.tolist() for k, v in self.global_weights.items()},
            "global_biases": {k: v.tolist() for k, v in self.global_biases.items()},
            "per_weights": {k: v.tolist() for k, v in self.per_weights.items()},
        }

    @torch.no_grad()
    def load_weights_from_dict(self, weights: dict[str, dict[str, list[float]]]):
        for k, v in weights["global_weights"].items():
            assert k in self.global_weights, f"Key '{k}' not found in global_weights."
            v = torch.tensor(v)
            assert (
                v.shape == self.global_weights[k].shape
            ), f"Shape mismatch for key '{k}', ({v.shape} != {self.global_weights[k].shape})"
            self.global_weights[k].data = v
        for k, v in weights["global_biases"].items():
            assert k in self.global_biases, f"Key '{k}' not found in global_biases."
            v = torch.tensor(v)
            assert (
                v.shape == self.global_biases[k].shape
            ), f"Shape mismatch for key '{k}', ({v.shape} != {self.global_biases[k].shape})"
            self.global_biases[k].data = v
        for k, v in weights["per_weights"].items():
            assert k in self.per_weights, f"Key '{k}' not found in per_weights."
            v = torch.tensor(v)
            v = v[: self.per_weights[k].numel()]  # Ensure correct size
            assert (
                v.shape == self.per_weights[k].shape
            ), f"Shape mismatch for key '{k}', ({v.shape} != {self.per_weights[k].shape})"
            self.per_weights[k].data = v

    def forward(self, batch) -> T:
        self.load_weights()

        return self.model(batch)

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self):
        raise NotImplementedError
