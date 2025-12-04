from typing import TypeVar

import torch
from torch import nn

from ._base import TaskVectorMergingModuleBase
from ..utils import load_weights, get_state_dict

T = TypeVar("T", bound=nn.Module)


class TaskVectorMergingModuleTaskWise[T](TaskVectorMergingModuleBase):
    def __init__(
        self,
        base_model_tensor: torch.Tensor,  # (num_model_parameters,)
        task_vectors_tensor: torch.Tensor,  # (num_task_vectors, num_model_parameters)
        model_without_params: T,
        shape_dict: dict[str, torch.Size],  # Sorted dict of parameter names to shapes
        initial_global_weight: float = 1.0,
        initial_global_bias: float = 0.0,
        initial_per_weight: float = 0.2,
        disable_softmax: bool = True,
    ):
        super().__init__(
            base_model_tensor,
            task_vectors_tensor,
            model_without_params,
            shape_dict,
            disable_softmax,
        )

        self.global_weights["all"] = nn.Parameter(torch.full((1,), initial_global_weight))
        self.global_biases["all"] = nn.Parameter(torch.full((1,), initial_global_bias))
        self.per_weights["all"] = nn.Parameter(torch.full((task_vectors_tensor.size(0),), initial_per_weight))

    def _merge_task_vectors(self):
        per_weights = self.per_weights["all"]  # (num_task_vectors,)

        if not self.disable_softmax:
            per_weights = torch.softmax(per_weights, dim=0)

        weights = self.global_weights["all"] * per_weights + self.global_biases["all"]  # (num_task_vectors,)
        task_vectors_stacked = (
            weights.unsqueeze(1) * self.task_vectors_tensor
        )  # (num_task_vectors, num_model_parameters)

        merged_params = self.base_model_tensor + task_vectors_stacked.sum(dim=0)  # (num_model_parameters,)
        return merged_params

    def load_weights(self):
        merged_params = self._merge_task_vectors()

        load_weights(self.model, merged_params, self.shape_dict)

        return self.model

    def get_state_dict(self):
        merged_params = self._merge_task_vectors()

        state_dict = get_state_dict(merged_params, self.shape_dict)

        return state_dict
