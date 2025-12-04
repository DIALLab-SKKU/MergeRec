from collections import defaultdict
from typing import TypeVar

import torch
from torch import nn

from ._base import TaskVectorMergingModuleBase
from ..utils import load_weights, get_state_dict

T = TypeVar("T", bound=nn.Module)


def group_parameters_by_layer(shape_dict):
    """
    layer_groups: {layer_key: [(param_name, start_idx, end_idx), ...], ...}
    We treat all parameters under 'longformer.encoder.layer.X.' as belonging to layer X.
    Others (like embeddings or heads) go into separate group keys.
    """
    layer_groups = defaultdict(list)
    offset = 0
    for name, shape in shape_dict.items():
        numel = shape.numel()
        if "encoder.layer." in name:
            # parse layer index from something like 'longformer.encoder.layer.11...'
            parts = name.split(".")
            # e.g. ["longformer", "encoder", "layer", "0", ...]
            layer_idx = parts[3]  # '0', '1', etc.
            layer_groups[layer_idx].append((name, offset, offset + numel))
        else:
            # embeddings / pooler / classifier etc.
            layer_groups["others"].append((name, offset, offset + numel))
        offset += numel
    return layer_groups


# class TaskVectorMergingModuleLayerWise(TaskVectorMergingModuleBase, Generic[T]):
class TaskVectorMergingModuleLayerWise[T](TaskVectorMergingModuleBase):
    def __init__(
        self,
        base_model_tensor: torch.Tensor,  # (num_model_parameters,)
        task_vectors_tensor: torch.Tensor,  # (num_task_vectors, num_model_parameters)
        model_without_params: T,
        shape_dict: dict[str, torch.Size],  # Sorted dict of parameter names to shapes
        initial_global_weight: float = 1.0,
        initial_global_bias: float = 0.0,
        initial_per_weight: float = 0.2,
        disable_softmax: bool = False,
    ):
        super().__init__(
            base_model_tensor,
            task_vectors_tensor,
            model_without_params,
            shape_dict,
            disable_softmax,
        )

        self._num_model_parameters = base_model_tensor.size(0)
        self.layer_groups = group_parameters_by_layer(shape_dict)
        for layer_key in self.layer_groups:
            self.global_weights[layer_key] = nn.Parameter(torch.full((1,), initial_global_weight))
            self.global_biases[layer_key] = nn.Parameter(torch.full((1,), initial_global_bias))
            self.per_weights[layer_key] = nn.Parameter(torch.full((task_vectors_tensor.size(0),), initial_per_weight))

    def _merge_task_vectors(self):
        merged_params = torch.zeros(self._num_model_parameters, device=self.base_model_tensor.device)
        for layer_key in self.layer_groups:
            global_weight = self.global_weights[layer_key]  # (1,)
            global_bias = self.global_biases[layer_key]  # (1,)
            per_weight = self.per_weights[layer_key]  # (num_task_vectors,)

            if not self.disable_softmax:
                per_weight = torch.softmax(per_weight, dim=0)
            weights = global_weight * per_weight + global_bias  # (num_task_vectors,)

            for param_name, start, end in self.layer_groups[layer_key]:
                base_chunk = self.base_model_tensor[start:end]  # (num_module_parameters,)
                task_chunk = self.task_vectors_tensor[:, start:end]  # (num_task_vectors, num_module_parameters)

                task_vectors_stacked = weights.unsqueeze(1) * task_chunk  # (num_task_vectors, num_module_parameters)
                merged_chunk = base_chunk + task_vectors_stacked.sum(dim=0)  # (num_module_parameters,)

                merged_params[start:end] = merged_chunk
        return merged_params

    def load_weights(self):
        merged_params = self._merge_task_vectors()

        load_weights(self.model, merged_params, self.shape_dict)

        return self.model

    def get_state_dict(self):
        merged_params = self._merge_task_vectors()

        state_dict = get_state_dict(merged_params, self.shape_dict)

        return state_dict
