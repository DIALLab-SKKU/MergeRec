from typing import Optional, List, Union, Sequence

import torch

from .algorithms import *
from .types import *
from .utils.model_operations import *


class ModelMerger:
    def __init__(
        self,
        models: Sequence[StateDict],
        base_model: Optional[StateDict] = None,
        align_key_order: bool = True,
    ):
        """
        Initialize the ModelMerger for merging multiple models.

        Args:
            models (Sequence[StateDict]): List of model state_dicts to be merged.
            base_model (Optional[StateDict]): The base model's state_dict if needed.
            align_key_order (bool): Whether to align the key order of the models. If False, only checked.
        """
        # Check model shape and store flattened models
        check_model_shape(models, base_model)
        if align_key_order:
            *models, base_model = align_dict_key_order(*models, base_model)
        else:
            assert self._keys_are_aligned(models, base_model), "Model keys are not aligned."

        self.models: List[FlattenedModel] = []
        if base_model is None:
            self.base_model = None

            model_flattened, self.shape_dict = flatten_model(models[0])
            self.models.append(model_flattened)
            models = models[1:]
        else:
            self.base_model, self.shape_dict = flatten_model(base_model)

        for model in models:
            model_flattened, _ = flatten_model(model)
            self.models.append(model_flattened)

    @torch.no_grad()
    def merge(self, merge_type: str, weights: Union[Sequence[float], float], **kwargs) -> StateDict:
        """
        Run the specified merging algorithm on the provided models and return the merged model.

        Args:
            merge_type (str):
                The merging algorithm to use.
            weights (Union[Sequence[float], float]):
                The weights to use for merging. If a single weight is provided, it is used for all models.

        Raises:
            ValueError: If the provided merge_type is not supported.
        """

        if isinstance(weights, float):
            weights = [weights] * len(self.models)
        elif not (isinstance(weights, list) and all(isinstance(w, float) for w in weights)):
            raise ValueError("Weights should be a float or a list of floats.")

        merge_fn_kwargs = dict(models=self.models, base_model=self.base_model, weights=weights, **kwargs)

        # Merge
        if merge_type == "linear":
            merged_model_flattened = merge_linear(**merge_fn_kwargs)
        elif merge_type == "task_vector":
            if self.base_model is None:
                raise ValueError("Task vector merge requires a base model.")
            merged_model_flattened = merge_task_vector(**merge_fn_kwargs)
        elif merge_type == "ties":
            if self.base_model is None:
                raise ValueError("TIES merge requires a base model.")
            merged_model_flattened = merge_ties(**merge_fn_kwargs)
        elif merge_type == "dare":
            if self.base_model is None:
                raise ValueError("DARE merge requires a base model.")
            merged_model_flattened = merge_dare(**merge_fn_kwargs)
        elif merge_type == "pcb":
            if self.base_model is None:
                raise ValueError("PCB merge requires a base model.")
            merged_model_flattened = merge_pcb(**merge_fn_kwargs)
        else:
            raise ValueError(f"Merge type '{merge_type}' is not supported.")

        # Unflatten
        merged_model = unflatten_model(merged_model_flattened, self.shape_dict)

        return merged_model

    @staticmethod
    def _keys_are_aligned(models: Sequence[StateDict], base_model: Optional[StateDict]):
        reference = models[0]
        compared = list(models[1:])

        if base_model is not None:
            compared.append(base_model)

        for model in compared:
            if list(model.keys()) != list(reference.keys()):
                return False

        return True
