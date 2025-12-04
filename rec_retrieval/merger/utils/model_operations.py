from typing import List, Optional, Tuple, Sequence, Iterable, Union

import torch

from ..types import *

__all__ = [
    "check_model_shape",
    "flatten_model",
    "unflatten_model",
    "align_dict_key_order",
]


def check_model_shape(models: Sequence[StateDict], base_model: Optional[StateDict] = None) -> None:
    """
    Checks whether all models (and optionally the base model) have identical keys and tensor shapes.

    Args:
        models (List[StateDict]): A list of model state_dicts.
        base_model (Optional[StateDict]): The base model's state_dict if needed.

    Raises:
        AssertionError: If any mismatch in keys or shapes is found.
    """
    # Reference keys/shapes from the first model
    ref_keys = set(models[0].keys())

    # Check each model
    for m in models:
        assert set(m.keys()) == ref_keys, "Models have different architectures."

    # Check shapes across all models
    for param_name in ref_keys:
        ref_shape = models[0][param_name].shape
        for m in models[1:]:
            assert m[param_name].shape == ref_shape, "Models have different shapes."

    # If base_model is present, check that too
    if base_model is not None:
        base_keys = set(base_model.keys())
        assert base_keys == ref_keys, "Base model has different architecture from the others."
        for param_name in ref_keys:
            assert base_model[param_name].shape == models[0][param_name].shape, "Base model has different shapes."


def flatten_model(model: StateDict) -> Tuple[FlattenedModel, ShapeDict]:
    """
    Flattens a model's parameters into a single tensor and returns a shape dictionary
    to restore them later.

    Args:
        model (StateDict): Model parameter dictionary.

    Returns:
        (flattened_params, shape_dict):
            flattened_params (torch.Tensor): 1D tensor of all parameters concatenated.
            shape_dict (ShapeDict): A dict from parameter name to original shape.
    """
    shape_dict = {k: v.shape for k, v in model.items()}
    flattened_list = [v.reshape(-1) for v in model.values()]
    flattened_tensor = torch.cat(flattened_list)
    return flattened_tensor, shape_dict


def unflatten_model(model: FlattenedModel, shape_dict: ShapeDict) -> StateDict:
    """
    Restores the flattened tensor into a StateDict with original shapes.

    Args:
        model (torch.Tensor):
            1D tensor that represents flattened model parameters.
        shape_dict (ShapeDict):
            A dictionary mapping parameter names to their original shapes.

    Returns:
        StateDict: Dictionary of parameters with the original shapes.
    """
    unflattened = {}
    start_idx = 0
    end_idx = 0
    for name, shape in shape_dict.items():
        end_idx = start_idx + shape.numel()
        unflattened[name] = model[start_idx:end_idx].reshape(shape)
        start_idx = end_idx

    # Check that we've used all values
    assert end_idx == model.numel(), "Flattened tensor size does not match the expected size."

    return unflattened


def align_dict_key_order(
    *models: Union[StateDict, None], key_order: Optional[List[str]] = None
) -> Iterable[Union[StateDict, None]]:
    """
    Reorders all models' state dicts such that they share the same sorted key order.

    Args:
        *models (StateDict): Variable number of model state dicts to be reordered.
        key_order (Optional[List[str]]): The order of keys to reorder the state dicts.
                                         If None, uses the sorted keys of the first model.

    Returns:
        Iterable[StateDict]: An iterable of reordered state dict, in the same order as the input models.
    """
    # Infer key order from the first model if not provided
    if key_order is None:
        for model in models:
            if isinstance(model, dict):
                keys = set(model.keys())
                key_order = sorted(model.keys())
                break
        else:
            raise ValueError("At least one model must be provided to infer the key order.")
    else:
        keys = set(key_order)

    reordered_models = []
    for model in models:
        if model is None:
            reordered_models.append(None)
            continue

        assert isinstance(model, dict), f"Model must be a dictionary, got {type(model)}."

        missing_keys = keys - set(model.keys())
        extra_keys = set(model.keys()) - keys

        assert not missing_keys and not extra_keys, (
            f"All models must have the same set of keys. " f"Missing keys: {missing_keys}, Extra keys: {extra_keys}"
        )

        reordered_models.append({k: model[k] for k in key_order})

    return reordered_models
