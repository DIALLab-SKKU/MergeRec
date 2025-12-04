import torch

from ...algorithms.localize_and_stitch import get_localize_and_stitch_vectors
from ...algorithms.pcb import get_pcb_vectors
from ...algorithms.task_vector import get_task_vectors
from ...algorithms.ties import get_ties_vectors
from ...merger import ModelMerger
from ...weight_learning.module._base import TaskVectorMergingModuleBase
from ...weight_learning.module.layer_wise import TaskVectorMergingModuleLayerWise
from ...weight_learning.module.task_wise import TaskVectorMergingModuleTaskWise
from ...weight_learning.utils import make_functional
from ....merger.enums import *
from ....merger.types import *


def _check_isinstance_state_dict(t):
    if not isinstance(t, dict):
        raise ValueError(f"Expected a state dict, got {type(t)}")

    for k, v in t.items():
        if not isinstance(k, str):
            raise ValueError(f"Expected a string key, got {type(k)}")
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Expected a tensor value, got {type(v)}")


def load_merging_module(
    merge_type: MergeType,
    learn_type: LearnType,
    model: torch.nn.Module,
    pretrain_state_dict: StateDict,
    finetune_state_dicts: list[StateDict],
    ignore_keys: set[str],
    ties_density: float | None = None,
    initial_global_weight: float = 1.0,
    initial_global_bias: float = 0.0,
    initial_per_weight: float = 0.2,
    disable_softmax: bool = False,
) -> TaskVectorMergingModuleBase:
    """
    !! Important

    Key orders are important to maintain the same order of the model parameters.

    We use key orders from the pre-trained model and align all fine-tuned models' keys to the same order.
    We reorder shape_dict to match the order of the keys from the pre-trained model.
    """
    assert isinstance(merge_type, MergeType), f"Invalid merge type: {merge_type}"
    assert isinstance(learn_type, LearnType), f"Invalid learn type: {learn_type}"

    _check_isinstance_state_dict(pretrain_state_dict)
    for model_ckpt in finetune_state_dicts:
        _check_isinstance_state_dict(model_ckpt)

    keys_to_keep = set(pretrain_state_dict.keys() & finetune_state_dicts[0].keys()) - ignore_keys

    # Reorder keys to same order
    pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if k in keys_to_keep}
    keys_order = {k: i for i, k in enumerate(pretrain_state_dict.keys())}

    finetune_state_dicts = [
        {k: v for k, v in model_ckpt.items() if k in keys_to_keep} for model_ckpt in finetune_state_dicts
    ]
    finetune_state_dicts = [
        dict(sorted(model_ckpt.items(), key=lambda x: keys_order[x[0]])) for model_ckpt in finetune_state_dicts
    ]

    # Convert model to functional form
    print("Converting model to functional form...")
    make_functional(model)

    # Calculate task vectors
    print("Calculating task vectors...")
    # merger.base_model and merger.models have keys aligned
    merger = ModelMerger(models=finetune_state_dicts, base_model=pretrain_state_dict, align_key_order=False)
    match merge_type:
        case MergeType.TASK_VECTOR:
            task_vectors_tensor = get_task_vectors(
                base_model=merger.base_model,
                models=merger.models,
            )
        case MergeType.TIES:
            assert ties_density is not None, "Density should be provided for ties merging."
            task_vectors_tensor = get_ties_vectors(
                base_model=merger.base_model,
                models=merger.models,
                density=ties_density,
            )
        case MergeType.PCB:
            task_vectors_tensor = get_pcb_vectors(
                base_model=merger.base_model,
                models=merger.models,
                density=ties_density,
            )
        case MergeType.LOCALIZE_AND_STITCH:
            task_vectors_tensor = get_localize_and_stitch_vectors(
                base_model=merger.base_model,
                models=merger.models,
                density=ties_density,
            )
        case _:
            raise ValueError(f"Invalid merge type: {merge_type}")
    base_model_tensor = merger.base_model.detach().requires_grad_(False)  # (num_model_parameters,)
    task_vectors_tensor = task_vectors_tensor.detach().requires_grad_(False)  # (num_task_vectors, num_model_parameters)
    shape_dict = merger.shape_dict  # {k: v.shape for k, v in base_model.items()}

    print("Creating merging module...")
    match learn_type:
        case LearnType.TASK_WISE:
            merging_module_cls = TaskVectorMergingModuleTaskWise
        case LearnType.LAYER_WISE:
            merging_module_cls = TaskVectorMergingModuleLayerWise
        case _:
            raise ValueError(f"Invalid learn type: {learn_type}")

    merging_module = merging_module_cls(
        base_model_tensor,
        task_vectors_tensor,
        model,
        shape_dict=shape_dict,
        initial_global_weight=initial_global_weight,
        initial_global_bias=initial_global_bias,
        initial_per_weight=initial_per_weight,
        disable_softmax=disable_softmax,
    )

    return merging_module
