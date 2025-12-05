from enum import Enum


__all__ = [
    "MergeType",
    "LearnType",
    "LossType",
]


class MergeType(Enum):
    """
    Enum for different merge types.
    """

    TASK_VECTOR = "TASK_VECTOR"
    TIES = "TIES"
    PCB = "PCB"
    LOCALIZE_AND_STITCH = "LOCALIZE_AND_STITCH"


class LearnType(Enum):
    """
    Enum for different learning types.
    """

    TASK_WISE = "TASK_WISE"
    LAYER_WISE = "LAYER_WISE"


class LossType(Enum):
    CE = "CE"
    KD = "KD"
    MSE = "MSE"
    ADAMERGING = "ADAMERGING"
    ADAMERGING_KD = "ADAMERGING_KD"
    MERGED_PSEUDO_LABEL = "MERGED_PSEUDO_LABEL"
    SINGLE_PSEUDO_LABEL = "SINGLE_PSEUDO_LABEL"
    MERGED_PSEUDO_LABEL_KD = "MERGED_PSEUDO_LABEL_KD"
    SINGLE_PSEUDO_LABEL_KD = "SINGLE_PSEUDO_LABEL_KD"
