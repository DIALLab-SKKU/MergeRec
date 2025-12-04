from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .base import BaseConfig, BaseMergeConfig


@dataclass(kw_only=True)
class DistillConfig(BaseMergeConfig, BaseConfig):
    """Configuration for model distillation and merging."""

    temperature: float = field(default=0.05, metadata={"help": "Temperature for distillation"})
    coefficient: float = field(default=1.0, metadata={"help": "Coefficient for distillation loss"})
    loss_fn_kwargs: dict = field(default_factory=dict, metadata={"help": "Additional arguments for the loss function"})
    patience: int = field(default=5, metadata={"help": "Patience for early stopping"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate for distillation"})
    max_epochs: int | None = field(default=None, metadata={"help": "Maximum epochs for distillation"})
    max_steps: int | None = field(default=None, metadata={"help": "Maximum steps for distillation"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})

    loss_type: str = field(default="kd", metadata={"help": "Type of distillation loss to use"})
    valid_ratio: float | None = field(default=None, metadata={"help": "Ratio of validation split"})
    initial_per_weight: float = field(default=0.2, metadata={"help": "Initial weight for per-item loss"})
    metrics_path: Path | None = field(default=None, metadata={"help": "Path to save distillation metrics"})
    predictions_path: Path | None = field(default=None, metadata={"help": "Path to save distillation predictions"})

    def __post_init__(self):
        # Call both parent class methods explicitly to ensure both are initialized properly
        BaseMergeConfig.__post_init__(self)
        BaseConfig.__post_init__(self)
        self.loss_type = self.loss_type.upper()
        self.loss_fn_kwargs = {"coefficient": self.coefficient} | self.loss_fn_kwargs


@dataclass(kw_only=True)
class DistillItemConfig(DistillConfig):
    item_per_dataset: int | None = field(default=None, metadata={"help": "Number of items to sample per dataset"})
    item_sample_method: Literal["random", "centroid"] = field(
        default="random", metadata={"help": "Item sampling strategy"}
    )

    def __post_init__(self):
        # Call both parent class methods explicitly to ensure both are initialized properly
        DistillConfig.__post_init__(self)
        BaseConfig.__post_init__(self)


@dataclass(kw_only=True)
class DistillSequenceConfig(DistillConfig):
    num_sequences_per_dataset: int | None = field(
        default=None, metadata={"help": "Number of sequences to sample per dataset"}
    )
    item_embeddings_paths: list[Path] = field(
        default_factory=list, metadata={"help": "Paths to item embeddings for sequences"}
    )
    sequence_embeddings_paths: list[Path] = field(
        default_factory=list, metadata={"help": "Paths to sequence embeddings for sequences"}
    )
    sample_method: Literal["random", "centroid", "popular"] = field(
        default="random", metadata={"help": "Sampling method for sequences"}
    )

    def __post_init__(self):
        # Call both parent class methods explicitly to ensure both are initialized properly
        DistillConfig.__post_init__(self)
        BaseConfig.__post_init__(self)
