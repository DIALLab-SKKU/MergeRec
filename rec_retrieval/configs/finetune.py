from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseConfig
from ..types import NegativeSampleOption


@dataclass(kw_only=True)
class NegativeSampleConfig:
    """Configuration for negative sampling strategy."""

    k: int | None = field(default=None, metadata={"help": "Number of negative samples; None for exhaustive"})
    in_batch: bool = field(default=False, metadata={"help": "Use in-batch negatives if True"})
    mode: NegativeSampleOption = field(init=False, metadata={"help": "Determined negative sampling mode"})

    def __post_init__(self):
        if self.k is None and not self.in_batch:
            self.mode = NegativeSampleOption.FULL
        elif self.k is not None and not self.in_batch:
            self.mode = NegativeSampleOption.SAMPLE
        elif self.k is None and self.in_batch:
            self.mode = NegativeSampleOption.IN_BATCH
        else:
            self.mode = NegativeSampleOption.IN_BATCH_SAMPLE


@dataclass(kw_only=True)
class FinetuneConfig(BaseConfig):
    """Configuration for fine-tuning the model."""

    temperature: float = field(default=0.05, metadata={"help": "Softmax temperature for training loss"})
    patience: int = field(default=5, metadata={"help": "Epochs with no improvement before early stopping"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Optimizer learning rate"})
    max_epochs: int = field(default=100, metadata={"help": "Maximum number of training epochs"})
    warmup_steps: int | float = field(default=100, metadata={"help": "Number or fraction of warmup steps"})
    weight_decay: float = field(default=0, metadata={"help": "Weight decay coefficient"})
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Steps to accumulate gradients before update"}
    )
    negative_sample: NegativeSampleConfig = field(
        default_factory=NegativeSampleConfig, metadata={"help": "Negative sampling configuration"}
    )
    gradient_clip_val: float | None = field(
        default=None, metadata={"help": "Max norm for gradient clipping; None to disable"}
    )
    log_every_n_steps: int = field(default=50, metadata={"help": "Logging frequency in steps"})
    valid_metric: str = field(default="val/NDCG@10", metadata={"help": "Validation metric to monitor"})

    def __post_init__(self):
        super().__post_init__()


@dataclass(kw_only=True)
class FinetuneSingleConfig(FinetuneConfig):
    data_path: Path = field(metadata={"help": "Path to training dataset"})

    def __post_init__(self):
        super().__post_init__()


@dataclass(kw_only=True)
class FinetuneJointConfig(FinetuneConfig):
    data_paths: list[Path] = field(
        default_factory=list,
        metadata={"help": "List of paths to training datasets for joint training"},
    )

    def __post_init__(self):
        if self.negative_sample is None:
            raise ValueError("Only negative sampling is supported for joint training.")

        super().__post_init__()
