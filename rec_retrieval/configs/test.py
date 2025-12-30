from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .base import BaseConfig, BaseMergeConfig


@dataclass(kw_only=True)
class TestConfig(BaseConfig):
    """Configuration for testing the model."""

    metrics_path: Path | None = field(default=None, metadata={"help": "Path to save test metrics"})
    predictions_path: Path | None = field(default=None, metadata={"help": "Path to save model predictions"})
    item_embeddings_path: Path | None = field(default=None, metadata={"help": "Path to save item embeddings"})
    user_embeddings_path: Path | None = field(default=None, metadata={"help": "Path to save user embeddings"})

    def __post_init__(self):
        super().__post_init__()


@dataclass(kw_only=True)
class TestSingleConfig(TestConfig):
    """Configuration for testing a single model."""

    data_path: Path = field(metadata={"help": "Path to the input data file"})
    finetune_checkpoint_path: Path = field(metadata={"help": "Path to the fine-tuned model checkpoint"})
    data_split: Literal["val", "test"] = field(metadata={"help": "Data split to use: 'val' or 'test'"})

    def __post_init__(self):
        super().__post_init__()


@dataclass(kw_only=True)
class TestMergeConfig(TestConfig, BaseMergeConfig):
    """Configuration for testing merged model outputs."""

    weight_file: Path | None = field(default=None, metadata={"help": "Path to merge weight file"})
    weight_file_line: int | float | None = field(default=None, metadata={"help": "Line number to read from weight file"})

    def __post_init__(self):
        # Call both parent class methods explicitly to ensure both are initialized properly
        TestConfig.__post_init__(self)
        BaseMergeConfig.__post_init__(self)
