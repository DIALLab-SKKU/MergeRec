import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(kw_only=True)
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""

    enable: bool = field(default=False, metadata={"help": "Enable LoRA modifications"})
    r: int = field(default=16, metadata={"help": "Rank for low-rank decomposition"})
    alpha: int = field(default=32, metadata={"help": "Scaling factor for LoRA updates"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate applied to LoRA layers"})
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"], metadata={"help": "Modules to apply LoRA"}
    )


@dataclass(kw_only=True)
class BaseConfig:
    """Base configuration for model setup and training parameters."""

    model_type: str = field(metadata={"help": "Model type identifier (uppercase enforced)"})
    pooling_method: str = field(
        default="cls", metadata={"help": "Pooling method for model output; e.g., 'cls', 'mean'"}
    )
    model_path: str | None = field(default=None, metadata={"help": "Path or identifier for the pretrained model"})
    tokenizer_path: str | None = field(default=None, metadata={"help": "Path or identifier for the tokenizer"})
    max_seq_len: int = field(default=512, metadata={"help": "Maximum sequence length"})
    max_attribute_len: int = field(default=32, metadata={"help": "Maximum attribute sequence length"})
    max_items: int = field(default=50, metadata={"help": "Maximum number of items per instance"})
    batch_size: int = field(default=32, metadata={"help": "Batch size for training/evaluation"})
    similarity: Literal["cosine", "dot"] = field(default="cosine", metadata={"help": "Similarity function to use"})
    sequence_prompt: str | None = field(default=None, metadata={"help": "Optional prompt prepended to each sequence"})
    item_prompt: str | None = field(default=None, metadata={"help": "Optional prompt prepended to each item"})
    reverse_sequence: bool = field(default=True, metadata={"help": "Reverse the sequence order if True"})
    lora: LoRAConfig = field(default_factory=LoRAConfig, metadata={"help": "LoRA (Low-Rank Adaptation) configuration"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    precision: str = field(default="bf16-mixed", metadata={"help": "Precision mode, e.g., 'bf16-mixed'"})
    num_workers: int = field(default=0, metadata={"help": "Number of DataLoader worker processes"})
    metric_names: list[str] = field(
        default_factory=lambda: ["NDCG", "RECALL"], metadata={"help": "List of evaluation metrics"}
    )
    ks: list[int] = field(
        default_factory=lambda: [1, 5, 10, 50], metadata={"help": "Cutoff values for ranking metrics"}
    )
    model_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional arguments for model initialization"},
    )
    tokenizer_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional arguments for tokenizer initialization"},
    )

    def __post_init__(self):
        self.model_type = self.model_type.upper()
        self.metric_names = [m.upper() for m in self.metric_names]
        if self.num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass(kw_only=True)
class BaseMergeConfig:
    """Configuration for merging multiple datasets or model outputs."""

    data_paths: list[Path] = field(metadata={"help": "List of input data file paths"})
    test_data_paths: list[Path] = field(
        default_factory=list, metadata={"help": "List of test data file paths (optional, can be empty)"}
    )
    finetune_checkpoint_paths: list[Path] = field(metadata={"help": "List of fine-tuned model checkpoint paths"})
    train_data_split: Literal["train", "val", "test", "item"] = field(
        metadata={"help": "Data split to use for merging training: 'train', 'val', or 'test'"}
    )
    test_data_split: Literal["val", "test"] = field(
        metadata={"help": "Data split to use for merging testing: 'val' or 'test'"}
    )
    data_split: Literal["val", "test"] | None = field(
        default=None, metadata={"help": "Data split to use: 'val' or 'test'"}
    )
    merge_type: str = field(metadata={"help": "Merge strategy identifier"})
    learn_type: str = field(metadata={"help": "Learning strategy identifier"})
    ties_density: float = field(
        default=0.2, metadata={"help": "Density threshold for tie-breaking in merge operations"}
    )
    use_softmax: bool = field(default=False, metadata={"help": "Whether to apply softmax to scores"})

    def __post_init__(self):
        if self.data_split is not None:
            if self.test_data_split != self.data_split:
                warnings.warn(
                    "data_split is set but does not match test_data_split. " "Using test_data_split for merging.",
                    UserWarning,
                )
            else:
                self.test_data_split = self.data_split
                warnings.warn(
                    "data_split is deprecated and will be removed in future versions. " "Use test_data_split instead.",
                    DeprecationWarning,
                )

        self.merge_type = self.merge_type.upper()
        self.learn_type = self.learn_type.upper()

        if len(self.test_data_paths) == 0:
            self.test_data_paths = self.data_paths
