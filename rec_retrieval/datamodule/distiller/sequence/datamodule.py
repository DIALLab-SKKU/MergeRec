from pathlib import Path

import lightning as L
import torch
from torch.utils.data import Subset, DataLoader
from transformers import PreTrainedTokenizer

from .utils import split_sequences, sample_popular
from ..item.utils import sample_centroid
from ...collator.distiller import DistillSequenceCollator
from ...collator.recommender import SingleItemCollator
from ...dataset import ChainedDataset, RecItemAsSequenceDataset
from ...recommender.utils import load_json_files

__all__ = [
    "DistillSequenceDataModule",
]


class DistillSequenceDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_paths: list[Path],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_len: int,
        max_attribute_len: int,
        max_items: int,
        sequence_embeddings: list[torch.Tensor],
        train_data_split: str,
        sequence_per_dataset: int | None = None,
        num_workers: int = 0,
        item_prompt: str | None = None,
        sequence_prompt: str | None = None,
        valid_ratio: float | None = None,
        reverse_sequence: bool = True,
        num_sequences_per_dataset: int | None = None,
        sample_method: str = "random",
    ):
        super().__init__()

        self.dataset_paths = dataset_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_attribute_len = max_attribute_len
        self.max_items = max_items
        self.sequence_embeddings = sequence_embeddings
        self.train_data_split = train_data_split
        self.sequence_per_dataset = sequence_per_dataset
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.reverse_sequence = reverse_sequence
        self.num_sequences_per_dataset = num_sequences_per_dataset
        self.sample_method = sample_method

        assert valid_ratio is None or 0 <= valid_ratio <= 1, "valid_ratio must be between 0 and 1 or None"
        assert len(dataset_paths) == len(
            sequence_embeddings
        ), "dataset_paths and user_embeddings must have the same length"

        self.item_prompt = item_prompt or ""
        self.sequence_prompt = sequence_prompt or ""

        self.score_train_datasets = []
        self.score_valid_datasets = []
        self.item_datasets = []
        self.item_collators = []
        self.item_dataloaders = []
        self.item_texts = []
        self.distill_collator = None

    def setup(self, stage: str):
        for dataset_path, sequence_embedding in zip(self.dataset_paths, self.sequence_embeddings):
            item_dataset, train_dataset, val_dataset, test_dataset, metadata, _, _ = load_json_files(
                dataset_path, self.max_items
            )

            if self.train_data_split == "train":
                dataset = train_dataset
            elif self.train_data_split == "val":
                dataset = val_dataset
            elif self.train_data_split == "test":
                dataset = test_dataset
            elif self.train_data_split == "item":
                dataset = RecItemAsSequenceDataset(item_dataset.items)
            else:
                raise ValueError(f"Unknown train_data_split: {self.train_data_split}")

            assert len(dataset) == len(
                sequence_embedding
            ), "item_dataset and sequence_embedding must have the same length"

            if self.num_sequences_per_dataset is not None:
                print(
                    f"Sampling {self.num_sequences_per_dataset} sequences per dataset "
                    f"from {len(dataset)} total sequences"
                )

                if self.sample_method == "random":
                    # Random sampling of sequences
                    indices = torch.randperm(len(dataset))[: self.num_sequences_per_dataset].tolist()
                elif self.sample_method == "centroid":
                    indices = sample_centroid(sequence_embedding, self.num_sequences_per_dataset)
                elif self.sample_method == "popular":
                    indices = sample_popular(test_dataset.sequence, self.num_sequences_per_dataset)
                else:
                    raise ValueError(f"Unknown sample_method: {self.sample_method}")
                dataset = Subset(dataset, indices)

            item_text = {item_id: self._flatten_key_value(m) for item_id, m in metadata.items()}
            item_collator = SingleItemCollator(
                tokenizer=self.tokenizer,
                item_text=item_text,
                max_seq_len=self.max_seq_len,
                item_prompt=self.item_prompt,
            )
            item_dataloader = DataLoader(
                item_dataset,
                batch_size=self.batch_size,
                collate_fn=item_collator,
                shuffle=False,
                num_workers=self.num_workers,
            )

            score_train_dataset, score_valid_dataset = split_sequences(dataset, self.valid_ratio)

            self.item_datasets.append(item_dataset)
            self.item_collators.append(item_collator)
            self.item_dataloaders.append(item_dataloader)
            self.score_train_datasets.append(score_train_dataset)
            self.score_valid_datasets.append(score_valid_dataset)
            self.item_texts.append(item_text)

        self.distill_collator = DistillSequenceCollator(
            tokenizer=self.tokenizer,
            item_texts=self.item_texts,
            max_seq_len=self.max_seq_len,
            separator="; ",
            sequence_prompt=self.sequence_prompt,
            reverse_sequence=self.reverse_sequence,
        )

    def _flatten_key_value(self, item_metadata: dict[str, str]) -> str:
        result = []
        for k, v in item_metadata.items():
            assert isinstance(v, str)
            tokenized = self.tokenizer.tokenize(v)
            tokenized = tokenized[: self.max_attribute_len]
            tokenized = self.tokenizer.convert_tokens_to_string(tokenized)
            result.append(f"{k}: {tokenized}")

        return " ".join(result)

    def train_dataloader(self):
        # sampled_train_datasets = []

        # for dataset in self.score_train_datasets:
        #     indices = torch.randperm(len(dataset))[: self.batch_size].tolist()
        #     subset = Subset(dataset, indices)
        #     sampled_train_datasets.append(subset)
        sampled_train_datasets = self.score_train_datasets

        return DataLoader(
            ChainedDataset(sampled_train_datasets),
            batch_size=self.batch_size,
            collate_fn=self.distill_collator,
            num_workers=self.num_workers,
            # shuffle=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ChainedDataset([valid_dataset], start_dataset_idx=i),
                batch_size=self.batch_size,
                collate_fn=self.distill_collator,
                num_workers=self.num_workers,
                shuffle=False,
            )
            for i, valid_dataset in enumerate(self.score_valid_datasets)
            if valid_dataset is not None
        ]
