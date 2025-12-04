from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer

from .utils import split_sequences, sample_popular
from ..item.utils import sample_centroid
from ...collator.distiller.recformer import RecformerDistillSequenceCollator
from ...collator.recommender import RecformerSingleItemCollator
from ...dataset import ChainedDataset, RecItemAsSequenceDataset
from ...recommender.utils import load_json_files
from ...utils import recformer_utils

__all__ = [
    "DistillSequenceDataModuleForRecformer",
]


class _IntFactory:
    def __init__(self):
        self._cnt = 0

    def __call__(self) -> int:
        self._cnt += 1
        return self._cnt


class DistillSequenceDataModuleForRecformer(L.LightningDataModule):
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
        valid_ratio: float | None = None,
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
        self.num_sequences_per_dataset = num_sequences_per_dataset
        self.sample_method = sample_method

        assert valid_ratio is None or 0 <= valid_ratio <= 1
        assert len(dataset_paths) == len(sequence_embeddings)

        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self._attr_name_id_map = defaultdict(_IntFactory())

        self.score_train_datasets = []
        self.score_valid_datasets = []
        self.item_datasets = []
        self.item_collators = []
        self.item_dataloaders = []
        self.tokenized_items = []
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

            assert len(dataset) == len(sequence_embedding)

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

            tokenized_item = {
                item_id: recformer_utils.tokenize_item(
                    metadata[item_id], self.tokenizer, self._attr_name_id_map, self.max_attribute_len
                )
                for item_id in metadata
            }
            item_collator = RecformerSingleItemCollator(
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                tokenized_items=tokenized_item,
                max_seq_len=self.max_seq_len,
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
            self.tokenized_items.append(tokenized_item)

        self.distill_collator = RecformerDistillSequenceCollator(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            tokenized_items=self.tokenized_items,
            max_seq_len=self.max_seq_len,
        )

    def train_dataloader(self):
        # sampled_train_datasets = []
        #
        # for dataset in self.score_train_datasets:
        #     indices = torch.randperm(len(dataset))[: self.batch_size].tolist()
        #     subset = Subset(dataset, indices)
        #     sampled_train_datasets.append(subset)
        sampled_train_datasets = self.score_train_datasets

        return DataLoader(
            ChainedDataset(sampled_train_datasets),
            batch_size=self.batch_size,
            collate_fn=self.distill_collator,
            # shuffle=False,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ChainedDataset([valid_dataset], start_dataset_idx=i),
                batch_size=self.batch_size,
                collate_fn=self.distill_collator,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for i, valid_dataset in enumerate(self.score_valid_datasets)
            if valid_dataset is not None
        ]
