import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer

from .utils import sample_items
from ...collator.distiller import RecformerDistillItemCollator
from ...collator.recommender import RecformerSingleItemCollator
from ...dataset import RecItemDataset, ChainedDataset
from ...utils import recformer_utils
from ....types import *

__all__ = [
    "DistillItemDataModuleForRecformer",
]


class _IntFactory:
    def __init__(self):
        self._cnt = 0

    def __call__(self) -> int:
        self._cnt += 1
        return self._cnt


class DistillItemDataModuleForRecformer(L.LightningDataModule):
    def __init__(
        self,
        dataset_paths: list[Path],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_len: int,
        max_attribute_len: int,
        item_embeddings: list[torch.Tensor],
        item_per_dataset: int | None = None,
        num_workers: int = 0,
        valid_ratio: float | None = None,
        item_sample_method: Literal["random", "centroid"] = "random",
    ):
        super().__init__()

        self.dataset_paths = dataset_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_attribute_len = max_attribute_len
        self.item_embeddings = item_embeddings
        self.item_per_dataset = item_per_dataset
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.item_sample_method = item_sample_method

        assert valid_ratio is None or 0 <= valid_ratio <= 1
        assert item_sample_method in ["random", "centroid"]
        assert len(dataset_paths) == len(item_embeddings)

        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self._attr_name_id_map = defaultdict(_IntFactory())

        self.score_train_datasets = []
        self.score_valid_datasets = []
        self.chained_train_dataset = []
        self.chained_valid_dataset = []
        self.item_datasets = []
        self.item_collators = []
        self.item_dataloaders = []
        self.tokenized_items = []
        self.distill_collator = None

    def setup(self, stage: str):
        for dataset_path, item_embedding in zip(self.dataset_paths, self.item_embeddings):
            items, tokenized_item = self._load_data(dataset_path)

            score_dataset = RecItemDataset(items=items)
            item_dataset = RecItemDataset(items=items)
            item_collator = RecformerSingleItemCollator(
                bos_token_id=self.bos_token_id,
                pad_token_id=self.pad_token_id,
                tokenized_items=tokenized_item,
                max_seq_len=self.max_seq_len,
            )

            assert len(item_dataset) == len(item_embedding), "item_dataset and item_embedding must have the same length"

            score_train_dataset, score_valid_dataset = sample_items(
                item_embedding, score_dataset, self.item_per_dataset, self.item_sample_method, self.valid_ratio
            )

            item_dataloader = DataLoader(
                item_dataset,
                batch_size=self.batch_size,
                collate_fn=item_collator,
                shuffle=False,
                num_workers=self.num_workers,
            )

            self.score_train_datasets.append(score_train_dataset)
            self.score_valid_datasets.append(score_valid_dataset)
            self.item_datasets.append(item_dataset)
            self.item_collators.append(item_collator)
            self.item_dataloaders.append(item_dataloader)
            self.tokenized_items.append(tokenized_item)

        self.distill_collator = RecformerDistillItemCollator(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            tokenized_items=self.tokenized_items,
            max_seq_len=self.max_seq_len,
        )

    def _tokenize_item(self, item_metadata: dict[str, str]) -> TokenizedItem:
        return recformer_utils.tokenize_item(
            item_metadata, self.tokenizer, self._attr_name_id_map, self.max_attribute_len
        )

    def _load_data(self, dataset_path: Path):
        with open(dataset_path / "meta_data.json") as f:
            metadata = json.load(f)
        with open(dataset_path / "smap.json") as f:
            smap = json.load(f)

        # keep only ids present in mapping
        metadata = {
            smap[item_asin]: item_metadata for item_asin, item_metadata in metadata.items() if item_asin in smap
        }
        items = list(smap.values())
        tokenized_items = {item_id: self._tokenize_item(item_metadata) for item_id, item_metadata in metadata.items()}

        return items, tokenized_items

    def train_dataloader(self):
        sampled_train_datasets = []

        for dataset in self.score_train_datasets:
            indices = torch.randperm(len(dataset))[: self.batch_size].tolist()
            subset = Subset(dataset, indices)
            sampled_train_datasets.append(subset)

        return DataLoader(
            ChainedDataset(sampled_train_datasets),
            batch_size=self.batch_size,
            collate_fn=self.distill_collator,
            shuffle=False,
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
