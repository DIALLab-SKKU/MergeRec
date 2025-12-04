import json
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch.utils.data import Subset, DataLoader
from transformers import PreTrainedTokenizer

from .utils import sample_items
from ...collator.distiller import DistillItemCollator
from ...collator.recommender import SingleItemCollator
from ...dataset import RecItemDataset, ChainedDataset

__all__ = [
    "DistillItemDataModule",
]


class DistillItemDataModule(L.LightningDataModule):
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
        item_prompt: str | None = None,
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

        assert valid_ratio is None or 0 <= valid_ratio <= 1, "valid_ratio must be between 0 and 1 or None"
        assert item_sample_method in ["random", "centroid"], "item_sample_method must be 'random' or 'centroid'"
        assert len(dataset_paths) == len(item_embeddings), "dataset_paths and item_embeddings must have the same length"

        if item_prompt is None:
            self.item_prompt = ""
        else:
            self.item_prompt = item_prompt

        self.score_train_datasets = []
        self.score_valid_datasets = []
        self.chained_train_dataset = []
        self.chained_valid_dataset = []
        self.item_datasets = []
        self.item_collators = []
        self.item_dataloaders = []
        self.item_texts = []
        self.distill_collator = None

    def setup(self, stage: str):
        for dataset_path, item_embedding in zip(self.dataset_paths, self.item_embeddings):
            items, item_text = self._load_data(dataset_path)

            score_dataset = RecItemDataset(items=items)
            item_dataset = RecItemDataset(items=items)
            item_collator = SingleItemCollator(
                tokenizer=self.tokenizer,
                item_text=item_text,
                max_seq_len=self.max_seq_len,
                item_prompt=self.item_prompt,
            )

            assert len(item_dataset) == len(item_embedding), "Item dataset and item embedding must have the same length"

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
            self.item_texts.append(item_text)

        self.distill_collator = DistillItemCollator(
            tokenizer=self.tokenizer,
            item_texts=self.item_texts,
            max_seq_len=self.max_seq_len,
            item_prompt=self.item_prompt,
        )

    def _flatten_key_value(self, item_metadata: dict[str, str]) -> str:
        result = []
        for k, v in item_metadata.items():
            # Tokenize value and truncate to max attribute length
            assert isinstance(v, str), "Item metadata value must be a string"

            tokenized = self.tokenizer.tokenize(v)
            tokenized = tokenized[: self.max_attribute_len]
            tokenized = self.tokenizer.convert_tokens_to_string(tokenized)

            # Append key and tokenized value to result
            result.append(f"{k}: {tokenized}")

        return " ".join(result)

    def _load_data(self, dataset_path: Path):
        with open(dataset_path / "meta_data.json", "r") as f:
            metadata = json.load(f)

        with open(dataset_path / "smap.json", "r") as f:
            smap = json.load(f)

        metadata = {
            smap[item_asin]: item_metadata for item_asin, item_metadata in metadata.items() if item_asin in smap
        }

        items = list(smap.values())
        item_text = {item_id: self._flatten_key_value(item_metadata) for item_id, item_metadata in metadata.items()}

        return items, item_text

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
            num_workers=self.num_workers,
            shuffle=False,
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
