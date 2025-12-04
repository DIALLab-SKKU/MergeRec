from collections import defaultdict
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .utils import load_json_files
from ..collator.recommender import RecformerItemSequenceCollator, RecformerSingleItemCollator
from ..utils import recformer_utils
from ...configs import NegativeSampleConfig
from ...types import TokenizedItem

__all__ = [
    "RecDataModuleForRecformer",
]


class _IntFactory:
    def __init__(self):
        self._counter = 0

    def __call__(self) -> int:
        self._counter += 1
        return self._counter


class RecDataModuleForRecformer(L.LightningDataModule):

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_len: int,
        max_attribute_len: int,
        max_items: int,
        negative_sample: NegativeSampleConfig,
        num_workers: int = 0,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_attribute_len = max_attribute_len
        self.max_items = max_items
        self.negative_sample = negative_sample
        self.num_workers = num_workers

        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self._attr_name_id_map = defaultdict(_IntFactory())

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.item_dataset = None
        self.metadata = None
        self.tokenized_items = None
        self.item_collator = None
        self.sequence_train_collator = None
        self.sequence_eval_collator = None

    def setup(self, stage: str):
        self.item_dataset, self.train_dataset, self.val_dataset, self.test_dataset, self.metadata, _, _ = (
            load_json_files(self.dataset_path, self.max_items)
        )

        self.tokenized_items = {
            item_id: self._tokenize_item(item_metadata) for item_id, item_metadata in self.metadata.items()
        }

        self.item_collator = RecformerSingleItemCollator(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            tokenized_items=self.tokenized_items,
            max_seq_len=self.max_seq_len,
        )

        self.sequence_train_collator = RecformerItemSequenceCollator(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            tokenized_items=self.tokenized_items,
            max_seq_len=self.max_seq_len,
            num_negative=self.negative_sample.k,
            in_batch_negative=self.negative_sample.in_batch,
        )
        self.sequence_eval_collator = RecformerItemSequenceCollator(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            tokenized_items=self.tokenized_items,
            num_negative=None,
            in_batch_negative=False,
            max_seq_len=self.max_seq_len,
        )

    def _tokenize_item(self, item_metadata: dict[str, str]) -> TokenizedItem:
        return recformer_utils.tokenize_item(
            item_metadata, self.tokenizer, self._attr_name_id_map, self.max_attribute_len
        )

    def item_dataloader(self):
        return DataLoader(
            self.item_dataset,
            batch_size=self.batch_size,
            collate_fn=self.item_collator,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.sequence_train_collator,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.sequence_eval_collator,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.sequence_eval_collator,
            shuffle=False,
            num_workers=self.num_workers,
        )
