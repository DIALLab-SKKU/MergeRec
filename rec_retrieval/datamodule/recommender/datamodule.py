from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .utils import load_json_files
from ..collator.recommender import SingleItemCollator, ItemSequenceCollator
from ...configs import NegativeSampleConfig

__all__ = [
    "RecDataModule",
]


class RecDataModule(L.LightningDataModule):
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
        sequence_prompt: str | None = None,
        item_prompt: str | None = None,
        reverse_sequence: bool = True,
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
        self.reverse_sequence = reverse_sequence

        if sequence_prompt is None:
            self.sequence_prompt = ""
        else:
            self.sequence_prompt = sequence_prompt
        if item_prompt is None:
            self.item_prompt = ""
        else:
            self.item_prompt = item_prompt

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.item_dataset = None
        self.metadata = None
        self.tokenized_items = None
        self.item_collator = None
        self.sequence_train_collator = None
        self.sequence_eval_collator = None
        self.item_text = None

    def setup(self, stage: str):
        # Load json files
        self.item_dataset, self.train_dataset, self.val_dataset, self.test_dataset, self.metadata, _, _ = (
            load_json_files(self.dataset_path, self.max_items)
        )

        self.item_text = {
            item_id: self._flatten_key_value(item_metadata) for item_id, item_metadata in self.metadata.items()
        }

        self.item_collator = SingleItemCollator(
            tokenizer=self.tokenizer,
            item_text=self.item_text,
            max_seq_len=self.max_seq_len,
            item_prompt=self.item_prompt,
        )

        self.sequence_train_collator = ItemSequenceCollator(
            tokenizer=self.tokenizer,
            item_text=self.item_text,
            max_seq_len=self.max_seq_len,
            num_negative=self.negative_sample.k,
            in_batch_negative=self.negative_sample.in_batch,
            sequence_prompt=self.sequence_prompt,
            item_prompt=self.item_prompt,
            reverse_sequence=self.reverse_sequence,
        )
        self.sequence_eval_collator = ItemSequenceCollator(
            tokenizer=self.tokenizer,
            item_text=self.item_text,
            max_seq_len=self.max_seq_len,
            num_negative=None,
            in_batch_negative=False,
            sequence_prompt=self.sequence_prompt,
            item_prompt=self.item_prompt,
            reverse_sequence=self.reverse_sequence,
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
