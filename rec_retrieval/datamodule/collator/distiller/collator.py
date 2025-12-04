import torch
from transformers import PreTrainedTokenizer

from ....types import *

__all__ = [
    "DistillItemCollator",
    "DistillSequenceCollator",
]


class DistillItemCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        item_texts: list[dict[ItemID, str]],
        max_seq_len: int,
        item_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.item_texts = item_texts
        self.max_seq_len = max_seq_len
        self.item_prompt = item_prompt

    def __call__(self, batch: list[tuple[int, ItemID, torch.Tensor]]) -> BatchDistillationItem:
        dataset_indexes = []
        item_ids = []
        item_texts = []
        for dataset_index, item_id in batch:
            dataset_indexes.append(dataset_index)
            item_ids.append(item_id)

            item_texts.append(self.item_prompt + self.item_texts[dataset_index][item_id])

        batch_encoding = self.tokenizer(
            item_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len
        )
        return BatchDistillationItem(dataset_indexes=dataset_indexes, item_ids=item_ids, items=batch_encoding)


class DistillSequenceCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        item_texts: list[dict[ItemID, str]],
        max_seq_len: int,
        separator: str = "; ",
        sequence_prompt: str = "",
        reverse_sequence: bool = True,
    ):
        self.tokenizer = tokenizer
        self.item_texts = item_texts
        self.max_seq_len = max_seq_len
        self.separator = separator
        self.sequence_prompt = sequence_prompt
        self.reverse_sequence = reverse_sequence

    def _split_sequence(self, batch: list[tuple[int, UserID, list[ItemID]]]):
        dataset_indexes = []
        sequence_ids: list[ItemID] = []
        sequence: list[list[ItemID]] = []

        for dataset_index, (sequence_id, seq) in batch:
            dataset_indexes.append(dataset_index)
            sequence_ids.append(sequence_id)
            if self.reverse_sequence:
                seq = seq[:-1][::-1]
            sequence.append(seq)

        return dataset_indexes, torch.tensor(sequence_ids, device=torch.device("cpu")), sequence

    def __call__(self, batch: list[tuple[int, list[ItemID]]]) -> BatchDistillationSequence:
        dataset_indexes, sequence_ids, sequence = self._split_sequence(batch)

        sequence_texts = [
            self.sequence_prompt + self.separator.join(self.item_texts[dataset_index][i] for i in sequence)
            for dataset_index, sequence in zip(dataset_indexes, sequence)
        ]
        batch_encoding = self.tokenizer(
            sequence_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        return BatchDistillationSequence(
            dataset_indexes=dataset_indexes,
            sequence_ids=sequence_ids,
            sequence=batch_encoding,
        )
