import random

import torch
from transformers import PreTrainedTokenizer

from ....types import ItemID, BatchItem, BatchSequence, BatchSequenceWithNegative

__all__ = [
    "SingleItemCollator",
    "ItemSequenceCollator",
]


class SingleItemCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        item_text: dict[ItemID, str],
        max_seq_len: int,
        item_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.item_text = item_text
        self.max_seq_len = max_seq_len
        self.item_prompt = item_prompt

    def __call__(self, batch: list[ItemID]) -> BatchItem:
        item_texts = [self.item_prompt + self.item_text[item] for item in batch]
        batch_encoding = self.tokenizer(
            item_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len
        )
        return BatchItem(items=batch_encoding)


class ItemSequenceCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        item_text: dict[ItemID, str],
        max_seq_len: int,
        num_negative: int | None,
        in_batch_negative: bool,
        separator: str = "; ",
        sequence_prompt: str = "",
        item_prompt: str = "",
        reverse_sequence: bool = True,
    ):
        self.tokenizer = tokenizer
        self.item_text = item_text
        self.max_seq_len = max_seq_len
        self.num_negative = num_negative
        self.in_batch_negative = in_batch_negative
        self.separator = separator
        self.sequence_prompt = sequence_prompt
        self.item_prompt = item_prompt
        self.reverse_sequence = reverse_sequence

        self._all_items = set(item_text.keys())

    def _split_sequence(self, batch: list[list[ItemID]]):
        input_items: list[list[ItemID]] = []
        target_items: list[ItemID] = []

        for _, seq in batch:
            input_seq = seq[:-1]
            if self.reverse_sequence:
                input_seq = input_seq[::-1]
            input_items.append(input_seq)
            target_items.append(seq[-1])

        return input_items, target_items

    def __call__(self, batch: list[list[ItemID]]) -> BatchSequence | BatchSequenceWithNegative:
        input_items, target_items = self._split_sequence(batch)

        # Cut sequence from start until it fits in max_seq_len
        if not self.reverse_sequence:
            input_items_cut = []
            for input_item in input_items:
                input_item_cut = input_item.copy()
                while True:
                    tokenized = self.tokenizer.tokenize(
                        self.sequence_prompt + self.separator.join(self.item_text[i] for i in input_item_cut)
                    )
                    if len(tokenized) <= self.max_seq_len:
                        break
                    input_item_cut.pop(0)
                input_items_cut.append(input_item_cut)
            input_items = input_items_cut

        sequence_encoding = self.tokenizer(
            [self.sequence_prompt + self.separator.join(self.item_text[i] for i in seq) for seq in input_items],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        if self.num_negative is None and not self.in_batch_negative:
            return BatchSequence(sequence=sequence_encoding, labels=torch.tensor(target_items, dtype=torch.long))

        target_encoding = self.tokenizer(
            [self.item_prompt + self.item_text[target_item] for target_item in target_items],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        if self.num_negative is None:
            negative_encoding = None
        else:
            negative_items = []
            for seq in batch:
                negatives = random.sample(list(self._all_items - set(seq)), self.num_negative)
                negative_items.extend(negatives)

            negative_encoding = self.tokenizer(
                [self.item_prompt + self.item_text[negative_item] for negative_item in negative_items],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_len,
            )

        return BatchSequenceWithNegative(
            sequence=sequence_encoding, target=target_encoding, negatives=negative_encoding
        )
