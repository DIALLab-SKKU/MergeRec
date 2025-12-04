import random

import torch

from ...utils.recformer_utils import concat_tokenized_items, pad_tokenized_sequences
from ....types import *

__all__ = [
    "RecformerSingleItemCollator",
    "RecformerItemSequenceCollator",
]


class RecformerSingleItemCollator:
    def __init__(
        self,
        bos_token_id: int,
        pad_token_id: int,
        tokenized_items: dict[ItemID, TokenizedItem],
        max_seq_len: int,
    ):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.tokenized_items = tokenized_items
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[ItemID]) -> BatchItem:
        tokenized_items = [
            concat_tokenized_items([self.tokenized_items[item_id]], self.bos_token_id) for item_id in batch
        ]
        batch_encoding = pad_tokenized_sequences(tokenized_items, self.pad_token_id, self.max_seq_len)
        return BatchItem(items=batch_encoding)


class RecformerItemSequenceCollator:
    def __init__(
        self,
        bos_token_id: int,
        pad_token_id: int,
        tokenized_items: dict[ItemID, TokenizedItem],
        max_seq_len: int,
        num_negative: int | None,
        in_batch_negative: bool,
    ):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.tokenized_items = tokenized_items
        self.max_seq_len = max_seq_len
        self.num_negative = num_negative
        self.in_batch_negative = in_batch_negative

        self._all_items = set(tokenized_items.keys())

    @staticmethod
    def _split_sequence(batch: list[list[ItemID]]):
        input_items: list[list[ItemID]] = []
        target_items: list[ItemID] = []

        for _, seq in batch:
            input_seq = seq[:-1]
            input_seq = input_seq[::-1]
            input_items.append(input_seq)
            target_items.append(seq[-1])

        return input_items, target_items

    def __call__(self, batch: list[list[ItemID]]) -> BatchSequence | BatchSequenceWithNegative:
        input_items, target_items = self._split_sequence(batch)

        input_items_tokenized = [[self.tokenized_items[item_id] for item_id in seq] for seq in input_items]
        input_items_sequence = [concat_tokenized_items(seq, self.bos_token_id) for seq in input_items_tokenized]
        sequence_encoding = pad_tokenized_sequences(input_items_sequence, self.pad_token_id, self.max_seq_len)

        if self.num_negative is None and not self.in_batch_negative:
            return BatchSequence(sequence=sequence_encoding, labels=torch.tensor(target_items, dtype=torch.long))

        target_items_tokenized = [[self.tokenized_items[item_id]] for item_id in target_items]
        target_items_sequence = [concat_tokenized_items(seq, self.bos_token_id) for seq in target_items_tokenized]
        target_encoding = pad_tokenized_sequences(target_items_sequence, self.pad_token_id, self.max_seq_len)

        if self.num_negative is None:
            negative_encoding = None
        else:
            negative_items = []
            for seq in batch:
                negatives = random.sample(list(self._all_items - set(seq)), self.num_negative)
                negative_items.extend(negatives)

            negative_items_tokenized = [[self.tokenized_items[item_id] for item_id in seq] for seq in negative_items]
            negative_items_sequence = [
                concat_tokenized_items(seq, self.bos_token_id) for seq in negative_items_tokenized
            ]
            negative_encoding = pad_tokenized_sequences(negative_items_sequence, self.pad_token_id, self.max_seq_len)

        return BatchSequenceWithNegative(
            sequence=sequence_encoding, target=target_encoding, negatives=negative_encoding
        )
