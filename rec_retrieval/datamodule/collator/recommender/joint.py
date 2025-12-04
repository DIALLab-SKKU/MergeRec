import random

import torch
from transformers import PreTrainedTokenizer

from ...utils.recformer_utils import concat_tokenized_items, pad_tokenized_sequences
from ....types import ItemID, BatchSequence, BatchSequenceWithNegative, TokenizedItem

__all__ = [
    "JointItemSequenceCollator",
    "RecformerJointItemSequenceCollator",
]


class JointItemSequenceCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        item_texts: list[dict[ItemID, str]],
        max_seq_len: int,
        num_negative: int | None,
        in_batch_negative: bool,
        separator: str = "; ",
        sequence_prompt: str = "",
        item_prompt: str = "",
        reverse_sequence: bool = True,
    ):
        self.tokenizer = tokenizer
        self.item_texts = item_texts
        self.max_seq_len = max_seq_len
        self.num_negative = num_negative
        self.in_batch_negative = in_batch_negative
        self.separator = separator
        self.sequence_prompt = sequence_prompt
        self.item_prompt = item_prompt
        self.reverse_sequence = reverse_sequence

        self._all_items = [set(item_text.keys()) for item_text in item_texts]

    def _split_sequence(self, batch: list[tuple[int, list[ItemID]]]):
        dataset_indexes = []
        input_items: list[list[ItemID]] = []
        target_items: list[ItemID] = []

        for dataset_index, seq in batch:
            dataset_indexes.append(dataset_index)
            input_seq = seq[:-1]
            if self.reverse_sequence:
                input_seq = input_seq[::-1]
            input_items.append(input_seq)
            target_items.append(seq[-1])

        return dataset_indexes, input_items, target_items

    def __call__(self, batch: list[tuple[int, list[ItemID]]]) -> BatchSequence | BatchSequenceWithNegative:
        dataset_indexes, input_items, target_items = self._split_sequence(batch)

        # Cut sequence from start until it fits in max_seq_len
        if not self.reverse_sequence:
            input_items_cut = []
            for input_item in input_items:
                input_item_cut = input_item.copy()
                while True:
                    tokenized = self.tokenizer.tokenize(
                        self.sequence_prompt
                        + self.separator.join(
                            self.item_texts[dataset_index][i]
                            for dataset_index, i in zip(dataset_indexes, input_item_cut)
                        )
                    )
                    if len(tokenized) <= self.max_seq_len:
                        break
                    input_item_cut.pop(0)
                input_items_cut.append(input_item_cut)
            input_items = input_items_cut

        sequence_encoding = self.tokenizer(
            [
                self.sequence_prompt + self.separator.join(self.item_texts[dataset_index][i] for i in seq)
                for dataset_index, seq in zip(dataset_indexes, input_items)
            ],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        if self.num_negative is None and not self.in_batch_negative:
            return BatchSequence(sequence=sequence_encoding, labels=torch.tensor(target_items, dtype=torch.long))

        target_encoding = self.tokenizer(
            [
                self.item_prompt + self.item_texts[dataset_index][target_item]
                for dataset_index, target_item in zip(dataset_indexes, target_items)
            ],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        if self.num_negative is None:
            negative_encoding = None
        else:
            negative_items = []
            negative_dataset_indexes = []
            for dataset_index, seq in batch:
                negatives = random.sample(list(self._all_items[dataset_index] - set(seq)), self.num_negative)
                negative_items.extend(negatives)
                negative_dataset_indexes.extend([dataset_index] * len(negatives))

            negative_encoding = self.tokenizer(
                [
                    self.item_prompt + self.item_texts[ds_idx][neg_item]
                    for ds_idx, neg_item in zip(negative_dataset_indexes, negative_items)
                ],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_len,
            )
        return BatchSequenceWithNegative(
            sequence=sequence_encoding, target=target_encoding, negatives=negative_encoding
        )


class RecformerJointItemSequenceCollator:
    def __init__(
        self,
        bos_token_id: int,
        pad_token_id: int,
        tokenized_items: list[dict[ItemID, TokenizedItem]],
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

        self._all_items = [set(item_dict.keys()) for item_dict in tokenized_items]

    @staticmethod
    def _split_sequence(batch: list[tuple[int, list[ItemID]]]):
        dataset_indexes = []
        input_items: list[list[ItemID]] = []
        target_items: list[ItemID] = []

        for dataset_index, seq in batch:
            dataset_indexes.append(dataset_index)
            input_seq = seq[:-1][::-1]
            input_items.append(input_seq)
            target_items.append(seq[-1])

        return dataset_indexes, input_items, target_items

    def __call__(self, batch: list[tuple[int, list[ItemID]]]) -> BatchSequence | BatchSequenceWithNegative:
        dataset_indexes, input_items, target_items = self._split_sequence(batch)

        input_items_tokenized = [
            [self.tokenized_items[dataset_index][item_id] for item_id in seq]
            for dataset_index, seq in zip(dataset_indexes, input_items)
        ]
        input_items_sequence = [concat_tokenized_items(seq, self.bos_token_id) for seq in input_items_tokenized]
        sequence_encoding = pad_tokenized_sequences(input_items_sequence, self.pad_token_id, self.max_seq_len)

        if self.num_negative is None and not self.in_batch_negative:
            return BatchSequence(sequence=sequence_encoding, labels=torch.tensor(target_items, dtype=torch.long))

        target_items_tokenized = [
            [self.tokenized_items[dataset_index][item_id]]
            for dataset_index, item_id in zip(dataset_indexes, target_items)
        ]
        target_items_sequence = [concat_tokenized_items(seq, self.bos_token_id) for seq in target_items_tokenized]
        target_encoding = pad_tokenized_sequences(target_items_sequence, self.pad_token_id, self.max_seq_len)

        if self.num_negative is None:
            negative_encoding = None
        else:
            negative_items = []
            negative_dataset_indexes = []
            for dataset_index, seq in batch:
                negatives = random.sample(list(self._all_items[dataset_index] - set(seq)), self.num_negative)
                negative_items.extend(negatives)
                negative_dataset_indexes.extend([dataset_index] * len(negatives))

            negative_items_tokenized = [
                [self.tokenized_items[ds_idx][item_id]]
                for ds_idx, item_id in zip(negative_dataset_indexes, negative_items)
            ]
            negative_items_sequence = [
                concat_tokenized_items(seq, self.bos_token_id) for seq in negative_items_tokenized
            ]
            negative_encoding = pad_tokenized_sequences(negative_items_sequence, self.pad_token_id, self.max_seq_len)

        return BatchSequenceWithNegative(
            sequence=sequence_encoding, target=target_encoding, negatives=negative_encoding
        )
