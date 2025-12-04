import torch

from ...utils.recformer_utils import concat_tokenized_items, pad_tokenized_sequences
from ....types import *

__all__ = [
    "RecformerDistillItemCollator",
    "RecformerDistillSequenceCollator",
]


class RecformerDistillItemCollator:
    def __init__(
        self,
        bos_token_id: int,
        pad_token_id: int,
        tokenized_items: list[dict[ItemID, TokenizedItem]],
        max_seq_len: int,
    ):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.tokenized_items = tokenized_items
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[tuple[int, ItemID, torch.Tensor]]) -> BatchDistillationItem:
        dataset_indexes: list[int] = []
        item_ids: list[ItemID] = []
        tokenized_sequences: list[TokenizedSequence] = []

        for dataset_index, item_id, *_ in batch:
            dataset_indexes.append(dataset_index)
            item_ids.append(item_id)

            # 단일 아이템 → BOS 토큰과 함께 하나의 시퀀스로 변환
            tokenized_item = self.tokenized_items[dataset_index][item_id]
            tokenized_sequences.append(concat_tokenized_items([tokenized_item], self.bos_token_id))

        batch_encoding = pad_tokenized_sequences(tokenized_sequences, self.pad_token_id, self.max_seq_len)

        return BatchDistillationItem(
            dataset_indexes=dataset_indexes,
            item_ids=item_ids,
            items=batch_encoding,
        )


class RecformerDistillSequenceCollator:
    def __init__(
        self,
        bos_token_id: int,
        pad_token_id: int,
        tokenized_items: list[dict[ItemID, TokenizedItem]],
        max_seq_len: int,
    ):
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.tokenized_items = tokenized_items
        self.max_seq_len = max_seq_len

    def _split_sequence(self, batch: list[tuple[int, list[ItemID]]]):
        dataset_indexes = []
        input_sequence: list[list[ItemID]] = []
        for dataset_index, seq in batch:
            dataset_indexes.append(dataset_index)
            input_sequence.append(seq)
        return dataset_indexes, input_sequence

    def __call__(self, batch: list[tuple[int, list[ItemID]]]) -> BatchDistillationSequence:
        dataset_indexes, input_items = self._split_sequence(batch)
        tokenized_sequences = []
        sequence_ids = []

        for ds_idx, (seq_idx, seq) in zip(dataset_indexes, input_items):
            seq_tokens = [self.tokenized_items[ds_idx][item_id] for item_id in seq[:-1]]
            tokenized_sequences.append(concat_tokenized_items(seq_tokens, self.bos_token_id))
            sequence_ids.append(seq_idx)

        batch_encoding = pad_tokenized_sequences(tokenized_sequences, self.pad_token_id, self.max_seq_len)

        return BatchDistillationSequence(
            dataset_indexes=dataset_indexes, sequence_ids=sequence_ids, sequence=batch_encoding
        )
