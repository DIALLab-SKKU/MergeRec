from dataclasses import dataclass, fields, replace
from typing import TypeVar

import torch
from transformers import BatchEncoding

__all__ = [
    "BatchItem",
    "BatchSequence",
    "BatchSequenceWithNegative",
    "BatchDistillationItem",
    "BatchDistillationSequence",
]


T = TypeVar("T", bound=dataclass)


class ToDeviceMixin:
    def to(self: T, device: torch.device) -> T:
        def _move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, BatchEncoding):
                return obj.to(device)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move(v) for v in obj)
            return obj

        moved_fields = {f.name: _move(getattr(self, f.name)) for f in fields(self)}
        return replace(self, **moved_fields)


@dataclass
class BatchItem(ToDeviceMixin):
    items: BatchEncoding


@dataclass
class BatchSequence(ToDeviceMixin):
    sequence: BatchEncoding
    labels: torch.Tensor


@dataclass
class BatchSequenceWithNegative(ToDeviceMixin):
    sequence: BatchEncoding

    target: BatchEncoding
    negatives: BatchEncoding | None


@dataclass
class BatchDistillationItem(ToDeviceMixin):
    dataset_indexes: list[int]
    item_ids: list[int]
    items: BatchEncoding


@dataclass
class BatchDistillationSequence(ToDeviceMixin):
    dataset_indexes: list[int]
    sequence_ids: list[int]
    sequence: BatchEncoding
