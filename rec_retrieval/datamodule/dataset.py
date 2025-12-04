import random
from _bisect import bisect_right

from torch.utils.data import Dataset

from ..types import *


class RecItemDataset(Dataset):
    def __init__(self, items: list[ItemID]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> ItemID:
        return self.items[index]


class RecItemAsSequenceDataset(Dataset):
    def __init__(self, items: list[ItemID]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> tuple[int, list[ItemID]]:
        return index, [self.items[index], -1]


class RecDataset(Dataset):
    def __init__(self, sequence: Sequence, sample: bool, max_items: int):
        self.sequence = list(sequence.values())
        self.sample = sample
        self.max_items = max_items

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index) -> tuple[int, list[ItemID]]:
        seq = self.sequence[index]

        if self.sample:
            seq_len = len(seq)
            if seq_len < 2:
                return seq

            i = 0
            j = random.randint(i + 2, seq_len)
            subseq = seq[i:j]

            return index, subseq[-(self.max_items + 1) :]
        else:
            return index, seq[-(self.max_items + 1) :]


class ChainedDataset(Dataset):
    def __init__(self, datasets: list, start_dataset_idx: int = 0):
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes(datasets)
        self.start_dataset_idx = start_dataset_idx

    @staticmethod
    def _compute_cumulative_sizes(datasets):
        cumulative_sizes = []
        running_sum = 0
        for d in datasets:
            running_sum += len(d)
            cumulative_sizes.append(running_sum)
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Index out of range")
            idx = len(self) + idx

        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx + self.start_dataset_idx, self.datasets[dataset_idx][sample_idx]
