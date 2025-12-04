import numpy as np
import torch
from sklearn.cluster import KMeans

from torch.utils.data import Dataset, Subset

__all__ = [
    "sample_items",
]


def sample_items(
    item_embedding: torch.Tensor,
    score_dataset: Dataset,
    item_per_dataset: int | None,
    item_sample_method: str,
    valid_ratio: float | None = None,
):
    if item_per_dataset is not None and item_per_dataset > 0:
        # Sample items from the dataset
        if item_sample_method == "random":
            indices = torch.randperm(len(score_dataset))[:item_per_dataset].tolist()
        elif item_sample_method == "centroid":
            indices = sample_centroid(item_embedding, item_per_dataset)
        else:
            raise ValueError(f"Unknown item sample method: {item_sample_method}")
        score_dataset = Subset(score_dataset, indices)

    if valid_ratio is None:
        score_train_dataset = score_dataset
        score_valid_dataset = None
    else:
        indices = torch.randperm(len(score_dataset))
        train_indices = indices[: int(len(score_dataset) * (1 - valid_ratio))]
        valid_indices = indices[int(len(score_dataset) * (1 - valid_ratio)) :]

        score_train_dataset = Subset(score_dataset, train_indices)
        score_valid_dataset = Subset(score_dataset, valid_indices)
    return score_train_dataset, score_valid_dataset


def sample_centroid(item_embedding: torch.Tensor, item_per_dataset: int) -> list[int]:
    assert isinstance(item_embedding, torch.Tensor), "item_embedding must be a torch.Tensor"
    assert item_embedding.ndim == 2, "item_embedding must be 2-dimensional (N, d)"
    N, d = item_embedding.shape
    assert 1 <= item_per_dataset <= N, "item_per_dataset must be between 1 and N"

    X = item_embedding.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=item_per_dataset)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    indices = []
    for i, center in enumerate(centers):
        member_idxs = np.where(labels == i)[0]
        assert member_idxs.size > 0, f"Cluster {i} has no members"
        member_vectors = X[member_idxs]
        dists = np.linalg.norm(member_vectors - center, axis=1)
        chosen = member_idxs[int(np.argmin(dists))]
        indices.append(chosen.item())

    assert len(set(indices)) == item_per_dataset, "Duplicate indices found"

    return indices
