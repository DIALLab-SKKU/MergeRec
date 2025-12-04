from typing import List

import torch

from .task_vector import get_task_vectors
from ..types import FlattenedModel


def _normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y


def _clamp(x, min_ratio: float, max_ratio: float):
    if len(x.size()) == 1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        clamp_min = sorted_x[int(d * min_ratio)]
        clamp_max = sorted_x[int(d * (1 - max_ratio) - 1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        clamp_min = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        clamp_max = sorted_x[:, int(d * (1 - max_ratio) - 1)].unsqueeze(1)

    clamped_x = torch.clamp(x, clamp_min, clamp_max)
    return clamped_x


def _act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y


def get_pcb_vectors(base_model: FlattenedModel, models: List[FlattenedModel], density: float = 0.2, **__):
    task_vectors = get_task_vectors(base_model, models).clone()

    n, d = task_vectors.shape

    all_checks_abs = _clamp(torch.abs(task_vectors), min_ratio=0.01, max_ratio=0.01)  # (n, d)
    clamped_all_checks = torch.sign(task_vectors) * all_checks_abs  # (n, d)

    self_pcb = _normalize(all_checks_abs, 1) ** 2  # (n, d)
    self_pcb_act = torch.exp(n * self_pcb)  # (n, d)

    cross_pcb = task_vectors * torch.sum(task_vectors, dim=0)  # (n, d)
    cross_pcb_act = _act(cross_pcb)  # (n, d)

    task_pcb = self_pcb_act * cross_pcb_act  # (n, d)

    scale = _normalize(_clamp(task_pcb, 1 - density, 0), dim=1)  # (n, d)
    pcb_vectors = clamped_all_checks * scale  # (n, d)
    pcb_vectors = pcb_vectors / torch.clamp(torch.sum(scale, dim=0, keepdim=True), min=1e-12)  # (n, d)
    pcb_vectors = pcb_vectors / n

    return pcb_vectors


def merge_pcb(
    base_model: FlattenedModel, models: List[FlattenedModel], weights: List[float], density: float = 0.2, **__
) -> FlattenedModel:
    assert len(models) == len(weights), "Number of models and weights should match."

    merged = base_model.clone()
    pcb_vectors = get_pcb_vectors(base_model, models, density=density)
    for i, pcb_vector in enumerate(pcb_vectors):
        merged += weights[i] * pcb_vector

    return merged
