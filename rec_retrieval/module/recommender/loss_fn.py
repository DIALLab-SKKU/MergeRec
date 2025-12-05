from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F

from ...merger.enums import LossType

__all__ = [
    "DistillLossBase",
    "DistillCELoss",
    "DistillKDLoss",
    "DistillMSELoss",
    "DistillPairwiseLoss",
    "DistillListNetLoss",
    "distill_loss_factory",
]


class DistillLossBase(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, merged_model_output: torch.Tensor, single_model_output: torch.Tensor):
        """
        Args:
            merged_model_output: The vector representations of the merged model. (num_classes, d_model)
            single_model_output: The vector representations of the single model. (num_classes, d_model)

        Returns:
            torch.Tensor: The loss value.
        """

        raise NotImplementedError("Subclasses should implement this method.")


class DistillCELoss(DistillLossBase):
    # noinspection PyMethodMayBeStatic
    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # single_model_logits에서 가장 큰 값을 정답 label로 사용
        single_model_logits = single_model_logits.to(merged_model_logits.device)
        target = torch.argmax(single_model_logits, dim=-1)
        loss = F.cross_entropy(merged_model_logits, target)
        return loss


class DistillKDLoss(DistillLossBase):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        single_model_logits = single_model_logits.to(merged_model_logits.device)
        loss = F.kl_div(
            F.log_softmax(merged_model_logits / self.temperature, dim=-1),
            F.softmax(single_model_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature * self.temperature)

        return loss


class DistillAdaMergingLoss(DistillLossBase):
    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # Entropy minimization of merged model logits
        probs = F.softmax(merged_model_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # shape: (N,)
        loss = entropy.mean()
        return loss


class DistillAdaMergingKDLoss(DistillLossBase):
    def __init__(self, temperature: float, coefficient: float):
        super().__init__()
        self.temperature = temperature
        self.coefficient = coefficient

        self.adamerging = DistillAdaMergingLoss()
        self.kd = DistillKDLoss(temperature)

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        loss1 = self.adamerging(merged_model_logits, single_model_logits)
        loss2 = self.kd(merged_model_logits, single_model_logits)

        loss = loss1 + self.coefficient * loss2

        return loss


class MergedPseudoLabelLoss(DistillLossBase):
    def __init__(self):
        super().__init__()

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # Choose pseudo labels from merged model logits
        merged_model_pseudo_labels = torch.argmax(merged_model_logits, dim=-1)  # (N,)

        loss = F.cross_entropy(
            merged_model_logits,
            merged_model_pseudo_labels.to(merged_model_logits.device),
            reduction="mean",
        )

        return loss


class MergedPseudoLabelKDLoss(DistillKDLoss):
    def __init__(self, temperature: float, coefficient: float):
        super().__init__(temperature)
        self.coefficient = coefficient

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # Choose pseudo labels from merged model logits
        merged_model_pseudo_labels = torch.argmax(merged_model_logits, dim=-1)  # (N,)

        pseudo_label_loss = F.cross_entropy(
            merged_model_logits,
            merged_model_pseudo_labels.to(merged_model_logits.device),
            reduction="mean",
        )
        kd_loss = super().forward(merged_model_logits, single_model_logits)

        loss = pseudo_label_loss + self.coefficient * kd_loss

        return loss


class SinglePseudoLabelLoss(DistillLossBase):
    def __init__(self):
        super().__init__()

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # Choose pseudo labels from single model logits
        single_model_pseudo_labels = torch.argmax(single_model_logits, dim=-1)  # (N,)

        loss = F.cross_entropy(
            merged_model_logits,
            single_model_pseudo_labels.to(merged_model_logits.device),
            reduction="mean",
        )

        return loss


class SinglePseudoLabelKDLoss(DistillKDLoss):
    def __init__(self, temperature: float, coefficient: float):
        super().__init__(temperature)
        self.coefficient = coefficient

    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        # Choose pseudo labels from single model logits
        single_model_pseudo_labels = torch.argmax(single_model_logits, dim=-1)  # (N,)

        pseudo_label_loss = F.cross_entropy(
            merged_model_logits,
            single_model_pseudo_labels.to(merged_model_logits.device),
            reduction="mean",
        )
        kd_loss = super().forward(merged_model_logits, single_model_logits)

        loss = pseudo_label_loss + self.coefficient * kd_loss

        return loss


class DistillMSELoss(DistillLossBase):
    # noinspection PyMethodMayBeStatic
    def forward(self, merged_model_logits: torch.Tensor, single_model_logits: torch.Tensor):
        single_model_logits = single_model_logits.to(merged_model_logits.device)

        loss = F.mse_loss(merged_model_logits, single_model_logits, reduction="mean")

        return loss


class DistillPairwiseLoss(DistillLossBase):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, merged_model_output: torch.Tensor, single_model_output: torch.Tensor):
        # single_model_output에서 가장 큰 값과 두 번째로 큰 값의 인덱스를 positive, negative로 사용
        single_model_output = single_model_output.to(merged_model_output.device)
        pos_idx = torch.argmax(single_model_output, dim=-1)
        # negative는 가장 큰 값이 아닌 것 중 가장 큰 값
        neg_mask = torch.ones_like(single_model_output, dtype=torch.bool)
        neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
        neg_vals = single_model_output.masked_fill(~neg_mask, float("-inf"))
        neg_idx = torch.argmax(neg_vals, dim=-1)

        # merged_model_output에서 positive, negative score 추출
        pos_score = merged_model_output.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
        neg_score = merged_model_output.gather(1, neg_idx.unsqueeze(1)).squeeze(1)

        loss = F.relu(self.margin - (pos_score - neg_score)).mean()
        return loss


class DistillListNetLoss(DistillLossBase):
    def __init__(self, temperature: float, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, merged_model_output: torch.Tensor, single_model_output: torch.Tensor):
        # ListNet: 두 분포의 cross entropy
        merged_model_output = merged_model_output / self.temperature
        single_model_output = single_model_output.to(merged_model_output.device) / self.temperature

        P = F.softmax(single_model_output, dim=-1)
        Q = F.log_softmax(merged_model_output, dim=-1)
        loss = -(P * Q).sum(dim=-1).mean()
        return loss


# Backwards compatibility alias
def distill_loss_factory(
    loss_type: LossType,
    temperature: float | None = None,
    **kwargs,
) -> DistillLossBase:
    """
    Factory function to create a distillation loss function.

    Args:
        loss_type: The type of loss function to create.
        temperature: The temperature for KDLoss. If None, it will be ignored.
        margin: The margin for PairwiseLoss. If None, it will be ignored.

    Returns:
        DistillLossBase: The created loss function.
    """
    match loss_type:
        case LossType.CE:
            return DistillCELoss(**kwargs)
        case LossType.KD:
            if temperature is None:
                raise ValueError("Temperature must be provided for KDLoss.")
            return DistillKDLoss(temperature)
        case LossType.MSE:
            return DistillMSELoss(**kwargs)
        case LossType.ADAMERGING:
            return DistillAdaMergingLoss(**kwargs)
        case LossType.ADAMERGING_KD:
            if temperature is None:
                raise ValueError("Temperature must be provided for AdaMergingKDLoss.")
            if "coefficient" not in kwargs:
                raise ValueError("Coefficient must be provided for AdaMergingKDLoss.")
            return DistillAdaMergingKDLoss(temperature, kwargs["coefficient"])
        case LossType.MERGED_PSEUDO_LABEL:
            return MergedPseudoLabelLoss()
        case LossType.MERGED_PSEUDO_LABEL_KD:
            if temperature is None:
                raise ValueError("Temperature must be provided for MergedPseudoLabelKDLoss.")
            if "coefficient" not in kwargs:
                raise ValueError("Coefficient must be provided for MergedPseudoLabelKDLoss.")
            return MergedPseudoLabelKDLoss(temperature, kwargs["coefficient"])
        case LossType.SINGLE_PSEUDO_LABEL:
            return SinglePseudoLabelLoss()
        case LossType.SINGLE_PSEUDO_LABEL_KD:
            if temperature is None:
                raise ValueError("Temperature must be provided for SinglePseudoLabelKDLoss.")
            if "coefficient" not in kwargs:
                raise ValueError("Coefficient must be provided for SinglePseudoLabelKDLoss.")
            return SinglePseudoLabelKDLoss(temperature, kwargs["coefficient"])
        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")
