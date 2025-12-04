from typing import Literal

import lightning as L
import torch
from torch import nn

from ...recommender.loss_fn import DistillLossBase
from ....merger.weight_learning import TaskVectorMergingModuleBase
from ....types.model_batch import BatchSequenceWithNegative, BatchDistillationSequence, BatchItem

__all__ = [
    "DistillSequenceModule",
]


class DistillSequenceModule(L.LightningModule):
    def __init__(
        self,
        merged_model: TaskVectorMergingModuleBase,
        score_embeddings: list[torch.Tensor],
        loss_fn: DistillLossBase,
        similarity: Literal["dot", "cosine"],
        learning_rate: float = 5e-5,
        trainable_args_kwargs: dict | None = None,
    ):
        super().__init__()
        self.merged_model = merged_model
        self.loss_fn = loss_fn
        self.similarity = similarity
        self.learning_rate = learning_rate
        self.trainable_args_kwargs = trainable_args_kwargs or {}

        self.score_embeddings = score_embeddings
        self.item_embeddings = None

        self._valid_metrics = []

    def _maybe_normalize(self, matrix: torch.Tensor):
        if self.similarity == "cosine":
            return nn.functional.normalize(matrix, p=2, dim=-1)
        return matrix

    def forward(self, batch: BatchSequenceWithNegative | BatchDistillationSequence):
        if isinstance(batch, BatchSequenceWithNegative):
            return self._forward_sequence_encoding(batch.sequence)
        elif isinstance(batch, BatchDistillationSequence):
            return self._forward_distill(batch)
        elif isinstance(batch, BatchItem):
            return self._forward_sequence_encoding(batch.items)  # For item encoding
        else:
            raise ValueError(f"Invalid batch type {type(batch)}")

    def _forward_sequence_encoding(self, sequence_batch):
        encoding = self.merged_model.forward(sequence_batch)
        encoding = self._maybe_normalize(encoding)

        return encoding

    def _forward_distill(self, batch: BatchDistillationSequence):
        merged_model_representations = self._forward_sequence_encoding(batch.sequence)  # (batch_size, dim)

        losses = []
        for i, (dataset_index, sequence_id) in enumerate(zip(batch.dataset_indexes, batch.sequence_ids)):
            merged_model_logit = (
                merged_model_representations[i] @ self.item_embeddings[dataset_index].T  # (num_items, )
            )
            single_model_logit = self.score_embeddings[dataset_index][sequence_id]  # (num_items, )

            loss = self.loss_fn(merged_model_logit.unsqueeze(0), single_model_logit.unsqueeze(0))
            losses.append(loss)

        loss = torch.stack(losses).mean()  # Average loss across all sequences in the batch

        return loss

    def training_step(self, batch: BatchDistillationSequence, batch_idx: int):
        loss = self._forward_distill(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics = []

    def validation_step(self, batch: BatchDistillationSequence, batch_idx: int, dataloader_idx: int = 0):
        loss = self._forward_distill(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self._valid_metrics.append(loss.item())
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/average_loss_epoch", torch.tensor(self._valid_metrics).mean(), prog_bar=True)
        self._valid_metrics = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.merged_model.trainable_parameters(**self.trainable_args_kwargs),
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        return optimizer
