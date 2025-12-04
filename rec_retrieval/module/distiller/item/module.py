from typing import Literal

import lightning as L
import torch
from torch import nn
from transformers import BatchEncoding

from ...models import BaseModel
from ...recommender.loss_fn import DistillLossBase
from ....merger.weight_learning import TaskVectorMergingModuleBase
from ....types import BatchItem, BatchSequence, BatchSequenceWithNegative, BatchDistillationItem

__all__ = [
    "DistillModule",
]


class DistillModule(L.LightningModule):
    def __init__(
        self,
        merged_model: TaskVectorMergingModuleBase,
        score_embeddings: list[torch.Tensor],
        loss_fn: DistillLossBase,
        similarity: Literal["dot", "cosine"],
        learning_rate: float = 5e-5,
        trainable_args_kwargs: dict | None = None,
    ):
        """
        RecRetrievalModule is a LightningModule that wraps a BaseModel for training and evaluation.

        Args:
            model_type (BaseModel): The model class to be used.
            model_kwargs (dict, optional): Additional arguments for the model.
        """
        super().__init__()
        self.merged_model = merged_model
        self.loss_fn = loss_fn
        self.similarity = similarity
        self.learning_rate = learning_rate
        self.trainable_args_kwargs = trainable_args_kwargs or {}

        self.score_embeddings = score_embeddings
        self.item_embeddings = None

        self.eval_scores = []
        self.eval_labels = []

    def _maybe_normalize(self, matrix: torch.Tensor):
        if self.similarity == "cosine":
            return nn.functional.normalize(matrix, p=2, dim=-1)
        return matrix

    def forward(self, batch: BatchItem | BatchSequenceWithNegative):
        """
        Forward pass of the model.

        Args:
            batch: The input batch.

        Returns:
            torch.Tensor: The output of the model.
        """
        if isinstance(batch, BatchItem):
            return self._forward_item_encoding(batch.items)
        elif isinstance(batch, BatchDistillationItem):
            return self._forward_distill(batch)
        else:
            raise ValueError(f"Invalid batch type {type(batch)}")

    def _forward_item_encoding(self, batch: BatchEncoding) -> torch.Tensor:
        """
        Forward pass for items.

        Args:
            batch: The input batch.

        Returns:
            BatchSequence: The output of the model.
        """
        encoding = self.merged_model.forward(batch)
        encoding = self._maybe_normalize(encoding)

        return encoding

    def _forward_distill(self, batch: BatchDistillationItem):
        # We assume all batches belong to a single dataset
        merged_model_representations = self._forward_item_encoding(batch.items)  # (batch_size, dim)

        losses = []
        for i, (dataset_index, item_id) in enumerate(zip(batch.dataset_indexes, batch.item_ids)):
            merged_model_logit = (
                merged_model_representations[i] @ self.item_embeddings[dataset_index].T  # (num_items, )
            )
            single_model_logit = self.score_embeddings[dataset_index][item_id]  # (num_items, )

            loss = self.loss_fn(merged_model_logit.unsqueeze(0), single_model_logit.unsqueeze(0))
            losses.append(loss)

        loss = torch.stack(losses).mean()  # Average loss across all items in the batch

        return loss

    def training_step(self, batch: BatchDistillationItem, batch_idx: int):
        """
        Training step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        loss = self._forward_distill(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics = []

    def validation_step(self, batch: BatchDistillationItem, batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
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
