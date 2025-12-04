from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Literal

import lightning as L
import torch
from torch import nn
from transformers import BatchEncoding, get_linear_schedule_with_warmup

from ..models import BaseModel
from ...configs import NegativeSampleConfig
from ...evaluator import Evaluator
from ...types import BatchItem, BatchSequence, BatchSequenceWithNegative, NegativeSampleOption

__all__ = [
    "RecModule",
    "RecJointModule",
]


class RecModuleBase(L.LightningModule, ABC):
    def __init__(
        self,
        model: BaseModel,
        evaluator: Evaluator,
        negative_sample: NegativeSampleConfig,
        similarity: Literal["dot", "cosine"],
        temperature: float = 0.05,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.negative_sample = negative_sample
        self.similarity = similarity

        self.tokenizer = self.model.tokenizer
        self.evaluator = evaluator
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if isinstance(self.warmup_steps, float):
            warmup_steps = self.trainer.estimated_stepping_batches * self.warmup_steps
        elif isinstance(self.warmup_steps, int):
            warmup_steps = self.warmup_steps
        else:
            raise ValueError(f"Invalid warmup_steps type {type(self.warmup_steps)}")

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def _maybe_normalize(self, matrix: torch.Tensor):
        if self.similarity == "cosine":
            return nn.functional.normalize(matrix, p=2, dim=-1)
        return matrix

    def _forward_negative_sample(
        self, sequence_batch: BatchEncoding, target_batch: BatchEncoding, negative_batch: BatchEncoding | None
    ):
        user_encoding = self.model.forward(sequence_batch)  # (batch_size, embedding_dim)
        user_encoding = self._maybe_normalize(user_encoding)

        target_encoding = self.model.forward(target_batch)  # (batch_size, embedding_dim)
        target_encoding = self._maybe_normalize(target_encoding)

        batch_size = len(user_encoding)

        match self.negative_sample.mode:
            case NegativeSampleOption.IN_BATCH:
                labels = torch.arange(batch_size, device=user_encoding.device)
                scores = user_encoding @ target_encoding.T  # (batch_size, batch_size)

            case NegativeSampleOption.SAMPLE:
                assert negative_batch is not None, "negative_batch must not be None in sample mode"

                negative_encoding = self.model.forward(negative_batch)  # (batch_size * num_negative, embedding_dim)
                negative_encoding = self._maybe_normalize(negative_encoding)

                negative_encoding = negative_encoding.reshape(user_encoding.shape[0], self.negative_sample.k, -1)
                # (batch_size, num_negative, embedding_dim)
                all_encoding = torch.cat((target_encoding.unsqueeze(1), negative_encoding), dim=1)
                # (batch_size, 1 + num_negative, embedding_dim)

                labels = torch.zeros(batch_size, dtype=torch.long, device=user_encoding.device)
                scores = torch.bmm(user_encoding.unsqueeze(1), all_encoding.transpose(1, 2)).squeeze(1)
                # (batch_size, 1 + num_negative)

            case NegativeSampleOption.IN_BATCH_SAMPLE:
                assert negative_batch is not None, "negative_batch must not be None in in_batch_sample mode"

                negative_encoding = self.model.forward(negative_batch)  # (batch_size * num_negative, embedding_dim)
                negative_encoding = self._maybe_normalize(negative_encoding)

                negative_encoding = negative_encoding.reshape(user_encoding.shape[0], self.negative_sample.k, -1)
                # (batch_size, num_negative, embedding_dim)

                labels = torch.arange(batch_size, device=user_encoding.device)
                in_batch_scores = user_encoding @ target_encoding.T  # (batch_size, batch_size)
                negative_sample_scores = torch.bmm(
                    user_encoding.unsqueeze(1), negative_encoding.transpose(1, 2)
                ).squeeze(1)
                # (batch_size, num_negative)
                scores = torch.cat((in_batch_scores, negative_sample_scores), dim=1)
                # (batch_size, batch_size + num_negative)

            case _:
                raise ValueError(f"Invalid negative sample mode: {self.negative_sample.mode}")

        return scores, labels

    def _forward_all_negative(self, batch: BatchEncoding, labels: torch.Tensor):
        user_encoding = self.model.forward(batch)  # (batch_size, embedding_dim)
        user_encoding = self._maybe_normalize(user_encoding)

        scores = user_encoding @ self.item_embeddings.T  # (batch_size, num_items)

        return scores, labels, user_encoding

    def _forward_item_encoding(self, batch: BatchEncoding):
        assert "labels" not in batch, "labels must not be in batch when encoding items"

        item_encoding = self.model.forward(batch)
        item_encoding = self._maybe_normalize(item_encoding)

        return item_encoding

    def forward(self, batch: BatchItem | BatchSequence | BatchSequenceWithNegative):
        """
        Forward pass of the model.

        Args:
            batch: The input batch.

        Returns:
            torch.Tensor: The output of the model.
        """
        if isinstance(batch, BatchItem):
            # Item encoding
            return self._forward_item_encoding(batch.items)
        elif isinstance(batch, BatchSequence):
            return self._forward_all_negative(batch.sequence, batch.labels)
        elif isinstance(batch, BatchSequenceWithNegative):
            return self._forward_negative_sample(batch.sequence, batch.target, batch.negatives)
        else:
            raise ValueError(f"Invalid batch type {type(batch)}")

    def training_step(self, batch: BatchSequence, batch_idx: int):
        """
        Training step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            torch.Tensor: The loss of the model.
        """
        output = self.forward(batch)
        if len(output) == 2:
            # In batch negative sampling
            scores, labels = output
        elif len(output) == 3:
            # In batch negative sampling with negative samples
            scores, labels, _ = output
        else:
            raise ValueError(f"Invalid output length {len(output)}")

        loss = nn.functional.cross_entropy(scores / self.temperature, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @abstractmethod
    def on_validation_epoch_start(self):
        """
        Clear the evaluation scores and labels at the start of the validation epoch.

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Log the evaluation metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def on_test_epoch_start(self):
        """
        Clear the evaluation scores and labels at the start of the test epoch.

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Test step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Log the evaluation metrics.
        """
        raise NotImplementedError


class RecModule(RecModuleBase):
    def __init__(
        self,
        model: BaseModel,
        evaluator: Evaluator,
        negative_sample: NegativeSampleConfig,
        similarity: Literal["dot", "cosine"],
        temperature: float = 0.05,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            model=model,
            evaluator=evaluator,
            negative_sample=negative_sample,
            similarity=similarity,
            temperature=temperature,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )

        self.item_embeddings: nn.Parameter | None = None
        self.eval_scores = []
        self.eval_labels = []
        self.eval_user_embeddings = []

    def on_validation_epoch_start(self):
        self.eval_scores = []
        self.eval_labels = []

    def validation_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        scores, labels, _ = self.forward(batch)

        self.eval_scores.append(scores.cpu())
        self.eval_labels.append(labels.cpu())

        loss = nn.functional.cross_entropy(scores / self.temperature, labels)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        self.eval_scores = torch.cat(self.eval_scores, dim=0)
        self.eval_labels = torch.cat(self.eval_labels, dim=0)

        loss = nn.functional.cross_entropy(self.eval_scores / self.temperature, self.eval_labels)

        metrics = self.evaluator.evaluate(self.eval_scores, self.eval_labels, metric_prefix="val/")
        metrics["val/epoch_loss"] = loss.item()

        self.log_dict(metrics, prog_bar=True)

    def on_test_epoch_start(self):
        self.eval_scores = []
        self.eval_labels = []
        self.eval_user_embeddings = []

    def test_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Test step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        scores, labels, user_encoding = self.forward(batch)

        self.eval_scores.append(scores.cpu())
        self.eval_labels.append(labels.cpu())
        self.eval_user_embeddings.append(user_encoding.cpu())

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch.
        """
        self.eval_scores = torch.cat(self.eval_scores, dim=0)
        self.eval_labels = torch.cat(self.eval_labels, dim=0)
        self.eval_user_embeddings = torch.cat(self.eval_user_embeddings, dim=0)

        loss = nn.functional.cross_entropy(self.eval_scores / self.temperature, self.eval_labels)

        metrics = self.evaluator.evaluate(self.eval_scores, self.eval_labels, metric_prefix="test/")
        metrics["test/loss"] = loss.item()

        self.log_dict(metrics, prog_bar=True)


class RecJointModule(RecModuleBase):
    def __init__(
        self,
        model: BaseModel,
        evaluator: Evaluator,
        negative_sample: NegativeSampleConfig,
        similarity: Literal["dot", "cosine"],
        temperature: float = 0.05,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            model=model,
            evaluator=evaluator,
            negative_sample=negative_sample,
            similarity=similarity,
            temperature=temperature,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )

        self.item_embeddings: list[nn.Parameter] | None = None
        self.eval_scores = defaultdict(list)
        self.eval_labels = defaultdict(list)
        self.eval_user_embeddings = defaultdict(list)

    def _forward_all_negative_index(self, batch: BatchEncoding, dataloader_idx: int):
        """
        Forward pass of the model with all negative samples.

        Args:
            batch: The input batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The output of the model.
        """
        user_encoding = self.model.forward(batch)
        user_encoding = self._maybe_normalize(user_encoding)

        scores = user_encoding @ self.item_embeddings[dataloader_idx].T  # (batch_size, num_items)

        return scores, user_encoding

    def on_validation_epoch_start(self):
        self.eval_scores = defaultdict(list)
        self.eval_labels = defaultdict(list)

    def validation_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        labels = batch.labels
        scores, _ = self._forward_all_negative_index(batch.sequence, dataloader_idx)

        self.eval_scores[dataloader_idx].append(scores.cpu())
        self.eval_labels[dataloader_idx].append(labels.cpu())

        loss = nn.functional.cross_entropy(scores / self.temperature, labels)
        metrics = self.evaluator.evaluate(scores, labels, metric_prefix=f"val/")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(labels))
        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        self.eval_scores = {k: torch.cat(eval_scores, dim=0) for k, eval_scores in self.eval_scores.items()}
        self.eval_labels = {k: torch.cat(eval_labels, dim=0) for k, eval_labels in self.eval_labels.items()}

        all_metric = defaultdict(list)
        for k, v in self.trainer.logged_metrics.items():
            if not k.startswith("val/") or k.count("/") != 2:
                continue
            _, metric_name, _ = k.split("/")
            all_metric[metric_name].append(v)

        all_metric = {k: torch.stack(v).mean() for k, v in all_metric.items()}
        all_metric = {f"val/{k}": v for k, v in all_metric.items()}

        self.log_dict(all_metric, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

    def on_test_epoch_start(self):
        self.eval_scores = defaultdict(list)
        self.eval_labels = defaultdict(list)
        self.eval_user_embeddings = defaultdict(list)

    def test_step(self, batch: BatchSequence, batch_idx: int, dataloader_idx: int = 0):
        """
        Test step of the model.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            torch.Tensor: The loss of the model.
        """
        scores, user_encoding = self._forward_all_negative_index(batch.sequence, dataloader_idx)

        self.eval_scores[dataloader_idx].append(scores.cpu())
        self.eval_labels[dataloader_idx].append(batch.labels.cpu())
        self.eval_user_embeddings[dataloader_idx].append(user_encoding.cpu())

        loss = nn.functional.cross_entropy(scores / self.temperature, batch.labels)
        metrics = self.evaluator.evaluate(scores, batch.labels, metric_prefix=f"test/")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(batch.labels))
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(batch.labels))

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch.
        """
        self.eval_scores = {k: torch.cat(eval_scores, dim=0) for k, eval_scores in self.eval_scores.items()}
        self.eval_labels = {k: torch.cat(eval_labels, dim=0) for k, eval_labels in self.eval_labels.items()}
        self.eval_user_embeddings = {
            k: torch.cat(eval_user_embeddings, dim=0) for k, eval_user_embeddings in self.eval_user_embeddings.items()
        }

        all_metric = defaultdict(list)
        for k, v in self.trainer.logged_metrics.items():
            if not k.startswith("test/") or k.count("/") != 2:
                continue
            _, metric_name, _ = k.split("/")
            all_metric[metric_name].append(v)
        all_metric = {k: torch.stack(v).mean() for k, v in all_metric.items()}
        all_metric = {f"test/{k}": v for k, v in all_metric.items()}
        self.log_dict(all_metric, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
