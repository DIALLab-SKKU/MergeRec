import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rec_retrieval.module.distiller import DistillModule
from .recommender import RecModule


class ItemEncoderMixin:
    @staticmethod
    @torch.no_grad()
    def encode_items(item_dataloader: DataLoader, pl_module: RecModule | DistillModule) -> torch.Tensor:
        assert isinstance(item_dataloader, DataLoader), "item_dataloader must be a DataLoader instance."

        train_status = pl_module.training
        pl_module.eval()

        item_embeddings = []

        for batch in tqdm(item_dataloader, desc="Encoding items"):
            output = pl_module.forward(batch.to(pl_module.device))
            item_embeddings.append(output)

        # Concatenate all item embeddings
        item_embeddings = torch.cat(item_embeddings, dim=0)

        pl_module.train(train_status)

        return item_embeddings

    def inject_item_embeddings(self, item_dataloader: DataLoader, pl_module: RecModule, requires_grad: bool = False):
        """
        Inject the item embeddings into the model.

        :param item_dataloader: The DataLoader containing the item data.
        :param pl_module: The RecModule instance.
        :param requires_grad: Whether the item embeddings should require gradients.
        """
        pl_module.item_embeddings = nn.Parameter(
            self.encode_items(item_dataloader, pl_module), requires_grad=requires_grad
        )


class ItemEncodingCallback(Callback, ItemEncoderMixin):
    def __init__(self, item_dataloader: DataLoader | None = None):
        self.item_dataloader = item_dataloader

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: RecModule):
        print(f"[Train - epoch {trainer.current_epoch} start] Encoding items.")
        self.inject_item_embeddings(self.item_dataloader, pl_module)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: RecModule):
        if pl_module.item_embeddings is None:
            print("[Test - epoch start] Encoding items as no item embeddings are found.")
            self.inject_item_embeddings(self.item_dataloader, pl_module)


class ItemEncodingNegativeSampleCallback(Callback, ItemEncoderMixin):
    def __init__(self, item_dataloader: DataLoader | None = None):
        self.item_dataloader = item_dataloader

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: RecModule):
        print(f"[Validation - epoch {trainer.current_epoch} start] Encoding items as no item embeddings are found.")
        self.inject_item_embeddings(self.item_dataloader, pl_module)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: RecModule):
        if pl_module.item_embeddings is None:
            print("[Test - epoch start] Encoding items as no item embeddings are found.")
            self.inject_item_embeddings(self.item_dataloader, pl_module)


class MultiDatasetItemEncodingCallback(Callback, ItemEncoderMixin):
    def __init__(self, item_dataloaders: list[DataLoader]):
        self.item_dataloaders = item_dataloaders

    def inject_item_embeddings(
        self, item_dataloaders: list[DataLoader], pl_module: DistillModule, requires_grad: bool = False
    ):
        if pl_module.item_embeddings is not None:
            print("Item embeddings already exist in the model. Skipping encoding.")
            return

        item_embeddings = []

        for idx, item_dataloader in enumerate(item_dataloaders, start=1):
            print(f"Encoding {idx} / {len(item_dataloaders)} datasets.")
            item_embeddings.append(
                nn.Parameter(self.encode_items(item_dataloader, pl_module), requires_grad=requires_grad)
            )

        pl_module.item_embeddings = nn.ParameterList(item_embeddings)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: DistillModule):
        print(f"[Train - epoch {trainer.current_epoch} start] Encoding items.")
        self.inject_item_embeddings(self.item_dataloaders, pl_module)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: DistillModule):
        if pl_module.item_embeddings is None:
            print("[Test - epoch start] Encoding items as no item embeddings are found.")
            self.inject_item_embeddings(self.item_dataloaders, pl_module)


class MultiDatasetNegativeItemEncodingCallback(Callback, ItemEncoderMixin):
    def __init__(self, item_dataloaders: list[DataLoader]):
        self.item_dataloaders = item_dataloaders

    def inject_item_embeddings(
        self, item_dataloaders: list[DataLoader], pl_module: DistillModule, requires_grad: bool = False
    ):
        item_embeddings = []

        for idx, item_dataloader in enumerate(item_dataloaders, start=1):
            print(f"Encoding {idx} / {len(item_dataloaders)} datasets.")
            item_embeddings.append(
                nn.Parameter(self.encode_items(item_dataloader, pl_module), requires_grad=requires_grad)
            )

        pl_module.item_embeddings = nn.ParameterList(item_embeddings)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: DistillModule):
        print(f"[Validation - epoch {trainer.current_epoch} start] Encoding items.")
        self.inject_item_embeddings(self.item_dataloaders, pl_module)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: DistillModule):
        if pl_module.item_embeddings is None:
            print("[Test - epoch start] Encoding items as no item embeddings are found.")
            self.inject_item_embeddings(self.item_dataloaders, pl_module)


class SaveWeightsCallback(Callback):
    def __init__(self, version: str | None = None, save_dir: str | Path = "weights", log_every_steps: int = 5):
        if version is None:
            version = str(uuid4())[:8]

        self.version = version
        self.save_dir = Path(save_dir)
        self.save_file = self.save_dir / f"{version}.jsonl"
        self.log_every_steps = log_every_steps

        if not self.save_dir.exists():
            print(f"{self.__class__.__name__}: Creating directory {self.save_dir.absolute()}.")
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self._file_handler = open(self.save_file, "w", encoding="utf-8")

        print(f"{self.__class__.__name__}: Weights will be saved to {self.save_file.absolute()}.")

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: DistillModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx % self.log_every_steps == 0:
            line = {
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "weights": pl_module.merged_model.serialize_weights(),
            }

            self._file_handler.write(f"{line}\n")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: DistillModule):
        self._file_handler.flush()

    def teardown(self, trainer: L.Trainer, pl_module: DistillModule, stage: str):
        if self._file_handler:
            self._file_handler.close()


class WeightCheckpointCallback(Callback):
    def __init__(self, monitor: str = "val/loss"):
        self.monitor = monitor  # Regex pattern to match the metric name
        self.best_score = float("inf")
        self.best_weights = None

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: DistillModule):
        scores = []
        for callback_metric in trainer.callback_metrics:
            # If the metric name matches the monitor pattern, add it to the scores
            if re.fullmatch(self.monitor, callback_metric):
                scores.append(trainer.callback_metrics[callback_metric].item())

        if len(scores) == 0:
            raise RuntimeError(f"No metrics found matching the monitor pattern: {self.monitor}")

        current_score = sum(scores) / len(scores)

        if current_score < self.best_score:
            print(f"New best score: {current_score}. Saving weights.")
            self.best_score = current_score
            self.best_weights = pl_module.merged_model.serialize_weights()

    def load_weights(self, pl_module: DistillModule):
        if self.best_weights is not None:
            pl_module.merged_model.load_weights_from_dict(self.best_weights)
            print("Weights loaded from the best checkpoint.")
        else:
            print("No best weights found. Skipping loading.")
