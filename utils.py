from pathlib import Path
from typing import List, Optional

import lightning as L
import pandas as pd
import torch
from tqdm import tqdm

from rec_retrieval.configs import NegativeSampleConfig
from rec_retrieval.datamodule.recommender import RecDataModule, RecDataModuleForRecformer
from rec_retrieval.merger.types import StateDict
from rec_retrieval.module.callbacks import ItemEncodingCallback
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender import RecModule


def remove_duplicate_prefix(state_dict: StateDict) -> StateDict:
    """
    Remove duplicate prefixes from state_dict keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "", 1)] = v
        else:
            print(f"Keeping key {k} without model. prefix")
            new_state_dict[k] = v

    return new_state_dict


def test_model(
    module: RecModule,
    model_type: ModelType,
    data_paths: List[Path],
    model_tokenizer,
    batch_size: int,
    max_seq_len: int,
    max_attribute_len: int,
    max_items: Optional[int],
    num_workers: int,
    sequence_prompt: str,
    item_prompt: str,
    reverse_sequence: bool,
    precision: str,
    data_split: str,
    metrics_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    item_embeddings_path: Optional[Path] = None,
    user_embeddings_path: Optional[Path] = None,
) -> tuple:
    """
    Test the model on the given datasets and return metrics, scores, and labels.
    """
    negative_sample_config = NegativeSampleConfig()

    datamodules = []
    item_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    for data_path in tqdm(data_paths, desc="Loading datasets", ncols=120):
        datamodule = _get_data_module(
            model_type=model_type,
            batch_size=batch_size,
            data_path=data_path,
            item_prompt=item_prompt,
            max_attribute_len=max_attribute_len,
            max_items=max_items,
            max_seq_len=max_seq_len,
            model_tokenizer=model_tokenizer,
            negative_sample_config=negative_sample_config,
            num_workers=num_workers,
            reverse_sequence=reverse_sequence,
            sequence_prompt=sequence_prompt,
        )
        datamodule.setup("fit")
        datamodules.append(datamodule)
        item_dataloaders.append(datamodule.item_dataloader())
        val_dataloaders.append(datamodule.val_dataloader())
        test_dataloaders.append(datamodule.test_dataloader())

    rec_trainer = L.Trainer(
        precision=precision,
        deterministic="warn",
        callbacks=[
            encoding_callback := ItemEncodingCallback(),
        ],
        logger=None,
        num_sanity_val_steps=0,
    )

    # Test the model
    if data_split == "val":
        dataloaders = val_dataloaders
    elif data_split == "test":
        dataloaders = test_dataloaders
    else:
        raise ValueError(f"Unknown data split: {data_split}")

    metric_dict = {}
    metrics = []
    scores = []
    labels = []
    item_embeddings = []
    user_embeddings = []

    for i, (item_dataloader, sequence_dataloader) in enumerate(zip(item_dataloaders, dataloaders)):
        encoding_callback.item_dataloader = item_dataloader
        module.item_embeddings = None
        metric = rec_trainer.test(module, sequence_dataloader, verbose=False)

        scores.append(module.eval_scores.detach().cpu().clone())
        labels.append(module.eval_labels.detach().cpu().clone())
        item_embeddings.append(module.item_embeddings.detach().cpu().clone())
        user_embeddings.append(module.eval_user_embeddings.detach().cpu().clone())
        metrics.append(metric[0])

        metric_dict |= {f"test/dataset_{i}/{k}": v for k, v in metric[0].items()}

    save_predictions(
        data_paths,
        item_embeddings,
        item_embeddings_path,
        labels,
        metrics,
        metrics_path,
        predictions_path,
        scores,
        user_embeddings,
        user_embeddings_path,
    )

    return metric_dict, metrics, scores, labels


def _get_data_module(
    model_type,
    batch_size,
    data_path,
    item_prompt,
    max_attribute_len,
    max_items,
    max_seq_len,
    model_tokenizer,
    negative_sample_config,
    num_workers,
    reverse_sequence,
    sequence_prompt,
):
    if model_type in (ModelType.RECFORMER, ModelType.RECFORMER_BASE, ModelType.RECFORMER_LARGE):
        return RecDataModuleForRecformer(
            dataset_path=data_path,
            tokenizer=model_tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_attribute_len=max_attribute_len,
            max_items=max_items,
            num_workers=num_workers,
            negative_sample=negative_sample_config,
        )
    else:
        return RecDataModule(
            dataset_path=data_path,
            tokenizer=model_tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_attribute_len=max_attribute_len,
            max_items=max_items,
            num_workers=num_workers,
            negative_sample=negative_sample_config,
            sequence_prompt=sequence_prompt,
            item_prompt=item_prompt,
            reverse_sequence=reverse_sequence,
        )


def save_predictions(
    data_paths: list[Path],
    item_embeddings: list[torch.Tensor],
    item_embeddings_path: Optional[Path],
    labels: list[torch.Tensor],
    metrics: list[dict],
    metrics_path: Optional[Path],
    predictions_path: Optional[Path],
    scores: list[torch.Tensor],
    user_embeddings: list[torch.Tensor],
    user_embeddings_path: Optional[Path],
):

    if metrics_path is not None:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.index = [data_path.name for data_path in data_paths]
        metrics_df.index.name = "dataset"
        metrics_df.to_csv(metrics_path)
        print(f"Saved metrics to {metrics_path}")

    if predictions_path is not None:
        predictions = {}
        for data_path, score, label in zip(data_paths, scores, labels):
            predictions[data_path.name] = {
                "scores": score,
                "labels": label,
            }
        torch.save(predictions, predictions_path)
        print(f"Saved predictions to {predictions_path}")

    if item_embeddings_path is not None:
        torch.save(item_embeddings, item_embeddings_path)
        print(f"Saved item embeddings to {item_embeddings_path}")

    if user_embeddings_path is not None:
        torch.save(user_embeddings, user_embeddings_path)
        print(f"Saved user embeddings to {user_embeddings_path}")
