from typing import cast

import lightning as L
import torch
import tyro
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from transformers import PreTrainedTokenizer

from rec_retrieval.configs import DistillItemConfig, NegativeSampleConfig
from rec_retrieval.datamodule.distiller import DistillItemDataModule, DistillItemDataModuleForRecformer
from rec_retrieval.evaluator import Evaluator
from rec_retrieval.merger.enums import *
from rec_retrieval.merger.weight_learning import load_merging_module
from rec_retrieval.module.callbacks import (
    SaveWeightsCallback,
    WeightCheckpointCallback,
    MultiDatasetItemEncodingCallback,
)
from rec_retrieval.module.distiller import DistillModule
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender import RecModule
from rec_retrieval.module.recommender.loss_fn import distill_loss_factory
from utils import remove_duplicate_prefix, test_model


def _test_after_train(config, merged_model):
    model = ModelType[config.model_type].value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs,
        tokenizer_kwargs=config.tokenizer_kwargs,
    )
    model.load_state_dict(merged_model.get_state_dict())

    # Test
    module = RecModule(
        model=model,
        evaluator=Evaluator(metrics=config.metric_names, ks=config.ks),
        negative_sample=NegativeSampleConfig(),  # Dummy
        similarity=config.similarity,
    )

    # Use the test_model utility function
    metric_dict, metrics, scores, labels = test_model(
        module=module,
        model_type=ModelType[config.model_type],
        data_paths=config.data_paths,
        model_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        max_attribute_len=config.max_attribute_len,
        max_items=config.max_items,
        num_workers=config.num_workers,
        sequence_prompt=config.sequence_prompt,
        item_prompt=config.item_prompt,
        reverse_sequence=config.reverse_sequence,
        precision=config.precision,
        data_split=config.data_split,
        metrics_path=config.metrics_path,
        predictions_path=config.predictions_path,
    )
    return metric_dict


def _get_data_module(config: DistillItemConfig, tokenizer: PreTrainedTokenizer, item_embeddings: list[torch.Tensor]):
    model_type = ModelType[config.model_type]
    if model_type in (ModelType.RECFORMER, ModelType.RECFORMER_BASE, ModelType.RECFORMER_LARGE):
        return DistillItemDataModuleForRecformer(
            dataset_paths=config.data_paths,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            max_attribute_len=config.max_attribute_len,
            item_embeddings=item_embeddings,
            item_per_dataset=config.item_per_dataset,
            num_workers=config.num_workers,
            valid_ratio=config.valid_ratio,
            item_sample_method=config.item_sample_method,
        )
    else:
        return DistillItemDataModule(
            dataset_paths=config.data_paths,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            max_attribute_len=config.max_attribute_len,
            item_embeddings=item_embeddings,
            num_workers=config.num_workers,
            item_prompt=config.item_prompt,
            item_per_dataset=config.item_per_dataset,
            valid_ratio=config.valid_ratio,
            item_sample_method=config.item_sample_method,
        )


def main():
    config = tyro.cli(DistillItemConfig)
    L.seed_everything(config.seed, workers=True)

    # Load finetune checkpoints
    finetune_state_dicts = []
    score_embeddings = []
    for finetune_checkpoint_path in config.finetune_checkpoint_paths:
        state_dict = torch.load(finetune_checkpoint_path, map_location="cpu")
        item_embeddings = state_dict.pop("item_embeddings")
        state_dict = remove_duplicate_prefix(state_dict)
        finetune_state_dicts.append(state_dict)

        item_embeddings = item_embeddings / item_embeddings.norm(dim=-1, keepdim=True)
        score_embeddings.append(item_embeddings @ item_embeddings.T)

    model = ModelType[config.model_type].value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs.copy(),
        tokenizer_kwargs=config.tokenizer_kwargs.copy(),
    )
    merged_model = load_merging_module(
        merge_type=cast(MergeType, MergeType[config.merge_type]),  # Casting to avoid type errors
        learn_type=cast(LearnType, LearnType[config.learn_type]),
        model=model,
        pretrain_state_dict=model.state_dict(),
        finetune_state_dicts=finetune_state_dicts,
        ignore_keys=set(),
        ties_density=config.ties_density,
        disable_softmax=not config.use_softmax,
        initial_per_weight=config.initial_per_weight,
    )
    module = DistillModule(
        merged_model=merged_model,
        score_embeddings=score_embeddings,
        loss_fn=distill_loss_factory(
            cast(LossType, LossType[config.loss_type]), temperature=config.temperature, **(config.loss_fn_kwargs or {})
        ),
        learning_rate=config.learning_rate,
        similarity=config.similarity,
        trainable_args_kwargs=(
            {
                "freeze_global_weight": True,
                "freeze_global_bias": True,
            }
            if not config.use_softmax
            else {}
        ),
    )
    datamodule = _get_data_module(config, model.tokenizer, item_embeddings)

    logger = WandbLogger(project="RecRetrievalWeightLearning", entity="gen-rec")
    callbacks = [
        save_weights_callback := SaveWeightsCallback(
            version=logger.experiment.id, log_every_steps=len(config.data_paths)
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        MultiDatasetItemEncodingCallback(datamodule.item_dataloaders),
    ]
    weights_checkpoint = None
    if config.valid_ratio is not None:
        callbacks.extend(
            [
                weights_checkpoint := WeightCheckpointCallback(monitor=r"val/loss_epoch/dataloader_idx_\d+"),
            ]
        )

    distill_trainer = L.Trainer(
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=len(config.data_paths),
        deterministic="warn",
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_val_batches=0 if config.valid_ratio is None else None,
        reload_dataloaders_every_n_epochs=1,
    )

    logger.log_hyperparams(vars(config) | {"weights_path": save_weights_callback.save_file})

    distill_trainer.fit(module, datamodule)
    if weights_checkpoint is not None:
        # Load the best weights
        weights_checkpoint.load_weights(module)

    # Log weight file to artifact
    artifact = wandb.Artifact(name=f"trained_weight_{wandb.run.id}", type="weight")
    artifact.add_file(save_weights_callback.save_file)
    wandb.log_artifact(artifact)

    # Test after training
    print("Running test after training...")
    test_metrics = _test_after_train(config, merged_model)

    # Log test metrics to wandb
    wandb.log(test_metrics)

    print(f"Test metrics after training: {test_metrics}")


if __name__ == "__main__":
    main()
