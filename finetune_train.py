import lightning as L
import torch
import tyro
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichModelSummary,
)
from lightning.pytorch.loggers import WandbLogger

from rec_retrieval.configs import FinetuneSingleConfig
from rec_retrieval.datamodule.recommender import RecDataModule, RecDataModuleForRecformer
from rec_retrieval.evaluator import Evaluator
from rec_retrieval.module.callbacks import ItemEncodingCallback, ItemEncodingNegativeSampleCallback
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender import RecModule

torch.set_float32_matmul_precision("high")


def get_data_module(model_type, module, config):
    if model_type in (ModelType.RECFORMER, ModelType.RECFORMER_BASE, ModelType.RECFORMER_LARGE):
        datamodule = RecDataModuleForRecformer(
            dataset_path=config.data_path,
            tokenizer=module.model.tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            max_attribute_len=config.max_attribute_len,
            max_items=config.max_items,
            negative_sample=config.negative_sample,
            num_workers=config.num_workers,
        )
    else:
        datamodule = RecDataModule(
            dataset_path=config.data_path,
            tokenizer=module.model.tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            max_attribute_len=config.max_attribute_len,
            max_items=config.max_items,
            negative_sample=config.negative_sample,
            num_workers=config.num_workers,
            sequence_prompt=config.sequence_prompt,
            item_prompt=config.item_prompt,
            reverse_sequence=config.reverse_sequence,
        )

    return datamodule


def main():
    config = tyro.cli(FinetuneSingleConfig)
    L.seed_everything(config.seed, workers=True)

    model_type = ModelType[config.model_type]
    model = model_type.value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs,
        tokenizer_kwargs=config.tokenizer_kwargs,
    )
    module = RecModule(
        model=model,
        evaluator=Evaluator(metrics=config.metric_names, ks=config.ks),
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        negative_sample=config.negative_sample,
        similarity=config.similarity,
    )
    datamodule = get_data_module(model_type, module, config)
    datamodule.setup("fit")

    trainer = L.Trainer(
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        deterministic="warn",
        callbacks=[
            (
                ItemEncodingNegativeSampleCallback(datamodule.item_dataloader())
                if config.negative_sample
                else ItemEncodingCallback(datamodule.item_dataloader())
            ),
            ModelCheckpoint(
                monitor=config.valid_metric,
                mode="max",
                filename="epoch_{epoch:02d}",
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(
                monitor=config.valid_metric,
                patience=config.patience,
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
            RichModelSummary(max_depth=2),
        ],
        logger=WandbLogger(
            project="MergeRecFineTune",
            config=config,
        ),
        num_sanity_val_steps=0,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
    )

    trainer.fit(module, datamodule)
    trainer.test(module, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
