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
from torch.utils.data import DataLoader

from rec_retrieval.configs import FinetuneJointConfig
from rec_retrieval.datamodule.collator.recommender import JointItemSequenceCollator, RecformerJointItemSequenceCollator
from rec_retrieval.datamodule.dataset import ChainedDataset
from rec_retrieval.datamodule.recommender import RecDataModule, RecDataModuleForRecformer
from rec_retrieval.evaluator import Evaluator
from rec_retrieval.module.callbacks import MultiDatasetItemEncodingCallback, MultiDatasetNegativeItemEncodingCallback
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender.module import RecJointModule

torch.set_float32_matmul_precision("high")


def get_data_module(model_type, data_path, module, config):
    if model_type in (ModelType.RECFORMER, ModelType.RECFORMER_BASE, ModelType.RECFORMER_LARGE):
        datamodule = RecDataModuleForRecformer(
            dataset_path=data_path,
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
            dataset_path=data_path,
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


def _get_collator(model_type, config, item_texts_or_tokenized_items, module):
    if model_type in (ModelType.RECFORMER, ModelType.RECFORMER_BASE, ModelType.RECFORMER_LARGE):
        return RecformerJointItemSequenceCollator(
            bos_token_id=module.model.tokenizer.bos_token_id,
            pad_token_id=module.model.tokenizer.pad_token_id,
            tokenized_items=item_texts_or_tokenized_items,
            max_seq_len=config.max_seq_len,
            num_negative=config.negative_sample.k,
            in_batch_negative=config.negative_sample.in_batch,
        )
    else:
        return JointItemSequenceCollator(
            tokenizer=module.model.tokenizer,
            item_texts=item_texts_or_tokenized_items,
            max_seq_len=config.max_seq_len,
            num_negative=config.negative_sample.k,
            in_batch_negative=config.negative_sample.in_batch,
            sequence_prompt=config.sequence_prompt or "",
            item_prompt=config.item_prompt or "",
            reverse_sequence=config.reverse_sequence,
        )


def main():
    config = tyro.cli(FinetuneJointConfig)
    L.seed_everything(config.seed, workers=True)

    model = ModelType[config.model_type].value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs,
        tokenizer_kwargs=config.tokenizer_kwargs,
    )
    module = RecJointModule(
        model=model,
        evaluator=Evaluator(metrics=config.metric_names, ks=config.ks),
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        negative_sample=config.negative_sample,
        similarity=config.similarity,
    )
    datamodules = []
    item_texts_or_tokenized_items = []
    train_datasets = []
    val_dataloaders = []
    test_dataloaders = []
    for data_path in config.data_paths:
        datamodule = get_data_module(ModelType[config.model_type], data_path, module, config)
        datamodule.setup("fit")
        datamodules.append(datamodule)
        if hasattr(datamodule, "item_text"):
            item_texts_or_tokenized_items.append(datamodule.item_text)
        elif hasattr(datamodule, "tokenized_items"):
            item_texts_or_tokenized_items.append(datamodule.tokenized_items)
        else:
            raise ValueError("No item text or tokenized items found in datamodule.")
        train_datasets.append(datamodule.train_dataset)
        val_dataloaders.append(datamodule.val_dataloader())
        test_dataloaders.append(datamodule.test_dataloader())

    item_dataloaders = [datamodule.item_dataloader() for datamodule in datamodules]
    chained_train_dataset = ChainedDataset(train_datasets)

    trainer = L.Trainer(
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        deterministic="warn",
        callbacks=[
            (
                MultiDatasetItemEncodingCallback(item_dataloaders)
                if config.negative_sample is None
                else MultiDatasetNegativeItemEncodingCallback(item_dataloaders)
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
            project="MergeRecJoint",
            config=config,
        ),
        num_sanity_val_steps=0,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
    )

    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            chained_train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=_get_collator(ModelType[config.model_type], config, item_texts_or_tokenized_items, module),
        ),
        val_dataloaders=val_dataloaders,
    )
    trainer.test(module, dataloaders=test_dataloaders, ckpt_path="best")


if __name__ == "__main__":
    main()
