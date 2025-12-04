import torch
import tyro

from rec_retrieval.configs import NegativeSampleConfig, TestSingleConfig
from rec_retrieval.evaluator import Evaluator
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender import RecModule
from utils import remove_duplicate_prefix, test_model


def main():
    config = tyro.cli(TestSingleConfig)

    negative_sample_config = NegativeSampleConfig()
    model = ModelType[config.model_type].value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs.copy(),
        tokenizer_kwargs=config.tokenizer_kwargs.copy(),
    )
    module = RecModule(
        model=model,
        evaluator=Evaluator(metrics=config.metric_names, ks=config.ks),
        negative_sample=negative_sample_config,
        similarity=config.similarity,
    )

    # Load checkpoint
    finetune_state_dict = torch.load(config.finetune_checkpoint_path, map_location="cpu")
    finetune_state_dict.pop("item_embeddings")
    finetune_state_dict = remove_duplicate_prefix(finetune_state_dict)
    model.load_state_dict(finetune_state_dict)

    test_model(
        module=module,
        model_type=ModelType[config.model_type],
        data_paths=[config.data_path],
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
        item_embeddings_path=config.item_embeddings_path,
        user_embeddings_path=config.user_embeddings_path,
    )


if __name__ == "__main__":
    main()
