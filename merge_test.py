from typing import cast

import torch
import tyro
from tqdm import tqdm

from rec_retrieval.configs import TestMergeConfig, NegativeSampleConfig
from rec_retrieval.evaluator import Evaluator
from rec_retrieval.merger.enums import *
from rec_retrieval.merger.weight_learning import load_merging_module
from rec_retrieval.module.models import ModelType
from rec_retrieval.module.recommender import RecModule
from utils import remove_duplicate_prefix, test_model


def main():
    config = tyro.cli(TestMergeConfig)

    # Load finetune checkpoints
    finetune_state_dicts = []
    for finetune_checkpoint_path in tqdm(config.finetune_checkpoint_paths, desc="Loading finetune checkpoints"):
        state_dict = torch.load(finetune_checkpoint_path, map_location="cpu")
        state_dict.pop("item_embeddings")
        state_dict = remove_duplicate_prefix(state_dict)
        finetune_state_dicts.append(state_dict)

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
    )

    # Load weights
    if config.weight_file.name == "average":
        weights = {
            "global_weights": {"all": [1.0]},
            "global_biases": {"all": [0.0]},
            "per_weights": {
                "all": [1.0 / len(finetune_state_dicts)] * len(finetune_state_dicts),
            }
        }
        print(f"Using average weights for {len(finetune_state_dicts)} models.")
    elif config.weight_file.name == "uniform":
        weight = config.weight_file_line
        weights = {
            "global_weights": {"all": [1.0]},
            "global_biases": {"all": [0.0]},
            "per_weights": {
                "all": [config.weight_file_line] * len(finetune_state_dicts)
            }
        }
        print(f"Using uniform weights for {len(finetune_state_dicts)} models: {weight}.")
    else:
        weights = [eval(line) for line in config.weight_file.read_text().strip().splitlines()]
        weights = weights[config.weight_file_line]["weights"]
    merged_model.load_weights_from_dict(weights)

    state_dict = {k: v.detach() for k, v in merged_model.get_state_dict().items()}
    model = ModelType[config.model_type].value(
        model_name_or_path=config.model_path,
        tokenizer_name_or_path=config.tokenizer_path,
        lora_config=(None if config.lora is not None and not config.lora.enable else config.lora),
        pooling_method=config.pooling_method,
        model_kwargs=config.model_kwargs,
        tokenizer_kwargs=config.tokenizer_kwargs,
    )
    model.load_state_dict(state_dict)

    # Test
    module = RecModule(
        model=model,
        evaluator=Evaluator(metrics=config.metric_names, ks=config.ks),
        negative_sample=NegativeSampleConfig(),
        similarity=config.similarity,
    )

    # Use the test_model utility function
    _, metrics, scores, labels = test_model(
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
        data_split=config.test_data_split,
        metrics_path=config.metrics_path,
        predictions_path=config.predictions_path,
        item_embeddings_path=config.item_embeddings_path,
        user_embeddings_path=config.user_embeddings_path,
    )


if __name__ == "__main__":
    main()
