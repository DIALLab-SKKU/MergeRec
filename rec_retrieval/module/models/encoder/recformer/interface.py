from typing import Literal

import torch
from transformers import BatchEncoding
from typing_extensions import deprecated

from .models import RecformerModel, RecformerConfig
from .tokenization import RecformerTokenizer
from .._base import BaseEncoderModel
from .....configs import LoRAConfig


class BaseRecformerModel(BaseEncoderModel):
    MODEL_CLS = RecformerModel
    TOKENIZER_CLS = RecformerTokenizer

    @staticmethod
    def _load_config(model_path):
        config = RecformerConfig.from_pretrained(model_path)
        config.max_attr_num = 3
        config.max_attr_length = 32
        config.max_item_embeddings = 51
        config.attention_window = [64] * 12

        return config

    def __init__(
        self,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        lora_config: LoRAConfig | None = None,
        pooling_method: Literal["cls", "mean"] = "cls",
    ):
        config = self._load_config(model_name_or_path or self.DEFAULT_MODEL_PATH)

        if not isinstance(model_kwargs, dict) or "ckpt_path" not in model_kwargs:
            raise ValueError(f"{self.__class__.__name__} requires a 'ckpt_path' in model_kwargs.")

        ckpt_path = model_kwargs.pop("ckpt_path")

        # Initialize the base model
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs=model_kwargs | {"config": config} if model_kwargs else {"config": config},
            tokenizer_kwargs=tokenizer_kwargs,
            lora_config=lora_config,
            _skip_model_load=True,
        )

        if lora_config is not None:
            raise ValueError(
                f"LoRA is not supported for {self.__class__.__name__}. Please set lora_config.enable=False."
            )

        self.model = RecformerModel(config)
        self.tokenizer = RecformerTokenizer.from_pretrained(tokenizer_name_or_path or self.DEFAULT_MODEL_PATH)

        # Load the state dict from the checkpoint
        print("Loading model state from checkpoint:", ckpt_path)
        print(self.model.load_state_dict(torch.load(ckpt_path), strict=False))

        self.pooling_method = pooling_method

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        if not isinstance(batch, BatchEncoding):
            raise TypeError("Input must be a BatchEncoding object.")

        required_keys = ["input_ids", "attention_mask", "token_type_ids", "item_position_ids"]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key in batch: {key}")

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            global_attention_mask=batch["global_attention_mask"],
            token_type_ids=batch["token_type_ids"],
            item_position_ids=batch["item_position_ids"],
        )

        return self.pool(outputs)


class RecformerBase(BaseRecformerModel):
    DEFAULT_MODEL_PATH = "allenai/longformer-base-4096"

    @staticmethod
    def _load_config(model_path):
        config = RecformerConfig.from_pretrained(model_path)
        config.max_attr_num = 3
        config.max_attr_length = 32
        config.max_item_embeddings = 51
        config.attention_window = [64] * 12

        return config


# Backwards compatibility
@deprecated("Recformer is deprecated, use RecformerBase instead.")
class Recformer(RecformerBase):
    """Alias for RecformerBase to maintain compatibility with existing code."""

    pass


class RecformerLarge(BaseRecformerModel):
    DEFAULT_MODEL_PATH = "allenai/longformer-large-4096"

    @staticmethod
    def _load_config(model_path):
        config = RecformerConfig.from_pretrained(model_path)
        config.max_attr_num = 3
        config.max_attr_length = 32
        config.max_item_embeddings = 51
        config.attention_window = [64] * 24

        return config
