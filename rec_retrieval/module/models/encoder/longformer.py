from typing import Literal

import torch
from transformers import BatchEncoding, LongformerConfig, LongformerModel, LongformerTokenizer

from ._base import BaseEncoderModel
from ....configs import LoRAConfig


class Longformer(BaseEncoderModel):
    MODEL_CLS = LongformerModel
    TOKENIZER_CLS = LongformerTokenizer
    DEFAULT_MODEL_PATH = "allenai/longformer-base-4096"

    def __init__(
        self,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        lora_config: LoRAConfig | None = None,
        pooling_method: Literal["cls", "mean"] = "cls",
    ):
        config = LongformerConfig.from_pretrained(model_name_or_path or self.DEFAULT_MODEL_PATH)
        config.attention_window = [64] * config.num_hidden_layers

        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs=model_kwargs | {"config": config} if model_kwargs else {"config": config},
            tokenizer_kwargs=tokenizer_kwargs,
            lora_config=lora_config,
        )

        self.pooling_method = pooling_method

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        # Ensure that the input is a BatchEncoding object
        if not isinstance(batch, BatchEncoding):
            raise TypeError("Input must be a BatchEncoding object.")

        global_attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)  # (batch_size, seq_len)
        global_attention_mask[:, 0] = 1  # Set the first token (CLS token) to have global attention
        batch["global_attention_mask"] = global_attention_mask

        outputs = self.model(**batch)

        return self.pool(outputs)
