from typing import Literal

import torch
from transformers import BatchEncoding

from .._base import BaseModel
from ....configs import LoRAConfig


class BaseDecoderModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        lora_config: LoRAConfig | None = None,
        pooling_method: Literal["last"] = "last",
    ):
        model_kwargs = model_kwargs or {}
        model_kwargs["torch_dtype"] = torch.bfloat16
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            lora_config=lora_config,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pooling_method = pooling_method

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        # Ensure that the input is a BatchEncoding object
        if not isinstance(batch, BatchEncoding):
            raise TypeError("Input must be a BatchEncoding object.")

        pad_token_count = torch.eq(batch["input_ids"], self.tokenizer.pad_token_id).sum(dim=1)
        sequence_count = batch["input_ids"].shape[-1] - pad_token_count

        outputs = self.model(**batch)

        return self.pool(outputs, sequence_count)

    def pool(self, outputs, sequence_count) -> torch.Tensor:
        batch_size = outputs.last_hidden_state.size(0)

        if self.pooling_method == "last":
            return outputs.last_hidden_state[torch.arange(batch_size), sequence_count - 1, :]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}.")
