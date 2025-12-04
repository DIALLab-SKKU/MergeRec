from typing import Literal

import torch
from transformers import BatchEncoding

from .._base import BaseModel
from ....configs import LoRAConfig


class BaseEncoderModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        lora_config: LoRAConfig | None = None,
        pooling_method: Literal["cls", "mean"] = "cls",
        _skip_model_load: bool = False,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            lora_config=lora_config,
            _skip_model_load=_skip_model_load,
        )

        self.pooling_method = pooling_method

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        # Ensure that the input is a BatchEncoding object
        if not isinstance(batch, BatchEncoding):
            raise TypeError("Input must be a BatchEncoding object.")

        outputs = self.model(**batch)

        return self.pool(outputs)

    def pool(self, outputs) -> torch.Tensor:
        if self.pooling_method == "mean":
            return outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
        elif self.pooling_method == "cls":
            return outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        elif self.pooling_method == "pooler":
            return outputs.pooler_output
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}.")
