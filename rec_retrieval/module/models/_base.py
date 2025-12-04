from abc import ABC, abstractmethod

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from torch import nn
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer

from ...configs.base import LoRAConfig


class BaseModel(nn.Module, ABC):
    MODEL_CLS = AutoModel
    TOKENIZER_CLS = AutoTokenizer
    DEFAULT_MODEL_PATH = None

    def __init__(
        self,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        lora_config: LoRAConfig | None = None,
        _skip_model_load: bool = False,
        **kwargs,
    ):
        """
        Base class for models.

        Args:
            model_name_or_path (str): Path to the model or model identifier from huggingface.co/models.
            tokenizer_name_or_path (str): Path to the tokenizer or tokenizer identifier from huggingface.co/models.
            model_kwargs (dict, optional): Additional arguments for the model.
            tokenizer_kwargs (dict, optional): Additional arguments for the tokenizer.
            lora_config (dict, optional): Configuration for LoRA.

        Raises:
            ValueError: If MODEL_CLS or TOKENIZER_CLS is not set.
        """
        if model_name_or_path is None and tokenizer_name_or_path is None:
            if self.DEFAULT_MODEL_PATH is None:
                raise ValueError(
                    "model_name_or_path and tokenizer_name_or_path must be provided if DEFAULT_MODEL_PATH is not set."
                )
            model_name_or_path = self.DEFAULT_MODEL_PATH
            tokenizer_name_or_path = self.DEFAULT_MODEL_PATH

        if model_kwargs is None:
            model_kwargs = {}
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        super().__init__()

        self.model_name_or_path = model_name_or_path
        if not _skip_model_load:
            self.model: PreTrainedModel | PeftModel = self.MODEL_CLS.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer: PreTrainedTokenizer = self.TOKENIZER_CLS.from_pretrained(
                tokenizer_name_or_path, **tokenizer_kwargs
            )
            if lora_config is not None:
                self.lora_config = LoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.alpha,
                    lora_dropout=lora_config.dropout,
                    target_modules=lora_config.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, peft_config=self.lora_config)
                print(self.model.print_trainable_parameters())

    @abstractmethod
    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            batch (BatchEncoding): The input batch.

        Returns:
            torch.Tensor: The output of the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
