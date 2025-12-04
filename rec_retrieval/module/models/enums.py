from enum import Enum

from .decoder.llama import Llama
from .decoder.mistral import Mistral
from .encoder.bert import BERT
from .encoder.blair import BLaIR, BLaIRBase, BLaIRLarge
from .encoder.longformer import Longformer
from .encoder.recformer import Recformer, RecformerBase, RecformerLarge
from .encoder.roberta import RoBERTa


class ModelType(Enum):
    BERT = BERT
    LONGFORMER = Longformer
    ROBERTA = RoBERTa
    BLAIR = BLaIR
    BLAIR_BASE = BLaIRBase
    BLAIR_LARGE = BLaIRLarge
    RECFORMER = Recformer
    RECFORMER_BASE = RecformerBase
    RECFORMER_LARGE = RecformerLarge

    LLAMA = Llama
    MISTRAL = Mistral
