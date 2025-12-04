from typing_extensions import deprecated

from ._base import BaseEncoderModel


@deprecated("BLaIR is deprecated, use BLaIRBase instead.")
class BLaIR(BaseEncoderModel):
    DEFAULT_MODEL_PATH = "hyp1231/blair-roberta-base"


class BLaIRBase(BaseEncoderModel):
    DEFAULT_MODEL_PATH = "hyp1231/blair-roberta-base"


class BLaIRLarge(BaseEncoderModel):
    DEFAULT_MODEL_PATH = "hyp1231/blair-roberta-large"
