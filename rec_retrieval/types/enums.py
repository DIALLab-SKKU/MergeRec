from enum import Enum


class NegativeSampleOption(Enum):
    FULL = "FULL"
    IN_BATCH = "IN_BATCH"
    SAMPLE = "SAMPLE"
    IN_BATCH_SAMPLE = "IN_BATCH_SAMPLE"
