from enum import Enum
from typing import Type

from .metrics import Recall, NDCG, BaseMetric


class MetricType(Enum):
    RECALL = ("RECALL", Recall)
    NDCG = ("NDCG", NDCG)

    def __init__(self, metric_name: str, metric_cls: Type[BaseMetric]):
        self.metric_name = metric_name
        self.metric_cls = metric_cls
