import torch

from .enums import MetricType


class Evaluator:
    def __init__(self, metrics: list[str], ks: list[int]):
        self.metric_names = metrics
        self.ks = ks
        self._max_k = max(ks)

        self._metrics = []
        for metric in metrics:
            for k in ks:
                self._metrics.append(MetricType[metric].metric_cls(k))

    def evaluate(self, scores: torch.Tensor, labels: torch.Tensor, metric_prefix: str = "") -> dict[str, float]:
        """
        Evaluate the model scores against the ground truth labels.

        Args:
            scores (torch.Tensor): The model scores.
            labels (torch.Tensor): The ground truth labels.
            metric_prefix (str): A prefix for the metric names.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """
        return self(scores, labels, metric_prefix)

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, metric_prefix: str = "") -> dict[str, float]:
        """
        Evaluate the model scores against the ground truth labels.

        Args:
            scores (torch.Tensor): The model scores.
            labels (torch.Tensor): The ground truth labels.
            metric_prefix (str): A prefix for the metric names.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """
        predictions = torch.topk(scores, self._max_k, dim=1).indices

        results = {}
        for metric in self._metrics:
            results[metric_prefix + metric.name] = metric(y_true=labels, y_pred=predictions)

        return results
