import torch


class BaseMetric:
    METRIC_NAME = None

    def __init__(self, k: int):
        super().__init__()

        self.k = k

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute the metric.

        Args:
            y_true (torch.Tensor): The ground truth labels (N, )
            y_pred (torch.Tensor): The predicted labels (N, C)

        Returns:
            float: The computed metric value.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def name(self) -> str:
        """
        Get the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return f"{self.METRIC_NAME}@{self.k}"

class Recall(BaseMetric):
    METRIC_NAME = "Recall"

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute the recall at k.

        Args:
            y_true (torch.Tensor): The ground truth labels (N, )
            y_pred (torch.Tensor): The predicted labels (N, C)

        Returns:
            float: The recall at k.
        """
        recalls = []

        y_pred_slice = y_pred[:, :self.k].tolist()

        for pred, true in zip(y_pred_slice, y_true.tolist()):
            if true in pred:
                recalls.append(1.0)
            else:
                recalls.append(0.0)

        return sum(recalls) / len(recalls) if recalls else 0.0


class NDCG(BaseMetric):
    METRIC_NAME = "NDCG"

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute the NDCG at k.
        Assume only one positive label in y_true.

        Args:
            y_true (torch.Tensor): The ground truth labels (N, )
            y_pred (torch.Tensor): The predicted labels (N, C)

        Returns:
            float: The NDCG at k.
        """
        ndcgs = []

        y_pred_slice = y_pred[:, :self.k].tolist()

        for pred, true in zip(y_pred_slice, y_true.tolist()):
            if true in pred:
                idx = pred.index(true)
                ndcgs.append(1 / (torch.log2(torch.tensor(idx + 2)).item()))
            else:
                ndcgs.append(0.0)

        return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
