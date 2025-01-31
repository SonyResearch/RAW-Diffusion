import torch
from .base_metric import BaseMetric


def mse(pred, gt):
    bs = pred.size(0)
    pred = pred.reshape(bs, -1)
    gt = gt.reshape(bs, -1)
    return torch.mean((pred - gt) ** 2, dim=1)


class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)

        mse_value = mse(pred, target)

        self.value += mse_value.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count
