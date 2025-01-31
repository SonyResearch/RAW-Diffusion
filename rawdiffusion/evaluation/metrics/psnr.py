import torch
from .base_metric import BaseMetric


def psnr(pred, gt, max_val=1.0):
    bs = pred.size(0)
    pred = pred.reshape(bs, -1)
    gt = gt.reshape(bs, -1)

    mse = torch.mean((pred - gt) ** 2, dim=1)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class PSNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)

        psnr_value = psnr(pred, target)

        self.value += psnr_value.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count
