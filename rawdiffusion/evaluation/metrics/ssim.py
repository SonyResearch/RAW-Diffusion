import torch
from .base_metric import BaseMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)

        self.ssim = self.ssim.to(pred.device)
        ssim_value = self.ssim(
            pred,
            target,
        )

        self.value += ssim_value.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count
