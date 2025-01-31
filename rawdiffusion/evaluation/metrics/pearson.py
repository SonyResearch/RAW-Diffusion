import torch
from .base_metric import BaseMetric
import numpy as np
from scipy.stats import pearsonr


def pearson(pred, gt):
    bs = pred.size(0)
    n_chans = pred.size(1)

    pearsons = []
    for i in range(bs):
        image_pearsons = []
        for c in range(n_chans):
            pred_np = pred[i, c, :, :].detach().cpu().numpy()
            gt_np = gt[i, c, :, :].detach().cpu().numpy()
            pearson = pearsonr(pred_np.flatten(), gt_np.flatten())[0]
            image_pearsons.append(pearson)

        image_pearson_mean = np.mean(image_pearsons)
        pearsons.append(image_pearson_mean)
    pearsons = np.array(pearsons)
    return pearsons


class PearsonMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)

        pearson_values = pearson(pred, target)
        pearson_values = pearson_values[~np.isnan(pearson_values)]

        if len(pearson_values) != pred.size(0):
            print(
                f"pearson for current batch is nan, {len(pearson_values)} / {pred.size(0)} valid"
            )

        if len(pearson_values) > 0:
            self.value += np.sum(pearson_values)
            self.count += len(pearson_values)

    def compute(self) -> float:
        if self.count > 0:
            return self.value / self.count
        else:
            return float("nan")
