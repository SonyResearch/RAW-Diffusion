class CollectionMetric:
    def __init__(self, metrics={}):
        self.metrics = metrics

    def add(self, metric, key):
        self.metrics[key] = metric

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target)

    def compute(self):
        return {key: metric.compute() for key, metric in self.metrics.items()}
