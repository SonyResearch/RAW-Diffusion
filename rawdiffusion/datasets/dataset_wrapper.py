from torch.utils.data import Dataset
import numpy as np


class PermutedDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset

        rnd = np.random.RandomState(seed=seed)
        n = len(dataset)
        self.permutation = rnd.permutation(n)

    def __getitem__(self, index):
        item = self.dataset[self.permutation[index]]
        return item

    def __len__(self):
        return len(self.dataset)


class RandomSampleDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset

        self.n = n
        self.dataset_size = len(dataset)
        print(f"RandomSampleDataset: dataset_size={self.dataset_size}, n={self.n}")

    def __getitem__(self, index):
        item = self.dataset[np.random.randint(self.dataset_size)]
        return item

    def __len__(self):
        return self.n
