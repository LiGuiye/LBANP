import torch
import numpy as np
from attrdict import AttrDict
from torch.utils.data import Dataset


def context_target_split(x, y, min_ctx_points=1, max_ctx_ratio=0.1):
    """
    Randomly select some context points and corresponding target points.
    """
    n_ctx_points = y.shape[1]

    max_ctx_points = round(max_ctx_ratio*n_ctx_points)
    max_ctx_points = max(min_ctx_points, max_ctx_points)
    num_ctx = torch.randint(
        low=min_ctx_points, high=max_ctx_points, size=[1]).item()

    batch = AttrDict()
    batch.x = x
    batch.y = y
    batch.xc = x[:, :num_ctx, :]
    batch.yc = y[:, :num_ctx, :]
    batch.xt = x[:, num_ctx:, :]
    batch.yt = y[:, num_ctx:, :]
    return batch


class CustomDataset(Dataset):
    def __init__(self, n_samples: int, is_train: bool, replace: bool, normalize: bool):
        super().__init__()
        self.is_train = is_train
        self.n_samples = n_samples
        self.replace = replace  # with or without replacement during sampling
        self.z_normalize = normalize

    def load_data(self):
        """Load and preprocess training and testing datasets"""
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.is_train:
            self._idx_precompute += 1
            if self._idx_precompute == self.n_samples:
                self.precompute_chunk_()
            return self.data[self._idx_precompute], self.targets[self._idx_precompute]
        else:
            return self.data[index], self.targets[index]

    def _sample_features_targets(self, n_points, n_samples, replace=True):
        """Generate random indices to sample the training set"""
        if replace:
            # with replacement (a point can be selected multiple times in one sample)
            idx = np.random.choice(
                range(len(self.data_all)), (n_samples, n_points))
        else:
            # without replacement (never see same points twice within one sample)
            idx = np.array([np.random.choice(
                range(len(self.data_all)), (n_points), replace=False) for _ in range(n_samples)])
        return self.data_all[idx], self.targets_all[idx]

    def get_samples(self, n_samples: int = None, n_points: int = None, replace=True):
        """Sample points from the context set. Default sampling n_cntxt points."""
        n_points = n_points if n_points is not None else self.n_cntxt
        n_samples = n_samples if n_samples is not None else self.n_samples

        data, targets = self._sample_features_targets(
            n_points, n_samples, replace=replace)
        return data, targets

    def precompute_chunk_(self, replace=True):
        """Sampling the context set to generate a new sub training dataset and start a new epoch"""
        self._idx_precompute = 0
        self.data, self.targets = self.get_samples(replace=replace)

    def set_samples_(self, data, targets):
        """Use the given features and target values as the data for testing."""
        self.data = data
        self.targets = targets
        self.n_samples = self.data.size(0)
