import torch
import numpy as np
from utils.utils import normalize


def calc_metrics(data1, data2):
    mae = calc_mae(data1, data2)
    rmse = calc_rmse(data1, data2)
    swd = calc_swd(data1, data2)
    return f"MAE: {mae} RMSE: {rmse} SWD: {swd}"


def calc_rmse(data1, data2):
    data1, data2 = data1.squeeze(), data2.squeeze()
    if isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
        data1 = data1 if isinstance(data1, np.ndarray) else data1.numpy()
        data2 = data2 if isinstance(data2, np.ndarray) else data2.numpy()

        rmse = np.sqrt(np.mean(np.square(data1 - data2)))
    else:
        rmse = torch.sqrt(torch.mean(torch.square(data1.cpu() - data2.cpu())))
    return rmse.item()


def calc_mae(data1, data2):
    data1, data2 = data1.squeeze(), data2.squeeze()
    if isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
        data1 = data1 if isinstance(data1, np.ndarray) else data1.numpy()
        data2 = data2 if isinstance(data2, np.ndarray) else data2.numpy()

        mae = np.mean(np.abs(data1 - data2))
    else:
        mae = torch.mean(torch.abs(data1.cpu() - data2.cpu()))
    return mae.item()


def calc_swd(data1, data2):
    data1, data2 = data1.squeeze()[:, None], data2.squeeze()[:, None]
    data1 = normalize(data1)
    data2 = normalize(data2)

    data1 = data1.cpu().float() if not isinstance(
        data1, np.ndarray) else data1.astype(float)
    data2 = data2.cpu().float() if not isinstance(
        data2, np.ndarray) else data2.astype(float)

    swd = sliced_wasserstein_np(data1, data2)
    return swd.item()


def sliced_wasserstein_np(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape  # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(
            A.shape[1], dirs_per_repeat
        )  # (descriptor_component, direction)
        dirs /= np.sqrt(
            np.sum(np.square(dirs), axis=0, keepdims=True)
        )  # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)  # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(
            projA, axis=0
        )  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)  # pointwise wasserstein distances
        # average over neighborhoods and directions
        results.append(np.mean(dists))
    return np.mean(results)  # average over repeats
