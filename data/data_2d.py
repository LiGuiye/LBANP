import torch
import numpy as np

from data.modules import CustomDataset
from utils.utils import normalize, scale, load_config


class Ogallala(CustomDataset):
    def __init__(self, n_samples=25000, is_train=True, normalize=True, replace=True):
        super().__init__(n_samples, is_train, replace, normalize)

        self.data_folder = f"data/datafile/Ogallala/npy"

        self.n_samples = n_samples if n_samples else 25000  # sample times
        self.get_attributes()
        self.load_data()
        # Put everything in Cuda to speed up the loading process
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)
        self.data_all = self.data_all.to(device)
        self.targets_all = self.targets_all.to(device)

    def get_attributes(self):
        config = load_config(f'{self.data_folder}/meta.yaml')
        self.train_lon_min = config["lon_min"]
        self.train_lon_max = config["lon_max"]
        self.train_lat_min = config["lat_min"]
        self.train_lat_max = config["lat_max"]
        self.train_z_min = config["WTE2013_min"]
        self.train_z_max = config["WTE2013_max"]
        self.train_z_mean = config["WTE2013_mean"]
        self.train_z_std = config["WTE2013_std"]

    def load_data(self):
        label = "train" if self.is_train else "test"
        dataset_path = f"{self.data_folder}/Ogallala_WTE2013_{label}.npy"

        dataset = np.load(dataset_path, allow_pickle=True).item()

        x = torch.from_numpy(dataset["x"]).contiguous()
        x = scale(x, self.train_lon_min, self.train_lon_max, -1, 1)

        y = torch.from_numpy(dataset["y"]).contiguous()
        y = scale(y, self.train_lat_min, self.train_lat_max, -1, 1)

        self.data_all = torch.vstack((x, y)).T.float()

        z = torch.from_numpy(dataset["z"]).float().contiguous()
        if self.z_normalize:
            z = normalize(z, self.train_z_mean, self.train_z_std)

        if self.is_train:
            self.targets_all = z[:, None]
            self.n_cntxt = len(z)  # total number of points in training set
            self.precompute_chunk_(replace=self.replace)
        else:
            self.data_all = self.data_all[None]
            self.targets_all = z[None, :, None]
            self.set_samples_(self.data_all, self.targets_all)
