import numpy as np
import matplotlib.pyplot as plt

import geopandas
from matplotlib import cm
from shapely.geometry import Point
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from utils.utils import *


def prep(x): return x.squeeze() if isinstance(
    x, np.ndarray) else x.cpu().squeeze()


def visualize(args, model, data_train, data_test, epoch=None):
    """
    Generate prediction on the test set.
    """
    load_and_pred = load_func("load_and_pred", f"models/{args.model_name}.py")

    save_path = f"{args.root}/predicts"
    mkdir(save_path)

    title = f"{args.model_name} | Ogallala | Epoch {epoch}"
    fig_name = f"Ogallala_{args.model_name}_epoch{epoch}"
    fig_path = f"{save_path}/{fig_name}.png"

    xc = data_train.data_all[None]
    yc = data_train.targets_all[None]
    xt = prep(data_test.data_all)
    raw_xt_lon = scale(
        xt[:, 0], -1, 1, data_test.train_lon_min, data_test.train_lon_max)
    raw_xt_lat = scale(
        xt[:, 1], -1, 1, data_test.train_lat_min, data_test.train_lat_max)
    predict_coor = np.stack((raw_xt_lon, raw_xt_lat), 1)

    xt = xt[None].to(xc.device)
    pred_mean, pred_std = load_and_pred(
        args, model, xc, yc, xt, epoch)
    pred_mean, pred_std = prep(pred_mean), prep(pred_std)

    if data_train.z_normalize:
        pred_mean = inverse_normalze(
            pred_mean, data_train.train_z_mean, data_train.train_z_std)

    pred_mean = np.concatenate(
        (predict_coor, pred_mean.squeeze()[:, None]), 1)
    pred_std = np.concatenate(
        (predict_coor, pred_std.squeeze()[:, None]), 1)

    plot_2d_Ogallala(data_train, pred_mean, pred_std, title, fig_path)

    np.savetxt(f"{save_path}/{fig_name}_mean.csv", pred_mean,
               delimiter=",", header="x,y,mean", comments='')
    np.savetxt(f"{save_path}/{fig_name}_std.csv", pred_std,
               delimiter=",", header="x,y,std", comments='')


def plot_2d_Ogallala(data_train, predict_mean, predict_std, title, fig_path):
    boundary = geopandas.read_file(
        "data/datafile/Ogallala/boundary/boundary.shp")

    # for training
    train_x = data_train.data_all.cpu()
    train_y = data_train.targets_all.cpu()

    # for visualization
    xyz_train = np.concatenate((
        np.stack((
            scale(train_x[:, 0], -1, 1, data_train.train_lon_min,
                  data_train.train_lon_max),
            scale(train_x[:, 1], -1, 1, data_train.train_lat_min, data_train.train_lat_max)), 1),
        inverse_normalze(train_y, data_train.train_z_mean, data_train.train_z_std)), 1)

    gdf_gt = geopandas.GeoDataFrame({'gt': xyz_train[:, 2]}, geometry=[
                                    Point(i) for i in xyz_train[:, :2]], crs="EPSG:4269")

    gdf_test = geopandas.GeoDataFrame({'mean': predict_mean[:, 2], 'std': predict_std[:, 2]}, geometry=[
                                      Point(i) for i in predict_mean[:, :2]], crs="EPSG:4269")

    rows, cols = 1, 3
    fig = plt.figure(figsize=(cols * 5, rows * 9))
    fig.suptitle(title, y=1.15)
    gs = gridspec.GridSpec(
        rows,
        cols + 2,
        top=1.08,
        bottom=0.6,
        right=0.7,
        width_ratios=[0.04] + [1 for _ in range(cols)] + [0.04],
        height_ratios=[1 for _ in range(rows)],
    )

    ax = fig.add_subplot(gs[0])
    ax.set(xticks=[], yticks=[])
    vmin, vmax = gdf_gt['gt'].min(), gdf_gt['gt'].max()
    fig.colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax)), cax=ax)
    ax.yaxis.set_ticks_position('left')

    ax = fig.add_subplot(gs[1])
    ax.set(xticks=[], yticks=[])
    ax.set_title("Ground Truth (training)")
    base = boundary.plot(ax=ax, color='white', edgecolor='black')
    gdf_gt.plot(ax=base, column='gt', marker='o',
                markersize=4, aspect=1, vmin=vmin, vmax=vmax)

    ax = fig.add_subplot(gs[2])
    ax.set(xticks=[], yticks=[])
    ax.set_title("Predict.Mean")
    base = boundary.plot(ax=ax, color='white', edgecolor='black')
    gdf_test.plot(ax=base, column='mean', marker='o',
                  markersize=4, aspect=1, vmin=vmin, vmax=vmax)

    vmin, vmax = gdf_test['std'].min(), gdf_test['std'].max()
    ax = fig.add_subplot(gs[3])
    ax.set(xticks=[], yticks=[])
    ax.set_title("Predict.Std")
    base = boundary.plot(ax=ax, color='white', edgecolor='black')
    gdf_test.plot(ax=base, column='std', marker='o', markersize=4,
                  cmap="gray", aspect=1, vmin=vmin, vmax=vmax)

    fig.colorbar(cm.ScalarMappable(norm=Normalize(
        vmin=vmin, vmax=vmax), cmap="gray"), cax=fig.add_subplot(gs[4]))

    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    return


def plot_2d_Ogallala_split(fig_path=None):
    root = "data/datafile/Ogallala"
    fig_path = f"{root}/npy/split.png" if fig_path is None else fig_path
    boundary = geopandas.read_file(f"{root}/boundary/boundary.shp")

    dataset = [np.load(f"{root}/npy/Ogallala_WTE2013_all.npy", allow_pickle=True).item(),
               np.load(f"{root}/npy/Ogallala_WTE2013_train.npy",
                       allow_pickle=True).item(),
               np.load(f"{root}/npy/Ogallala_WTE2013_test.npy",
                       allow_pickle=True).item(),
               np.loadtxt(f"{root}/npy/Ogallala_TestGrid.csv", delimiter=",", dtype=float, skiprows=1)]
    gdf_data = []
    for i, data in enumerate(dataset):
        if i == 3:
            gdf_data.append(geopandas.GeoDataFrame(
                {}, geometry=[Point(i) for i in data[:, :2]], crs="EPSG:4269"))
        else:
            gdf_data.append(geopandas.GeoDataFrame({'data': data['z']}, geometry=[
                            Point(i) for i in np.stack((data['x'], data['y']), 1)], crs="EPSG:4269"))

    labels = ["Ground Truth", "Train", "Evaluate", "Target"]

    rows, cols = 1, 4
    fig = plt.figure(figsize=(cols * 5, rows * 9))
    gs = gridspec.GridSpec(
        rows,
        cols + 2,
        top=1.08,
        bottom=0.6,
        right=0.7,
        width_ratios=[0.04] + [1 for _ in range(cols)] + [0.04],
        height_ratios=[1 for _ in range(rows)],
    )

    ax = fig.add_subplot(gs[0])
    ax.set(xticks=[], yticks=[])

    vmin, vmax = gdf_data[0]['data'].min(), gdf_data[0]['data'].max()
    fig.colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax)), cax=ax)
    ax.yaxis.set_ticks_position('left')

    for i in range(4):
        ax = fig.add_subplot(gs[i+1])
        ax.set(xticks=[], yticks=[])
        ax.set_title(labels[i])
        base = boundary.plot(ax=ax, color='white', edgecolor='black')
        if i == 3:
            gdf_data[i].plot(ax=base, marker='o',
                             markersize=0.001, aspect=1, color='black')
        else:
            gdf_data[i].plot(ax=base, column='data', marker='o',
                             markersize=2, aspect=1, vmin=vmin, vmax=vmax)

    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    return
