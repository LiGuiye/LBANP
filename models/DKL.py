"""
https://docs.gpytorch.ai/en/v1.10/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
"""
import os
import time
import torch
import gpytorch
from attrdict import AttrDict
from torch.utils.data import TensorDataset, DataLoader

from utils.utils import *
from utils.metrics import calc_metrics


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, input_dim, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )

        self.feature_extractor = LargeFeatureExtractor(input_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_model(args, data_train):
    train_x = data_train.data_all
    train_y = data_train.targets_all.squeeze()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(2, train_x, train_y, likelihood).cuda()
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return model, optimizer


def train_model(model, optimizer, args, data_train, data_test, verbose):
    backup_args(args, os.path.join(args.root, 'args.yaml'))

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    logfilename = os.path.join(
        args.root, 'train_{}.log'.format(time.strftime('%Y%m%d-%H%M')))

    logger = get_logger(logfilename)

    xc = data_train.data_all.cuda()
    yc = data_train.targets_all.squeeze().cuda()

    for epoch in range(1, args.n_epochs+1):
        model.train()

        optimizer.zero_grad()

        output = model(xc)
        loss = -mll(output, yc)

        loss.backward()
        optimizer.step()

        if verbose and epoch % args.log_freq_epoch == 0:
            line = f'{args.model_name} {args.expid} Epoch: {epoch}/{args.n_epochs} '
            line += f'Loss: {round(loss.item(),2)} '
            if data_test:
                line += load_and_eval(args, model, None, data_test, None, False)
            logger.info(line)

        if epoch % args.save_freq == 0 or epoch == args.n_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1

            torch.save(ckpt, ckpt_path(args, epoch))
            del ckpt


def load_and_pred(args, model, xc, yc, xt, epoch):
    if epoch:
        ckpt = torch.load(ckpt_path(args, epoch))
        model.load_state_dict(ckpt.model)

    training_status = model.training

    test_dataset = TensorDataset(xt.squeeze())
    test_loader = DataLoader(test_dataset, batch_size=1024)

    mean = []
    std = []

    model.eval()
    with torch.no_grad():
        for x_batch in test_loader:
            preds = model(x_batch[0])
            mean.append(preds.mean)
            std.append(preds.stddev)

    if training_status:
        model.train()

    return [torch.cat(mean, dim=-1), torch.cat(std, dim=-1)]


def load_and_eval(args, model, data_train, data_test, epoch=None, verbose=True):
    mean, std = load_and_pred(args, model, None, None,
                              data_test.data_all.cuda(), epoch)

    yt = data_test.targets_all.squeeze()
    if data_test.z_normalize:
        yt = inverse_normalze(yt, data_test.train_z_mean,
                              data_test.train_z_std)
        mean = inverse_normalze(
            mean, data_test.train_z_mean, data_test.train_z_std)

    if verbose:
        print(f"Epoch {epoch} {calc_metrics(yt, mean)}")
    else:
        return calc_metrics(yt, mean)
