"""
https://docs.gpytorch.ai/en/v1.10/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
"""
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

import os
import time
from attrdict import AttrDict
from utils.utils import *
from utils.metrics import calc_metrics
from torch.utils.data import TensorDataset, DataLoader


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(
                output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        ) + ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(
                    gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGP(DeepGP):
    def __init__(self, input_dim, n_hidden_dims: int = 10):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=n_hidden_dims,
            mean_type='constant'
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output


def build_model(args):
    model = DeepGP(2).double().cuda()
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def train_model(model, optimizer, args, data_train, data_test, verbose=False):
    backup_args(args, os.path.join(args.root, 'args.yaml'))

    data_train = TensorDataset(data_train.data_all, data_train.targets_all)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    mll = DeepApproximateMLL(VariationalELBO(
        model.likelihood, model, len(data_train)))

    logfilename = os.path.join(
        args.root, 'train_{}.log'.format(time.strftime('%Y%m%d-%H%M')))

    logger = get_logger(logfilename)

    for epoch in range(1, args.n_epochs+1):
        model.train()
        for i, (xc, yc) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(xc.cuda())  # output 10 samples
            loss = -mll(output, yc.squeeze().cuda())

            loss.backward()
            optimizer.step()

        if verbose and epoch % args.log_freq_epoch == 0:
            line = f'{args.model_name} {args.expid} Epoch: {epoch}/{args.n_epochs} '
            line += f'Loss: {round(loss.item(),2)} '
            if data_test:
                line += load_and_eval(args, model, None,
                                      data_test, None, False)
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

    model.eval()

    test_loader = DataLoader(TensorDataset(xt), batch_size=args.batch_size)
    with torch.no_grad():
        mus = []
        stds = []
        for [x_batch] in test_loader:
            preds = model.likelihood(model(x_batch))
            mus.append(preds.mean.mean(0))
            stds.append(preds.stddev.mean(0))

    if training_status:
        model.train()

    return [torch.concat(mus), torch.concat(stds)]


def load_and_eval(args, model, data_train, data_test, epoch=None, verbose=True):
    mean, std = load_and_pred(args, model, None, None,
                              data_test.data_all.squeeze().cuda(), epoch)
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
