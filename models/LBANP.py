import copy
import time
from tqdm import tqdm
from attrdict import AttrDict
from einops import rearrange, repeat

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils.utils import *
from utils.metrics import calc_metrics
from data.modules import context_target_split


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(
            context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs) + x


class PostNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = x + self.fn(x, **kwargs)

        return self.norm(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_feedforward=128, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward * 2),
            GEGLU(),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LBANPEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(LBANPEncoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)

        if norm_first:
            self.latent_self_attn = PreNorm(self.latent_dim, Attention(
                self.latent_dim, heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout))
            self.latent_ff = PreNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model,
                                      heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=self.d_model)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            self.latent_self_attn = PostNorm(self.latent_dim, Attention(
                self.latent_dim, heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout))
            self.latent_ff = PostNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model,
                                       heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=self.d_model)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))

    def forward(self, context_encodings, latents):
        x = latents
        x = self.cross_attn(x, context=context_encodings)
        x = self.cross_ff(x)

        x = self.latent_self_attn(x)
        x = self.latent_ff(x)

        return x


class LBANPEncoder(nn.Module):
    """
        Iterative Attention-based model that encodes context datapoints into a list of embeddings
    """

    def __init__(self, encoder_layer, num_layers, return_only_last=False):
        super(LBANPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_only_last = return_only_last

    def forward(self, context_encodings, latents):
        b, *axis = context_encodings.shape
        latents = repeat(latents, 'n d -> b n d', b=b)

        layer_outputs = []
        last_layer_output = None
        for layer in self.layers:
            latents = layer(context_encodings, latents)
            layer_outputs.append(latents)
            last_layer_output = latents
        if self.return_only_last:
            return [last_layer_output]
        else:
            return layer_outputs


class NPDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(NPDecoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)
        # Self Attention performs  the linear operations
        if norm_first:
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads=nhead,
                                      dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=self.latent_dim)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            self.cross_attn = PostNorm(self.latent_dim, Attention(
                self.latent_dim, self.d_model, heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=self.latent_dim)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(
                self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))

    def forward(self, query_encodings, context):

        x = query_encodings
        x = self.cross_attn(x, context=context)
        x = self.cross_ff(x)

        return x


class NPDecoder(nn.Module):
    """
        Attention-based model that retrieves information via the context encodings to make predictions for the query/target datapoints
    """

    def __init__(self, decoder_layer, num_layers):
        super(NPDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query_encodings, context_encodings):
        assert len(context_encodings) == self.num_layers

        x = query_encodings
        for layer, context_enc in zip(self.layers, context_encodings):
            x = layer(x, context=context_enc)

        out = x
        return out


class LBANP(nn.Module):
    """
        Latent Bottlenecked Attentive Neural Process (LBANPs)
    """

    def __init__(
        self,
        num_latents,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        norm_first=True,
        bound_std=False
    ):
        super(LBANP, self).__init__()

        self.latent_dim = d_model
        self.latents = nn.Parameter(torch.randn(
            num_latents, self.latent_dim), requires_grad=True)  # Learnable latents!

        # Context Related
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = LBANPEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.encoder = LBANPEncoder(encoder_layer, num_layers)

        # Query Related
        self.query_embedder = build_mlp(dim_x, d_model, d_model, emb_depth)

        decoder_layer = NPDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.decoder = NPDecoder(decoder_layer, num_layers)

        # Predictor Related
        self.bound_std = bound_std

        self.norm_first = norm_first
        if self.norm_first:
            self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def get_context_encoding(self, batch):
        # Perform Encoding
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        context_embeddings = self.embedder(x_y_ctx)
        context_encodings = self.encoder(context_embeddings, self.latents)
        return context_encodings

    def get_predict_encoding(self, batch, context_encodings=None):

        if context_encodings is None:
            context_encodings = self.get_context_encoding(batch)

        # Perform Decoding
        query_embeddings = self.query_embedder(batch.xt)
        encoding = self.decoder(query_embeddings, context_encodings)
        # Make predictions
        if self.norm_first:
            encoding = self.norm(encoding)
        return encoding

    def predict(self, xc, yc, xt, context_encodings=None, num_samples=None):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt

        encoding = self.get_predict_encoding(
            batch, context_encodings=context_encodings)

        out = self.predictor(encoding)

        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)

    def forward(self, batch, num_samples=None, reduce_ll=True):

        pred_tar = self.predict(batch.xc, batch.yc, batch.xt)

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = - (outs.tar_ll)

        return outs


def build_model(args):
    model = LBANP(num_latents=256, dim_x=2, dim_y=1,
                  d_model=64, emb_depth=4, dim_feedforward=128, nhead=4, dropout=0.0,
                  num_layers=6).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def load_and_pred(args, model, xc, yc, xt, epoch):
    if epoch:
        ckpt = torch.load(ckpt_path(args, epoch))
        model.load_state_dict(ckpt.model)

    training_status = model.training

    model.eval()
    with torch.no_grad():
        outs = model.predict(xc, yc, xt, num_samples=1)

    if training_status:
        model.train()

    return [outs.mean, outs.stddev]


def load_and_eval(args, model, data_train, data_test, epoch=None, verbose=True):
    xc = data_train.data_all[None].cuda()
    yc = data_train.targets_all[None].cuda()
    xt = data_test.data_all.cuda()
    yt = data_test.targets_all

    mean, std = load_and_pred(args, model, xc, yc, xt, epoch)

    if data_test.z_normalize:
        yt = inverse_normalze(yt, data_test.train_z_mean,
                              data_test.train_z_std)
        mean = inverse_normalze(
            mean, data_test.train_z_mean, data_test.train_z_std)

    if verbose:
        print(f"Epoch {epoch} {calc_metrics(yt, mean)}")
    else:
        return calc_metrics(yt, mean)


def train_model(model, optimizer, args, data_train, data_test, verbose=False):
    backup_args(args, os.path.join(args.root, 'args.yaml'))

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*args.n_epochs)

    logfilename = os.path.join(
        args.root, 'train_{}.log'.format(time.strftime('%Y%m%d-%H%M')))

    logger = get_logger(logfilename)

    for epoch in range(1, args.n_epochs+1):
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):

            batch = context_target_split(x.cuda(), y.cuda())

            optimizer.zero_grad()
            if args.model_name in ["LNP", "ANP"]:
                train_num_samples = 4
                outs = model(batch, num_samples=train_num_samples)
            else:
                outs = model(batch)

            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and i % args.log_freq_iters == 0:
                line = f'{args.model_name} {args.expid} Epoch: {epoch}/{args.n_epochs} '
                line += f"Iter: {i+1}/{len(train_loader)} "
                line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
                line += f'Loss: {outs.loss.item()}'
                logger.info(line)

        if verbose and i % args.log_freq_epoch == 0:
            line = f'{args.model_name} {args.expid} Epoch: {epoch}/{args.n_epochs} '
            line += f'Loss: {outs.loss.item()} '
            if data_test:
                line += load_and_eval(args, model, data_train,
                                      data_test, None, False)
            logger.info(line)

        if epoch % args.save_freq == 0 or epoch == args.n_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1

            torch.save(ckpt, ckpt_path(args, epoch))
            del ckpt
