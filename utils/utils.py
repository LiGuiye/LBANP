import os
import yaml
import torch
import numpy as np

import logging
from importlib.machinery import SourceFileLoader


def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger


def ckpt_path(args, epoch=None, suffix=None, filetype="tar"):
    if epoch is None:
        return f"{args.root}/weights"
    elif suffix is None:
        return f"{args.root}/weights/Ogallala_{args.model_name}_epoch{epoch}.{filetype}"
    else:
        return f"{args.root}/weights/Ogallala_{args.model_name}_epoch{epoch}_{suffix}.{filetype}"


def load_func(module, path):
    return getattr(SourceFileLoader(module, path).load_module(), module)


def mkdir(dirs):
    if type(dirs) == list:
        for dirs in dirs:
            os.makedirs(dirs, exist_ok=True)
    else:
        os.makedirs(dirs, exist_ok=True)


def load_config(config_path, parser_args=None):
    """Load config from yaml. If parser_args, override and return parser_args according to saved config"""

    if not os.path.exists(config_path):
        print(f"No pre-saved configuration at given path: {config_path}.")
        return parser_args
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Override the default arguments
    if parser_args:
        for key, val in vars(parser_args).items():
            if key in config:
                if val is None:
                    # Override None in args using saved config
                    vars(parser_args)[key] = config[key]
                elif val != config[key]:
                    print(f"Overriding argument {key}: {val}")
        return parser_args
    else:
        return config


def backup_args(parser_args, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(parser_args.__dict__, f)


def print_args(args):
    print('-' * 80)
    for k, v in args.__dict__.items():
        print('%-32s %s' % (k, v))
    print('-' * 80)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_random_seed(seed=111, verbose=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        print("Random Seed: ", seed)
    return seed


def normalize(data, mean=None, std=None):
    mean = data.mean() if mean is None else mean
    std = data.std() if std is None else std
    return (data-mean)/(max(std, 1e-5))


def inverse_normalze(data, original_mean=None, original_std=None):
    return (data*max(original_std, 1e-5))+original_mean


def scale(data, original_min, original_max, target_min, target_max):
    """
    Min-max stretch
    """
    pixel_min = torch.min(data) if original_min is None else original_min
    pixel_max = torch.max(data) if original_max is None else original_max

    return torch.add(
        torch.multiply(
            torch.divide(
                torch.add(data, -pixel_min),
                torch.maximum(
                    torch.add(
                        pixel_max, -pixel_min), torch.tensor(1e-5, device=data.device)
                ),
            ),
            torch.add(target_max, -target_min),
        ),
        target_min,
    )
