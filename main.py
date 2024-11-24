import os
import argparse
from glob import glob
from tqdm import tqdm

from utils.utils import *
from utils.plot import visualize
from data.data_2d import Ogallala


def load_module(model_name):
    path = f"models/{model_name}.py"
    build_model = load_func("build_model", path)
    train_model = load_func("train_model", path)
    load_and_eval = load_func("load_and_eval", path)

    return build_model, train_model, load_and_eval


def eval(args, func, model, data_train, data_test):
    epochs = sum([[int(i.split('epoch')[1].split('.')[0].split('_')[0])
                   for i in glob(ckpt_path(args) + f'/*.{t}')]
                  for t in ["tar", "npy"]], [])
    epochs = list(set(epochs))  # unique

    if args.eval_epoch == -1:
        # for the latest epoch
        epoch = max(epochs) if epochs else 1
        func(args, model, data_train, data_test, epoch)
    elif args.eval_epoch == None:
        # for all epochs in the folder
        epochs.sort()
        for epoch in tqdm(epochs):
            func(args, model, data_train, data_test, epoch)
    else:
        # for chosen epoch
        func(args, model, data_train, data_test, args.eval_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='LBANP', type=str)

    # Experiment
    parser.add_argument('--mode', choices=['train', 'eval', 'visualize'], default='visualize')
    parser.add_argument('--expid', type=str, help="folder name", default='Toy')

    # Train
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_samples', type=int, help="Meta-samples", default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)

    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--save_freq', type=int, help="# epoch for 1 save", default=None)
    parser.add_argument('--verbose', type=boolean_string, help="log in training", default=None)
    parser.add_argument('--log_freq_iters', type=int, help="# iters for 1 log", default=None)
    parser.add_argument('--log_freq_epoch', type=int, help="# epoch for 1 log", default=None)

    # Evaluation
    parser.add_argument('--eval_epoch', type=int, help="leave None to eval or visualize all", default=None)

    args = parser.parse_args()
    args = load_config(f'configs/{args.model_name}/Ogallala.yaml', args)

    args.root = os.path.join(f"results/{args.model_name}/{args.expid}")
    mkdir([f"{args.root}/weights", f"{args.root}/predicts"])
    print_args(args)

    # load dataset
    data_train = Ogallala(n_samples=args.n_samples, is_train=True)
    data_test = Ogallala(is_train=False)

    # load model
    build_model, train_model, load_and_eval = load_module(args.model_name)
    if args.model_name == "DKL":
        model, optimizer = build_model(args, data_train)
    else:
        model, optimizer = build_model(args)

    set_random_seed(666)
    if args.mode == "train":
        train_model(model, optimizer, args, data_train,
                    data_test, args.verbose)
    else:
        func = visualize if args.mode == "visualize" else load_and_eval
        eval(args, func, model, data_train, data_test)


if __name__ == '__main__':
    main()
