import argparse

import numpy as np

from transfer_learning import main
from utils.utils import check_positive

seeds = np.random.randint(0, int(1e6), size=(25,))
models = [f'resnet{n}' for n in [18, 34, 50]]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning')
    parser.add_argument('--model', nargs='*', type=str, help='model to train', choices=models)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--seed', nargs='*', type=int, choices=np.arange(0, int(1e6)).tolist())
    parser.add_argument('--early_stop_patience', default=5, type=int, help='early stopping patience')
    parser.add_argument('--benford', action='store_true')
    parser.add_argument('--scale', default=1, type=float, help='scaling factor for the benford optimization')
    parser.add_argument('--benford_iter', default=10, type=check_positive, help='number of benford iterations')

    args = parser.parse_args()

    if args.seed is not None:
        seeds = args.seed

    if args.model is not None:
        models = args.model

    for seed in seeds:
        for model in models:
            args.seed = int(seed)
            args.model = model
            print(args)
            main(args)
