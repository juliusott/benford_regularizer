import argparse

import numpy as np

from train_cifar import main

models = ['PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', 'densenet201', 'renext', 'vit', 'swin']

seeds = np.random.randint(0, int(1e6), size=(25,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', nargs='*', type=str, help='model to train', choices=models)
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--seed', nargs='*', type=int, choices=np.arange(0, int(1e6)).tolist())
    parser.add_argument('--benford', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--scale', type=float, default=0.1, help='scaling factor for quantile loss')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument('--data_size', type=float, default=1.0, help='dataset size in percentage')

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
