from transfer_learning import main
import argparse

# all seeds = [96010, 20004, 69469, 92983, 96872, 94213, 96723, 42,
#            3638, 76325, 14009,  6885, 3407, 84738, 58775, 82009, 72163,
#        18833, 18632,  5817, 64279, 42826, 61553, 75118]
seeds = [96010, 20004,69469, 92983, 96872, 94213, 96723, 42,
            3638, 76325, 14009,  6885, 3407, 84738, 58775, 82009, 72163,
        18833, 18632,  5817, 64279, 42826, 61553, 75118]
models = [f'resnet{n}' for n in [18, 34, 50]]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Transfer Learning')
    parser.add_argument('--model', nargs='*', type = str, help='model to train', choices=models)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--seed', nargs='*', type=int)
    parser.add_argument('--early_stop_patience', default=5, type=int, help='early stopping patience')
    parser.add_argument('--benford', action='store_true')
    parser.add_argument('--scale', default=1, type=float, help='scaling factor for the benford optimization')

    args = parser.parse_args()

    if args.seed is not None:
        seeds = args.seed

    if args.model is not None:
        models = args.model

    for seed in seeds:
        for model in models:
            args.seed = seed
            args.model = model
            print(args)
            main(args)