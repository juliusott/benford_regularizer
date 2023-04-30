'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from models.preact_resnet import PreActResNet101, PreActResNet50
from models.densenet import DenseNet121, DenseNet169
from models.resnext import ResNeXt29_2x64d
from utils import EarlyStopper
import os
import argparse
from benford_regularizer import quantile_loss, compute_kl
from models.densenet import *
from models.preact_resnet import *
from models.resnext import *
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt
import random



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0  # best test accuracy
    # Data
    print('==> Preparing data..')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    set_seed(args.seed)


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    if args.resume:
        save_dir = f'./experiments/{args.model}_from_checkpoint/'
    else:
        save_dir = f'./experiments/{args.model}/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Model
    print(f'==> Building model: {args.model}')

    if args.model == 'PreActresnet101':
        net = PreActResNet101()
    elif args.model == 'PreActresnet50':
        net = PreActResNet50()
    elif args.model == 'densenet121':
        net = DenseNet121()
    elif args.model == 'densenet169':
        net = DenseNet169()
    elif args.model == "renext":
        net = ResNeXt29_2x64d()
    else:
        raise(NotImplementedError)


    # Try multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)




    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120, 160], gamma=0.1)
    early_stopper = EarlyStopper(patience=args.early_stop_patience)

    if args.resume:
        if not os.path.isfile(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}False.pth'):
            return
        checkpoint = torch.load(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}False.pth')
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        model_state_dict = {"module."+key: value for key, value in model_state_dict.items()}
        net.load_state_dict(model_state_dict)
        print("--------------loaded model state_dict---------")
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['acc']
        print("start from accuracy: ", best_acc)


    # Training
    def train_mlh(epoch, n_quantiles, scale=1):
        optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(10):
            net.train()
            optimizer2.zero_grad()
            q_loss = quantile_loss(model=net, device=device, n_quantiles=n_quantiles) * scale
            q_loss.backward()
            optimizer2.step()
            mlh = compute_kl(net)
            print('Train Epoch: {} mlh {:.5f} loss {:.5f}'.format(epoch, mlh, q_loss.item()))
        return mlh

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return train_loss / total , 100*correct/total


    def test(epoch, best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving best accuracy: ', acc)
            net_to_save = net.module if isinstance(net, nn.DataParallel) else net

            state = {
                'model_state_dict': net_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'loss': test_loss/total,
                'epoch': epoch
            }
            if not os.path.isdir(f'{save_dir}checkpoint'):
                os.mkdir(f'{save_dir}checkpoint')
            torch.save(state, f'{save_dir}checkpoint/ckpt_{args.model}_{args.seed}{args.benford}.pth')
            best_acc = acc
        mlh = compute_kl(net)

        return acc, test_loss/total, best_acc, mlh

    test_losss, test_accs = [], []

    if not args.benford:

        for epoch in range(start_epoch, start_epoch+args.epochs):
            train_loss, train_acc = train(epoch)
            test_acc, test_loss, best_acc, _ = test(epoch, best_acc)
            scheduler.step()
            test_losss.append(test_loss)
            test_accs.append(test_acc)
            if epoch % 10 == 0:
                np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losss))
                np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accs))

        np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losss))
        np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accs))

    else:

        test_losssb, test_accsb, mlhs = [], [], []
        benford_epochs = []
        train_benford = False
        if args.resume:
            train_benford = True

        for epoch in range(start_epoch, start_epoch+args.epochs):
            if train_benford:
                mlh = train_mlh(epoch, n_quantiles=100000, scale=args.scale)
                benford_epochs.append(epoch)
            else:
                train_loss, train_acc = train(epoch)
            test_acc, test_loss, best_acc, mlh = test(epoch, best_acc)
            train_benford = early_stopper.early_stop(test_loss)
            test_losssb.append(test_loss)
            test_accsb.append(test_acc)
            mlhs.append(mlh)
            if epoch % 10 == 0:
                np.save(f"{save_dir}test_loss_benford{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(test_losssb))
                np.save(f"{save_dir}test_accs_benford{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(test_accsb))
                np.save(f"{save_dir}benford_kl_{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(mlhs))
                np.save(f"{save_dir}benford_epochs_{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(benford_epochs))

        np.save(f"{save_dir}test_loss_benford{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(test_losssb))
        np.save(f"{save_dir}test_accs_benford{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(test_accsb))
        np.save(f"{save_dir}benford_epochs_{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(benford_epochs))
        np.save(f"{save_dir}benford_kl_{args.model}_{args.seed}_scale{args.scale}.npy", np.asarray(mlhs))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    available_models = ['PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', "renext"]
    parser.add_argument('--model', default='densenet121', help='model to train', choices=available_models)
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--seed', default=0, type=int, help='random seed', choices=np.arange(0,25).tolist())
    parser.add_argument('--early_stop_patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--benford', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--scale', default=1, type=float, help='scaling factor for the benford optimization')

    args = parser.parse_args()

    main(args=args)