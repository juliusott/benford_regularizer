import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

import numpy as np
import random
import argparse
import os
from benford_regularizer import quantile_loss, compute_kl
from utils import progress_bar, EarlyStopper



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

best_acc = 0  # best test accuracy

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    

    save_dir = f'./experiments_transfer_learning/{args.model}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    set_seed(args.seed)

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    if args.model == "resnet18":
        net = resnet18(pretrained=True)
    elif args.model == "resnet34":
        net = resnet34(pretrained=True)

    elif args.model == "resnet50":
        net = resnet50(pretrained=True)

    elif args.model == "resnet101":
        net = resnet101(pretrained=True)

    elif args.model == "resnet152":
        net = resnet152(pretrained=True)

    else:
        raise(NotImplementedError)

    for name, layer in net.named_modules():
        layer.requires_grad = False

    num_ftrs = net.fc.in_features
    mlp = nn.Sequential( nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 10))
    net.fc = mlp

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    early_stopper = EarlyStopper(patience=args.early_stop_patience)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    def train_mlh(epoch, n_quantiles):
        optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(20):
            net.train()
            optimizer2.zero_grad()
            q_loss = quantile_loss(model=net, device=device, n_quantiles=n_quantiles)
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
            #scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return train_loss / total , 100*correct/total

    def test(epoch):
            global best_acc 
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
                print('Saving..')
                state = {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': acc,
                    'loss': test_loss/total,
                    'epoch': epoch
                }
                if not os.path.isdir(f'{save_dir}checkpoint'):
                    os.mkdir(f'{save_dir}checkpoint')
                torch.save(state, f'{save_dir}checkpoint/ckpt_{args.model}_{args.seed}_{args.benford}.pth')
                best_acc = acc

            return acc, test_loss/total

    test_losss, test_accs = [], []
    if not args.benford:

            for epoch in range(start_epoch, start_epoch+args.epochs):
                train_loss, train_acc = train(epoch)
                test_acc, test_loss = test(epoch)
                test_losss.append(test_loss)
                test_accs.append(test_acc)
                if epoch % 10 == 0:
                    np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losss))
                    np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accs))

            np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losss))
            np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accs))

    else:

        test_losssb, test_accsb = [], []
        benford_epochs = []
        train_benford = False

        for epoch in range(start_epoch, start_epoch+args.epochs):
            if train_benford:
                mlh = train_mlh(epoch, n_quantiles=100000)
                benford_epochs.append(epoch)
            else:
                train_loss, train_acc = train(epoch)
            
            test_acc, test_loss = test(epoch)
            train_benford = early_stopper.early_stop(test_loss)
            test_losssb.append(test_loss)
            test_accsb.append(test_acc)
            if epoch % 10 == 0:
                np.save(f"{save_dir}test_loss_benford_{args.model}2_{args.seed}.npy", np.asarray(test_losssb))
                np.save(f"{save_dir}test_accs_benford_{args.model}2_{args.seed}.npy", np.asarray(test_accsb))
                np.save(f"{save_dir}benford_epochs_{args.model}2_{args.seed}.npy", np.asarray(benford_epochs))

        np.save(f"{save_dir}test_loss_benford_{args.model}2_{args.seed}.npy", np.asarray(test_losssb))
        np.save(f"{save_dir}test_accs_benford_{args.model}2_{args.seed}.npy", np.asarray(test_accsb))
        np.save(f"{save_dir}benford_epochs_{args.model}2_{args.seed}.npy", np.asarray(benford_epochs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    available_models = [f'resnet{n}' for n in [18, 34, 50, 101, 152]]
    seeds = [64213, 96010, 20004, 69469, 92983, 96872, 94213, 96723, 42,
            3638, 76325, 14009,  6885, 3407, 84738, 58775, 82009, 72163,
        18833, 18632,  5817, 64279, 42826, 61553, 75118]
    parser.add_argument('--model', default='resnet18', help='model to train', choices=available_models)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--seed', default=0, type=int, help='random seed', choices=seeds)
    parser.add_argument('--early_stop_patience', default=5, type=int, help='early stopping patience')
    parser.add_argument('--benford', action='store_true')

    args = parser.parse_args()
