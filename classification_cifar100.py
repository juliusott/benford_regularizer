'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import os
import argparse
from benford_regularizer import quantile_loss, compute_kl
from models.densenet import *
from models.preact_resnet import *
from models.resnext import *
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                return True
        return False


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='resnet18', help='model to train', choices=[f'resnet{n}' for n in [18, 34, 50, 101, 152]])
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--seed', default=0, help='random seed', choices=np.arange(0,25).tolist())
parser.add_argument('--early_stop_patience', default=15, type=int, help='early stopping patience')
parser.add_argument('--benford', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

save_dir = f'./experiments/{args.model}/'
# Model
print(f'==> Building model: {args.model}')

if args.model == 'resnet18':
    net = resnet18(num_classes=100)
elif args.model == 'resnet34':
    net = resnet34(num_classes=100)
elif args.model == 'resnet50':
    net = resnet50(num_classes = 100)
elif args.model == 'resnet101':
    net = resnet101(num_classes=100)
elif args.model == 'resnet152':
    net = resnet152(num_classes=100)
else:
    raise(NotImplementedError)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
early_stopper = EarlyStopper(patience=args.early_stopping_patience)


# Training
def train_mlh(epoch, n_quantiles):
    optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(5):
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'{save_dir}checkpoint/ckpt_{args.model}_{args.seed}.pth')
        best_acc = acc

    return acc, test_loss/total

test_losss, test_accs = [], []

if args.benford:

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_loss, train_acc = train(epoch)
        test_acc, test_loss = test(epoch)
        # train_benford = early_stopper.early_stop(test_loss)
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
    train_benford = True

    for epoch in range(best_epoch, start_epoch+args.epochs):
        if train_benford:
            mlh = train_mlh(epoch, n_quantiles=1)
            benford_epochs.append(epoch)
        else:
            train_loss, train_acc = train(epoch)
        test_acc, test_loss = test(epoch)
        train_benford = early_stopper.early_stop(test_loss)
        test_losssb.append(test_loss)
        test_accsb.append(test_acc)
        if epoch % 10 == 0:
            np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losssb))
            np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accsb))
            np.save(f"{save_dir}benford_epochs_{args.model}_{args.seed}.npy", np.asarray(benford_epochs))

    np.save(f"{save_dir}test_loss_{args.model}_{args.seed}.npy", np.asarray(test_losssb))
    np.save(f"{save_dir}test_accs_{args.model}_{args.seed}.npy", np.asarray(test_accsb))
    np.save(f"{save_dir}benford_epochs_{args.model}_{args.seed}.npy", np.asarray(benford_epochs))