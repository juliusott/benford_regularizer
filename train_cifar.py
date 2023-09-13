"""Train CIFAR with PyTorch."""
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from torchvision.models import vit_b_16

from models.densenet import *
from models.preact_resnet import *
from models.resnext import *
from utils.benford_regularizer import quantile_loss, compute_kl
from utils.utils import EarlyStopper
from utils.utils import progress_bar
from transformers import get_cosine_schedule_with_warmup

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


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

    g = torch.Generator()
    g.manual_seed(args.seed)
    set_seed(args.seed)

    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
        args.num_classes = 10
    
    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100

    if "vit" in args.model:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224,224), scale=(0.08,  1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(0, 9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

        ])

        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)

    val_size = 10000
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if args.data_size < 1.0:
        print("old number of training data: " , len(trainset))
        K = int(args.data_size * len(trainset))
        subsample_train_indices = torch.randperm(len(trainset))[:K]
        trainset = torch.utils.data.Subset(trainset, indices=subsample_train_indices)
        print("new number of training data: ", len(trainset))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=2)

    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    if args.finetune:
        save_dir = f'./experiments/{args.model}_from_checkpoint/'
    else:
        save_dir = f'./experiments/{args.model}/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Model
    print(f'==> Building model: {args.model}')

    if args.model == 'PreActresnet101':
        net = PreActResNet101(num_classes = args.num_classes)
    elif args.model == 'PreActresnet50':
        net = PreActResNet50(num_classes = args.num_classes)
    elif args.model == 'densenet121':
        net = DenseNet121(num_classes = args.num_classes)
    elif args.model == 'densenet169':
        net = DenseNet169(num_classes = args.num_classes)
    elif args.model == 'densenet201':
        net = DenseNet201(num_classes = args.num_classes)
    elif args.model == "renext":
        net = ResNeXt29_2x64d(num_classes = args.num_classes)
    elif args.model == 'vitb16':
        net = vit_b_16(num_classes = args.num_classes)
    else:
        raise NotImplementedError

    model_weights = []
    n_layers = 0
    for _, param in net.named_parameters():
        model_weights.append(torch.flatten(param))
        n_layers += 1
    model_weights = torch.cat(model_weights, dim=0)

    args.scale = 1/n_layers * 0.5

    print(f"number of parameters {int(model_weights.shape[0])}")

    # Try multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    

    if "vit" in args.model:
        optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0)
        criterion = SoftTargetCrossEntropy()
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

    if "vit" in args.model:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=282, num_warmup_steps=10)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


    if args.resume:
        if os.path.isfile(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}False.pth'):
            checkpoint = torch.load(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}False.pth')
        elif os.path.isfile(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}True.pth'):
            checkpoint = torch.load(f'./experiments/{args.model}/checkpoint/ckpt_{args.model}_{args.seed}True.pth')
        else:
            return
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        if isinstance(net, nn.DataParallel):
            model_state_dict = {"module." + key: value for key, value in model_state_dict.items()}
        net.load_state_dict(model_state_dict)
        print("--------------loaded model state_dict---------")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("start from accuracy: ", best_acc)

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        # Do training
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if "vit" in args.model:
                y_hat = F.one_hot(targets, num_classes=args.num_classes)
            else:
                y_hat = targets
            loss = criterion(outputs, y_hat)
            if args.benford and epoch > args.benford_iter:
                for name, param in net.named_parameters():
                    if "bias" not in name and args.exclude_bias:
                        loss += args.scale * quantile_loss(torch.flatten(param), device)
                    else:
                        loss += args.scale * quantile_loss(torch.flatten(param), device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        scheduler.step()

        return train_loss / total, 100 * correct / total

    def eval(epoch, best_acc, force_store=False):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        # Do evaluation
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                if "vit" in args.model:
                    y_hat = F.one_hot(targets, num_classes=args.num_classes)
                else:
                    y_hat = targets
                loss = criterion(outputs, y_hat)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc or force_store:
            print('Saving best accuracy: ', acc)
            net_to_save = net.module if isinstance(net, nn.DataParallel) else net

            state = {
                'model_state_dict': net_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
                'loss': test_loss / total,
                'epoch': epoch
            }
            if not os.path.isdir(f'{save_dir}checkpoint'):
                os.mkdir(f'{save_dir}checkpoint')
            torch.save(state, f'{save_dir}checkpoint/ckpt_{args.model}_{args.seed}{args.benford}.pth')
            best_acc = acc
        bl_kl = compute_kl(net)

        return acc, test_loss / total, best_acc, bl_kl

    def test():
        # Compute the test metric for the model with the best validation score
        checkpoint = torch.load(f'{save_dir}checkpoint/ckpt_{args.model}_{args.seed}{args.benford}.pth')
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        if isinstance(net, nn.DataParallel):
            model_state_dict = {"module." + key: value for key, value in model_state_dict.items()}
        net.load_state_dict(model_state_dict)
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                if "vit" in args.model:
                    y_hat = F.one_hot(targets, num_classes=args.num_classes)
                else:
                    y_hat = targets
                loss = criterion(outputs, y_hat)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total

        return acc, test_loss / total

    val_losss, val_accs, bl_kls = [], [], []


    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train(epoch)
        val_acc, val_loss, best_acc, bl_kl = eval(epoch, best_acc)
        val_losss.append(val_loss)
        val_accs.append(val_acc)
        bl_kls.append(bl_kl)
        if epoch % 10 == 0:
            np.save(f"{save_dir}test_loss_{args.model}_{args.seed}_BL:{args.benford}.npy", np.asarray(val_losss))
            np.save(f"{save_dir}test_accs_{args.model}_{args.seed}_BL:{args.benford}.npy", np.asarray(val_accs))
            np.save(f"{save_dir}benford_kl_{args.model}_{args.seed}_BL:{args.benford}.npy",
                    np.asarray(bl_kls))

        test_acc, test_loss = test()
        print(f"test acc {test_acc}, test_loss {test_loss}")
        np.save(f"{save_dir}test_loss_acc_{args.model}_{args.seed}.npy", np.asarray([test_acc, test_loss]))
        np.save(f"{save_dir}test_loss_{args.model}_{args.seed}_BL:{args.benford}.npy", np.asarray(val_losss))
        np.save(f"{save_dir}test_accs_{args.model}_{args.seed}_BL:{args.benford}.npy", np.asarray(val_accs))
        np.save(f"{save_dir}benford_kl_{args.model}_{args.seed}_BL:{args.benford}.npy", np.asarray(bl_kls))
