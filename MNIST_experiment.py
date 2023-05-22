import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from utils.utils import progress_bar
from utils.benford_regularizer import compute_kl
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl


mpl.style.use("seaborn-deep")
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

class CNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier."""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def main():

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transform_train)
    K = 6000 # enter your length here
    subsample_train_indices = torch.randperm(len(trainset))[:K]
    #trainset = torch.utils.data.Subset(trainset, indices=subsample_train_indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=200, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    print(f"len train {len(trainset)} test {len(testset)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = CNNClassifier()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    best_acc = 0


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
            loss = F.nll_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            # scheduler.step()
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
                loss = F.nll_loss(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


        layer_means = []
        layer_vars= []
        for param in net.parameters():
            param = param.view(-1).detach().cpu().numpy()
            mean = np.mean(param)
            var = np.var(param)
            layer_means.append(mean)
            layer_vars.append(var)
        layer_info = (np.asarray(layer_means), np.asarray(layer_vars))
        sdl_kl = compute_kl(net)
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
            print("best accuracy achieved! :)")

        return acc, test_loss/total, best_acc, sdl_kl, layer_info

    test_losss, test_accs, sdl_kls = [], [], []
    num_epochs = 1000
    layer_mean = []
    layer_var = []
    for epoch in range(0, num_epochs):
        train_loss, train_acc= train(epoch)
        test_acc, test_loss, best_acc, sdl_kl, layer_info = test(epoch, best_acc)
        layer_m, layer_v = layer_info
        layer_mean.append(np.mean(layer_m))
        layer_var.append(np.mean(layer_v))
        test_losss.append(test_loss)
        test_accs.append(test_acc)
        sdl_kls.append(sdl_kl)

    seed = torch.seed()
    np.save(f"./experiments/mnist/test_loss_{seed}.npy", np.asarray(test_losss))
    np.save(f"./experiments/mnist/test_acc_{seed}.npy", np.asarray(test_accs))
    np.save(f"./experiments/mnist/sdl_kl_{seed}.npy", np.asarray(sdl_kls))
    np.save(f"./experiments/mnist/layer_mean_{seed}.npy", np.asarray(layer_mean))
    np.save(f"./experiments/mnist/layer_mean_{seed}.npy", np.asarray(layer_var))

    fig,(ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    ax1.plot(test_losss)
    ax1.set_title("test loss")
    ax2.plot(1- np.asarray(test_accs)/100)
    ax2.set_title("test accuracy")
    ax3.plot(sdl_kls)
    ax3.set_title("SDL KL")
    ax4.plot(layer_mean, label="layer mean")
    ax4.plot(layer_var, label="layer var")
    ax4.legend()
    plt.show()

def visualize():
    files = os.listdir("./experiments/mnist/")
    test_loss_files = [file for file in files if "test_loss" in file]
    seeds = [int(file[:-4].replace("test_loss_", "")) for file in test_loss_files]

    test_loss = np.concatenate([np.expand_dims(np.load(f"./experiments/mnist/test_loss_{seed}.npy"), axis=1) for seed in seeds], axis=1)
    print(test_loss.shape)
    test_acc = np.concatenate([np.expand_dims(np.load(f"./experiments/mnist/test_acc_{seed}.npy"),axis=1) for seed in seeds], axis=1)
    test_acc[test_acc<85] = np.nan
    sdl_kl = np.concatenate([np.expand_dims(np.load(f"./experiments/mnist/sdl_kl_{seed}.npy"),axis=1) for seed in seeds],axis=1)

    fig,(ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(6,4.5))
    ax1.set_ylim([0,0.08])
    ax1.plot(1- test_acc[:,0]/100, label="MNIST 60000")
    ax1.plot(1- test_acc[:,1]/100, label="MNIST 6000")
    ax1.plot(1- test_acc[:,2]/100, label="MNIST 600")
    ax1.set_title("test  error in $\%$")
    ax2.plot(sdl_kl[:,0], label="MNIST 60000")
    ax2.plot(sdl_kl[:,1], label="MNIST 6000")
    ax2.plot(sdl_kl[:,2], label="MNIST 600")
    ax2.legend()
    ax2.set_title("KL between BL/CNN")
    ax1.set_xlabel("epochs")
    ax2.set_xlabel("epochs")
    fig.savefig("kl_test_loss.pdf")
    plt.show()


if __name__ == "__main__":
    main()
    visualize()