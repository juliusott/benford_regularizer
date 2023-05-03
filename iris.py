import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs, load_iris
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from benford_regularizer import compute_kl, quantile_loss
from utils import progress_bar
from utils import EarlyStopper
from sklearn.metrics import accuracy_score


sns.set(style="darkgrid", font_scale=1.4)



def main():
    X,y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train_t = torch.from_numpy(X_train).to(torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch.float32)
    X_val_t = torch.from_numpy(X_val).to(torch.float32)
    y_val_t = torch.from_numpy(y_val).to(torch.float32)
    X_test_t = torch.from_numpy(X_test).to(torch.float32)
    y_test_t = torch.from_numpy(y_test).to(torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    trainloader = DataLoader(train_dataset, batch_size=128)
    valloader = DataLoader(val_dataset, batch_size=128)
    testloader = DataLoader(test_dataset, batch_size=128)

    device = "cpu"
    net = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 3)
            )

    net.to(device)
    init_state_dict = net.state_dict()
    best_state_dict = net.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    def show_separation(save=False, name_to_save=""):
        sns.set(style="white")

        xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        batch = torch.from_numpy(grid).type(torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(net(batch).reshape(xx.shape))
            probs = probs.numpy().reshape(xx.shape)

        f, ax = plt.subplots(figsize=(16, 10))
        ax.set_title("Decision boundary", fontsize=14)
        contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                            vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
                cmap="RdBu", vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)

        ax.set(xlabel="$X_1$", ylabel="$X_2$")
        if save:
            plt.savefig(name_to_save)
        else:
            plt.show()

    def train_bl(epoch, n_quantiles, scale=1):
        optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(3):
            net.train()
            optimizer2.zero_grad()
            q_loss = quantile_loss(model=net, device=device, n_quantiles=n_quantiles) * scale
            q_loss.backward()
            optimizer2.step()
            bl_kl = compute_kl(net)
            print('Train Epoch: {} mlh {:.5f} loss {:.5f}'.format(epoch, bl_kl, q_loss.item()))
        return bl_kl

    def train():
        total = 0
        correct = 0
        train_loss = 0
        for it, (X_batch, y_batch) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        return correct/total, train_loss/total

    @torch.no_grad()
    def test():
        net.load_state_dict(best_state_dict)
        net.eval()
        total = 0
        correct = 0
        test_loss = 0
        for x_batch, targets in testloader:
            outputs = net(x_batch)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct/total, test_loss/total

    @torch.no_grad()
    def eval():
        net.eval()
        total = 0
        correct = 0
        test_loss = 0
        for x_batch, targets in valloader:
            outputs = net(x_batch)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct/total, test_loss/total

    epochs = 15
    best_acc = 0
    for epoch in range(0,epochs):
        train_acc, train_loss = train()
        test_acc, test_loss = eval()
        if test_acc > best_acc:
            best_state_dict = net.state_dict()
            best_acc = test_acc
    test_acc, test_loss = test()
    print(f"test acc {test_acc} test loss {test_loss}")

    net.load_state_dict(init_state_dict)

    best_acc = 0
    for epoch in range(0, epochs):
        if epoch > 5 and epoch % 5 == 0:
            _ = train_bl(epoch=epoch ,n_quantiles=10000, scale=0.1)
        else:
            train_acc, train_loss = train()
        test_acc, test_loss = test()
        if test_acc > best_acc:
            best_state_dict = net.state_dict()
            best_acc = test_acc
        test_acc, test_loss = test()
    print(f"test acc {test_acc} test loss {test_loss}")


    #show_separation(save=True, name_to_save="benford")

if __name__ == "__main__":
    main()