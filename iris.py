import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

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
from utils import mean_confidence_interval
from sklearn.metrics import accuracy_score


mpl.style.use("seaborn-deep")
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)



def main(benford=False, seed=42):
    torch.manual_seed(seed)
    X,y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=seed)
    X_train_t = torch.from_numpy(X_train).to(torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch.float32)
    X_val_t = torch.from_numpy(X_val).to(torch.float32)
    y_val_t = torch.from_numpy(y_val).to(torch.float32)
    X_test_t = torch.from_numpy(X_test).to(torch.float32)
    y_test_t = torch.from_numpy(y_test).to(torch.float32)

    print("training ", len(X_train), "validation ", len(X_val), "testing ", len(X_test))
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    trainloader = DataLoader(train_dataset, batch_size=128)
    valloader = DataLoader(val_dataset, batch_size=128)
    testloader = DataLoader(test_dataset, batch_size=128)

    device = "cpu"
    net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
            )

    net.to(device)
    best_state_dict = net.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    early_stopper = EarlyStopper(patience=10)

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

    def train_bl(epoch, scale=1):
        optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(10):
            net.train()
            optimizer2.zero_grad()
            q_loss = quantile_loss(model=net, device=device) * scale
            q_loss.backward()
            optimizer2.step()
            bl_kl = compute_kl(net)
            #print('Train Epoch: {} mlh {:.5f} loss {:.5f}'.format(epoch, bl_kl, q_loss.item()))
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
        
        bl_kl = compute_kl(net)
        return correct/total, test_loss/total, bl_kl

    best_acc = 110
    epochs = 1000
    best_epoch = 0
    train_benford = False
    bl_kls, val_accs, bl_epochs = [], [], []
    for epoch in range(0, epochs):
        if train_benford:
                _ = train_bl(epoch=epoch, scale=1)
                bl_epochs.append(epoch)
                print(f"benford {epoch}")
        else:
            train_acc, train_loss = train()
        val_acc, val_loss, bl_kl = eval()
        val_accs.append(val_loss)
        bl_kls.append(bl_kl)
        if benford:
            train_benford = early_stopper.early_stop(val_loss)
        if val_loss < best_acc:
            best_state_dict = net.state_dict()
            best_acc = val_loss
            best_epoch = epoch

    net.load_state_dict(best_state_dict)
    test_acc , test_loss = test()
    print(f" best acc {best_acc} test acc {test_acc} in epoch {best_epoch}")
    #print(f"test acc {test_acc} test loss {test_loss}")

    return test_acc, np.asarray(val_accs), np.asarray(bl_kls), bl_epochs


    #show_separation(save=True, name_to_save="benford")

if __name__ == "__main__":
    seeds = np.random.randint(0, int(1e6), size=(10,))

    test_accs = []
    seeds = [604859]
    for seed in seeds:
        print(seed)
        test_acc, val_accs_bl, bl_kls_bl, benford_epochs  = main(benford=True, seed=seed)
        test_accs.append(test_acc)
        

    mean, conf = mean_confidence_interval(test_accs)
    print(f"BENFORD: mean {np.mean(test_accs)} std {np.std(test_accs)} 95% {conf} best {np.amax(test_accs)} worst {np.amin(test_accs)}")
    test_accs = []
    for seed in seeds:
        test_acc, val_accs, bl_kls, _= main(benford=False, seed=seed)
        test_accs.append(test_acc)

    mean, conf = mean_confidence_interval(test_accs)
    print(f" mean {np.mean(test_accs)} std {np.std(test_accs)} 95% {conf} best {np.amax(test_accs)} worst {np.amin(test_accs)}")

    fig, (ax,ax1)  = plt.subplots(2,1, figsize=(6,4.5), sharex=True)
    #fig.tight_layout()
    ax.set_title("validation error")
    ax.plot(val_accs_bl, label="MLP + BL reg")
    #ax.scatter(np.argmin(val_accs_bl), val_accs_bl[np.argmin(val_accs_bl)], marker="^", label="min error BL reg", color="red", s=30)
    #ax.scatter(np.argmin(val_accs), val_accs_bl[np.argmin(val_accs)], marker="^", label="min error", color="orange", s=30)
    #ax1.scatter(benford_epochs, bl_kls_bl[benford_epochs])
    ax.plot(val_accs, label="MLP")
    ax1.set_title("BL KL")
    ax1.plot(bl_kls_bl)
    ax1.plot(bl_kls)
    ax.set_yscale("log")
    ax.set_ylim([0, 10**-2])
    ax1.set_yscale("log")
    ax.legend()
    ax1.legend()
    ax1.set_xlabel("epochs")
    fig.savefig("2-layerMLP.pdf")
    plt.show()
