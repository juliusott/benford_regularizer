import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from copy import deepcopy
import sys
import os

benford = np.array([30.1, 17.6, 12.5, 9.7,
                    7.9, 6.7, 5.8, 5.1, 4.6]
                   ) / 100

def discrete_kl(bin_percent):
    kl = -np.sum(benford * np.log(bin_percent/benford))
    return kl


def mlh(bin_percent):
    return pearsonr(benford, bin_percent[1:])[0]

def bincount(tensor):
    counts = torch.zeros(10)
    for i in range(10):
        counts[i] = torch.count_nonzero(
        tensor == i
        )   
    return counts

@torch.no_grad()
def bin_percent(tensor):
    tensor = tensor.abs() * 1e10
    long_tensor = torch.log10(tensor).long()
    tensor = tensor // 10 ** long_tensor
    tensor = bincount(tensor.long())
    return tensor / tensor.sum()

@torch.no_grad()
def compute_kl(model):
    model_weights = []
    for _name, param in model.named_parameters():
        model_weights.append(torch.flatten(param))
    model_weights = torch.cat(model_weights, dim=0)
    b = bin_percent(model_weights)
    kl_benford = discrete_kl(b.numpy()[1:])
    return kl_benford

def quantile_loss(model, device, n_quantiles):
    model_weights = []
    for _, param in model.named_parameters():
        model_weights.append(torch.flatten(param))
    model_weights = torch.cat(model_weights, dim=0)
    n_quantiles = int(model_weights.shape[0] * n_quantiles)
    model_weights = diffmod1(torch.log10(torch.abs(model_weights)), device)
    # uni = torch.distributions.uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))
    # uni_samples = uni.sample(sample_shape = model_weights.shape).to(device)
    quantile_steps = torch.linspace(start=0,end=1, steps=n_quantiles).to(device)
    model_quantiles = torch.quantile(model_weights, quantile_steps)
    # uniform_quantiles = torch.quantile(uni_samples, quantile_steps)
    loss = F.mse_loss(model_quantiles, quantile_steps)
    return loss

def diffmod1(x, device):
    pi = torch.Tensor([np.pi]).to(device)
    x = pi * x
    y = torch.atan(-1.0 / (torch.tan(x))) + 0.5 * pi
    y = 1 / pi * y
    return y




def train_mlh(model, device, optimizer, epoch, n_quantiles):
    model.train()
    optimizer.zero_grad()
    q_loss = quantile_loss(model, device, n_quantiles)
    q_loss.backward()
    optimizer.step()
    mlh = compute_mlh(model)
    print('Train Epoch: {} mlh {:.5f} loss {:.5f}'.format(epoch, mlh, q_loss.item()))
    return mlh

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))

    return test_loss, test_accuracy
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    best_model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    train_loader, test_loader = load_data()

    optimizer2 = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 25
    best_accuracy = 0
    early_stopping = 0
    mlhs = list()
    val_accuracy = list()
    test_losss = list()
    v_lines = []
    step = 0
    for epoch in range(1, n_epochs + 1):
            train_loss = train(model, device, train_loader, optimizer,scheduler, epoch)
            benf = compute_mlh(model)
            test_loss, test_accuracy = test(model, device, test_loader)
            mlhs.append(benf)
            val_accuracy.append(test_accuracy)
            test_losss.append(test_loss)
            step += 1
            if test_accuracy > best_accuracy:
                early_stopping = 0
                best_accuracy = test_accuracy
                #best_accuracy_state_dict = model.state_dict()
                #best_model = deepcopy(model)
                test_loss, test_acc = test(best_model, device, test_loader)
                print(f"best accuracy so far {test_acc} {best_accuracy} {test_accuracy}")
            else:
                early_stopping += 1
            
            if early_stopping >= 3:
                early_stopping = 0
                v_lines.append(step)
                v_lines.append(step+5)
                for i in range(5):
                    benf = train_mlh(model=model, device=device, optimizer=optimizer2, epoch=epoch, n_quantiles=100000)
                    test_loss, test_accuracy = test(model=model, device=device, test_loader=test_loader)
                    mlhs.append(benf)
                    val_accuracy.append(test_accuracy)
                    test_losss.append(test_loss)
                    step += 1

    print(v_lines)
    print("----DONE WITH TRAINING-----")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 7))
    ax1.plot(mlhs)
    ax1.set_title("MLH")
    ax2.plot(val_accuracy)
    ax2.set_title("validation accuracy")
    ax3.plot(test_losss)

    ax1.vlines(v_lines,min(mlhs),max(mlhs), linestyles="dashed")
    ax2.vlines(v_lines,min(val_accuracy),max(val_accuracy), linestyles="dashed")
    ax3.vlines(v_lines,min(test_losss),max(test_losss), linestyles="dashed")
    ax3.set_title("validation loss")
    fig.savefig("no_mlh_loss.png")



