import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

benford = np.array([30.1, 17.6, 12.5, 9.7,
                    7.9, 6.7, 5.8, 5.1, 4.6]
                   ) / 100

def discrete_kl(bin_percent):
    kl = -np.sum(benford * np.log(bin_percent/benford + 1e-6))
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

def quantile_loss(model, device):
    max_weights = int(1e6)
    model_weights = []
    for _, param in model.named_parameters():
        model_weights.append(torch.flatten(param))
    model_weights = torch.cat(model_weights, dim=0)
    if int(model_weights.shape[0]) > max_weights:
        idx = torch.randperm(int(model_weights.shape[0]))[:max_weights].to(device)
        model_weights = model_weights[idx]
    n_quantiles = int(model_weights.shape[0])
    model_weights = diffmod1(torch.log10(torch.abs(model_weights)+1e-6), device)
    quantile_steps = torch.linspace(start=0,end=1, steps=n_quantiles).to(device)
    model_quantiles = torch.quantile(model_weights, quantile_steps)
    loss = F.mse_loss(model_quantiles, quantile_steps)
    return loss

def diffmod1(x, device):
    pi = torch.Tensor([np.pi]).to(device)
    x = pi * x
    y = torch.atan(-1.0 / (torch.tan(x))) + 0.5 * pi
    y = 1 / pi * y
    return y




def train_bl(model, device, optimizer, epoch, n_quantiles):
    model.train()
    optimizer.zero_grad()
    q_loss = quantile_loss(model, device, n_quantiles)
    q_loss.backward()
    optimizer.step()
    bl_kl = compute_kl(model)
    print('Train Epoch: {} mlh {:.5f} loss {:.5f}'.format(epoch, bl_kl, q_loss.item()))
    return bl

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


