"""Some helper functions for PyTorch and Benford regularization, including:
    - benford : the benford distribution
    - discrete_kl: calculate the discrete Kullback-Leibler divergence
    - binpercent: calculate the percentage of significant digits
    - compute_kl: compute the kl divergence between benford and the significant digits of model weights
    - quantile loss: calculate differences between the quantiles of the uniform distribution and the log_10 mod 1 der model weights
    - diffmod1: differential method of the modulo 1 operation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

benford = np.array([30.1, 17.6, 12.5, 9.7,
                    7.9, 6.7, 5.8, 5.1, 4.6]
                   ) / 100


def discrete_kl(bin_percent):
    kl = -np.sum(benford * np.log(bin_percent / benford + 1e-6))
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


def quantile_loss(model_weights, device):
    n_quantiles = int(model_weights.shape[0])
    model_weights = log10mod1(torch.abs(model_weights)+1e-9)
    quantile_steps = torch.linspace(start=0, end=1, steps=n_quantiles).to(device)
    model_quantiles = torch.quantile(model_weights, quantile_steps)
    uniform_quantiles = quantile_steps
    loss = F.mse_loss(model_quantiles, uniform_quantiles)
    return loss

def log10mod1(x):
    return torch.remainder(torch.log10(x), 1)
