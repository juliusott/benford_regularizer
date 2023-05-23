import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.preact_resnet import PreActResNet101, PreActResNet50
from models.densenet import DenseNet121, DenseNet169
from models.resnext import ResNeXt29_2x64d
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn.functional as F
import torch.nn as nn
import argparse
from benford_regularizer import bin_percent
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import os

mpl.style.use("seaborn-deep")
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)


benford = np.array([30.1, 17.6, 12.5, 9.7,
                    7.9, 6.7, 5.8, 5.1, 4.6]
                   ) / 100

def get_bins(model, benford):
    if model == 'PreActresnet101':
        net = PreActResNet101()
    elif model == 'PreActresnet50':
        net = PreActResNet50()
    elif model == 'densenet121':
        net = DenseNet121()
    elif model == 'densenet169':
        net = DenseNet169()
    elif model == "renext":
        net = ResNeXt29_2x64d()
    elif model == "resnet18":
        net = resnet18()
    elif model == "resnet34":
        net = resnet34()

    elif model == "resnet50":
        net = resnet50()

    elif model == "resnet101":
        net = resnet101()

    elif model == "resnet152":
        net = resnet152()
    else:
        raise(NotImplementedError)

    if "resnet" in model[:7]:
        num_ftrs = net.fc.in_features
        mlp = nn.Sequential( nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 10))
        net.fc = mlp
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    

    if "resnet" in model[:7]:
        save_dir = f'./experiments_transfer_learning/{model}/checkpoint/'
    else:
        save_dir = f'./experiments/{model}/checkpoint/'

    if benford:
        checkpoint_file = str([f for f in os.listdir(save_dir) if "True" in f][0])
    else:
        checkpoint_file = str([f for f in os.listdir(save_dir) if "False" in f][0])

    print(save_dir+checkpoint_file)
    checkpoint = torch.load(save_dir+checkpoint_file, map_location=torch.device('cpu'))
    acc = checkpoint['acc']
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
    net.load_state_dict(model_state_dict)
    net = net.to(device)
    
    model_weights = []
    for name, param in net.named_parameters():
        model_weights.append(torch.flatten(param))
    model_weights = torch.cat(model_weights, dim=0)
    b = bin_percent(model_weights)
    hist= b.numpy()[1:]
    return hist

def main(args):

    histograms = {"BL" : benford}
    weights = np.empty((9, len(args.model*2)+1))
    for i, model in enumerate(args.model):
        histograms[model+"+ BL opt"] = get_bins(model, benford=True)
        histograms[model] = get_bins(model, benford=False)

    i=0
    for key, value in histograms.items():
        print(i)
        weights[:,i] = value
        i +=1

    n_bins = 9
    bins = np.concatenate([np.arange(1,10).reshape((-1,1)) for _ in range(weights.shape[1])], axis=1)
    print(histograms)
    print(weights)
    print(weights.shape)
    print(bins.shape)
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.hist(x=bins, bins= n_bins, weights=weights, label=list(histograms.keys()))
    ax.set_xlabel("digit $n$")
    ax.set_ylabel("Occurence of digit $n$ as first digit in $\%$")
    ax.legend()
    fig.savefig("digits_of_trained_networks.pdf")
    plt.show()

    print(histograms)


    



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    available_models = ['PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', "renext"] + [f"resnet{x}" for x in [18 , 34, 50]]
    parser.add_argument('--model', default='densenet121', help='model to evaluate', choices=available_models)
    parser.add_argument('--benford', action='store_true')

    args = parser.parse_args()

    args.model = ['densenet121']
    main(args=args)