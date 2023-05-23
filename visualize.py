import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
from utils.utils import mean_confidence_interval

mpl.style.use("seaborn-deep")
font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

mpl.rc('font', **font)

model_names = ["resnet18", "resnet34", "resnet50", "densenet121", "PreActresnet50"]

model_name = "densenet169"
model_name2 = "densenet169"

experiment_type = "experiments_transfer_learning" if "resnet" in model_name[:7] else "experiments"
experiment_type = "experiments/speech_commands"
epochs = 50  # if "resnet" in model_name[:7] else 200

from_checkpoint = False

checkpoint_seeds = [18833, 18632, 5817, 64279, 42826, 61553, 75118]
path_to_dir_check = f"/home/pplcount/users/ottj/benford_regularizer/{experiment_type}/{model_name}_from_checkpoint/"

path_to_dir = f"/home/pplcount/users/ottj/benford_regularizer/{experiment_type}/{model_name}/"
if "speech_commands" in experiment_type:
    path_to_dir = f"/home/pplcount/users/ottj/benford_regularizer/{experiment_type}/"
    model_name = ""
    model_name2 = ""

test_loss = []
test_loss_benford = []
test_acc = []
test_acc_benford = []
benford_epochs_single_seed = []
test_acc_single_seed = []
test_acc_benford_single_seed = []

seed = 5817

baseline = []
if from_checkpoint:
    for path in os.listdir(path_to_dir):
        for seed in checkpoint_seeds:
            if f"test_accs_benford{model_name2}" in path and str(seed) in path:
                data = np.load(path_to_dir_check + path)
                d = np.amax(data)
                benford_epochs_path = f"benford_epochs_{model_name2}_{seed}.npy"
                benford_epochs = np.load(path_to_dir_check + benford_epochs_path)
                benford_epochs_baseline = np.load(path_to_dir + benford_epochs_path)
                print("from checkpoint", d)
                test_acc_benford.append(d)
                baseline = np.amax(np.load(path_to_dir + path))
                test_acc.append(baseline)
                print(path, "baseline acc", baseline, "baseline from epoch", benford_epochs_baseline[0])
                print("improvement", d - baseline)

    improve = np.asarray(test_acc_benford) - np.asarray(test_acc)
    mean, interval = mean_confidence_interval(improve)
    print(f"mean {mean} interval {interval}")
    print(f"IMPROveMENT : mean {np.mean(improve)}  std {np.std(improve)} max {np.amax(improve)} min {np.amin(improve)}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), sharey=True)
    ax.plot(np.ones(data.shape) * (1 - baseline / 100), label="Baseline Test Error", color="orange")
    ax.plot(1 - data / 100, label="Benford optimization")
    ax.scatter(benford_epochs, 1 - data[benford_epochs] / 100, marker="v", label="Benford epochs")
    ax.set_xlabel("epochs")
    ax.set_ylabel("test_error")
    ax.set_xlabel("epochs")
    plt.show()
    sys.exit()

for path in os.listdir(path_to_dir):
    if f"test_loss_benford_{model_name2}" in path or f"test_loss_benford{model_name2}" in path:
        test_loss_benford.append(np.load(path_to_dir + path))

    elif f"benford_epochs_{model_name2}" in path and str(seed) in path or f"benford_epochs{model_name2}" in path:
        benford_epochs_single_seed = np.load(path_to_dir + path)

    elif f"test_accs_benford_{model_name2}" in path or f"test_accs_benford{model_name2}" in path:
        d = np.load(path_to_dir + path)
        if np.amax(d) > 6:
            test_acc_benford.append(d)
        print(path, "acc", np.amax(np.load(path_to_dir + path)))
        if str(seed) in path:
            print("laod seed", seed)
            test_acc_benford_single_seed = np.load(path_to_dir + path)

    elif f"test_loss_{model_name}" in path:
        test_loss.append(np.load(path_to_dir + path))

    elif f"test_accs_{model_name}" in path:
        d = np.load(path_to_dir + path)
        if np.amax(d) > 6:
            test_acc.append(d)
        print(path, "acc", np.amax(np.load(path_to_dir + path)))
        if str(seed) in path:
            test_acc_single_seed = np.load(path_to_dir + path)

print("single test loss ", test_loss[0].shape)
test_loss = np.concatenate([x.reshape(1, *x.shape) for x in test_loss if x.shape[0] == epochs], axis=0)
test_acc = np.concatenate([x.reshape(1, *x.shape) for x in test_acc if x.shape[0] == epochs], axis=0)
test_loss_benford = np.concatenate([x.reshape(1, *x.shape) for x in test_loss_benford if x.shape[0] == epochs], axis=0)
test_acc_benford = np.concatenate([x.reshape(1, *x.shape) for x in test_acc_benford if x.shape[0] == epochs], axis=0)

print("worst NO REG", 100 - np.amin(np.amax(test_acc, axis=1)))
print("worst REG", 100 - np.amin(np.amax(test_acc_benford, axis=1)))
mean, interval = mean_confidence_interval(np.amax(test_acc, axis=1))
print(f"NO REG : mean {100 - mean} +- {interval} STD {np.std(np.amax(test_acc, axis=1))}")

mean, interval = mean_confidence_interval(np.amax(test_acc_benford, axis=1))
print(f"BENFORD REG : mean {100 - mean} +- {interval} STD {np.std(np.amax(test_acc_benford, axis=1))}")

print(f"BEST {100 - np.amax(test_acc)} benford {100 - np.amax(test_acc_benford)}")

print("all test loss ", test_loss.shape, test_loss_benford.shape)
print("benford epochs", test_acc_benford_single_seed)

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), sharey=True)
ax.plot(100 - test_acc_single_seed, label="ERM", color="orange")
ax.plot(100 - test_acc_benford_single_seed, label="benford regularized")
ax.set_xlabel("epochs")
ax.set_ylabel("test_error")
ax.set_xlabel("epochs")
ax.scatter(benford_epochs_single_seed, 100 - test_acc_benford_single_seed[benford_epochs_single_seed], marker="v",
           label="benford iterations")

ax.legend()

fig.suptitle(f"{model_name} Cifar10")
# fig.savefig(f"{model_name}_cifar10.png")
plt.show()
