import matplotlib.pyplot as plt
import numpy as np
import os

model_names = ["DenseNet121", "PreActResNet18"]

model_name = model_names[0]

test_loss_path = f"test_loss_{model_name}.npy"
test_loss_benford_path = f"test_loss_{model_name}_benford.npy"
benford_epochs_path = f"test_accs_{model_name}_benford_epochs.npy"

test_loss = np.load(test_loss_path)
test_loss_benford = np.load(test_loss_benford_path)
benford_epochs = np.load(benford_epochs_path)

fig, ax = plt.subplots(1,2, figsize=(8,5), sharey=True)
print(f"min test loss {np.amin(test_loss)} min test loss benford {np.amin(test_loss_benford)}")
ax[0].plot(test_loss, label="ERM", color="orange")
ax[0].plot(range(len(test_loss)-len(test_loss_benford), len(test_loss)),test_loss_benford, label="benford regularized", color="blue")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[1].set_xlabel("epochs")
ax[1].plot(range(len(test_loss)-len(test_loss_benford), len(test_loss)), test_loss[-len(test_loss_benford):], label="ERM", color="orange")
ax[1].plot(range(len(test_loss)-len(test_loss_benford), len(test_loss)),test_loss_benford, label="benford regularized", color="blue")
ax[1].scatter(benford_epochs, test_loss_benford[benford_epochs-benford_epochs[0]], marker="v", label="benford iterations", color="green")

ax[0].legend()
ax[1].legend()

fig.suptitle(f"{model_name} Cifar10", fontsize=16)
fig.savefig(f"{model_name}_cifar10.png")
plt.show()