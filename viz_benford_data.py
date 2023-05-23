import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
import torch
import torchvision
import torchvision.transforms as transforms
from utils.speechcommands import SubsetSC, data_processing
import matplotlib as mpl

mpl.style.use("seaborn-deep")
font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}

mpl.rc('font', **font)

benford = np.array([30.1, 17.6, 12.5, 9.7,
                    7.9, 6.7, 5.8, 5.1, 4.6]
                   ) / 100

cifar100_dct = np.array([3.6891e-01, 1.9702e-01, 7.4292e-02, 5.8507e-02, 5.8646e-02,
                         5.9596e-02, 6.0672e-02, 6.1296e-02, 6.1060e-02])

cifar10_dct = np.array([3.8528e-01, 1.8939e-01, 6.5186e-02, 5.7383e-02, 5.7675e-02,
                        5.9007e-02, 6.0824e-02, 6.2321e-02, 6.2923e-02])

mnist_dct = np.array(([2.2217e-01, 1.4378e-01, 1.4299e-01, 1.3671e-01, 1.0497e-01,
                       8.4600e-02, 6.8126e-02, 5.3837e-02, 4.2829e-02]))

speech_fft = np.array([2.7007e-01, 1.7958e-01, 1.2984e-01, 1.0167e-01, 8.3500e-02,
                       7.0833e-02, 6.1484e-02, 5.4341e-02, 4.8686e-02])

print(cifar100_dct, cifar10_dct, speech_fft)


def dct2(a):
    return dct(dct(a.T, norm=None).T, norm=None)


def bincount(tensor):
    counts = torch.zeros(10)
    for i in range(10):
        counts[i] = torch.count_nonzero(
            tensor == i
        )
    return counts


def bin_percent(tensor):
    tensor = tensor.abs() * 1e10
    # print("first", tensor[0])
    long_tensor = torch.log10(tensor).long()
    # print("second", long_tensor[0])
    tensor = tensor // 10 ** long_tensor
    # print("third", tensor[0])
    tensor = bincount(tensor.abs().long())
    # print("fourth", tensor[0])
    return (tensor / tensor.sum()).numpy()


def get_digits(dataloader):
    digits = torch.zeros(10)
    for inputs, targets in dataloader:
        inputs = inputs.flatten() * 255.
        # print(f"min image {inputs.min()} max {inputs.max()}")
        dct_inputs = dct2(inputs.numpy())
        # print(dct_inputs.shape, inputs.shape)
        digits += bin_percent(torch.from_numpy(dct_inputs) * 1e3)

    return digits / len(dataloader)


transform = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == "__main__":
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    import torchaudio

    dataset = SubsetSC(".\data", "training")
    print(len(dataset))

    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
              'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
              'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
              'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    n_class = len(labels)
    n_fft = 512
    hop_length = 128
    n_feat = n_fft // 2 + 1
    stft_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             collate_fn=lambda x: data_processing(x, 'train', labels, stft_transform,
                                                                                  None))

    # print(dataloader[0])

    # digits = get_digits(dataloader)
    mean_images = (cifar100_dct + cifar10_dct + mnist_dct + speech_fft) / 4
    print(np.arange(1, 10), benford)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(np.arange(1, 10), bins=9, weights=benford * 100, label="Benford's Law")
    ax.plot(np.arange(1, 10), cifar10_dct * 100, label="DCT Cifar10", marker="<", alpha=0.6)
    ax.plot(np.arange(1, 10), cifar100_dct * 100, label="DCT Cifar100", marker=">", alpha=0.6)
    ax.plot(np.arange(1, 10), mnist_dct * 100, label="DCT MNIST", marker="^", alpha=0.6)
    ax.plot(np.arange(1, 10), speech_fft * 100, label="FFT Google Speech Commands", marker="s", alpha=0.6)
    ax.plot(np.arange(1, 10), mean_images * 100, label="All Datasets", marker="o", color="red")
    ax.set_xlabel("digit $n$")
    ax.set_ylabel("Occurence of digit $n$ as first digit in $\%$")
    plt.legend()
    fig.savefig("datasets.pdf")
    plt.show()
    # print(digits)
