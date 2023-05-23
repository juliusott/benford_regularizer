import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.speechcommands import SubsetSC
from utils.benford_regularizer import compute_kl, quantile_loss
from utils.utils import progress_bar, check_positive
import argparse
import torchaudio
import numpy as np
import os
from utils.utils import EarlyStopper
import random

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
          'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
          'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
          'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main(args):
    print(args)
    set_seed(args.seed)
    train_dataset = SubsetSC("./data", "training")
    val_dataset = SubsetSC("./data", "validation")
    test_dataset = SubsetSC("./data", "testing")
    # print(train_dataset.shape, test_dataset.shape)

    batch_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    new_sample_rate = 8000
    waveform, sample_rate, label, speaker_id, utterance_number = train_dataset[0]
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate_fn, num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers, pin_memory=pin_memory,
                                             )

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                            collate_fn=collate_fn,
                                            num_workers=num_workers, pin_memory=pin_memory,
                                            )

    net = M5(n_input=transformed.shape[0], n_output=len(labels))
    net.to(device)
    transform = transform.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    early_stopper = EarlyStopper(patience=args.early_stop_patience)

    best_acc = 0
    best_state_dict = net.state_dict()
    save_dir = f'./experiments/speech_commands/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            inputs = transform(inputs)
            outputs = net(inputs)
            loss = F.nll_loss(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            pred = get_likely_index(outputs)
            correct += number_of_correct(pred, targets)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return train_loss / total, 100 * correct / total

    def train_bl(epoch, n_quantiles, scale=1):
        optimizer2 = optim.Adam(net.parameters(), lr=1e-3)
        for i in range(args.benford_iter):
            net.train()
            optimizer2.zero_grad()
            q_loss = quantile_loss(model=net, device=device) * scale
            q_loss.backward()
            optimizer2.step()
            bl_kl = compute_kl(net)
            progress_bar(i, 10, 'Loss: %.3f '
                         % bl_kl)
        return bl_kl

    def eval(best_acc, best_state_dict):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = transform(inputs)
                outputs = net(inputs)
                loss = F.nll_loss(outputs.squeeze(), targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                pred = get_likely_index(outputs)
                correct += number_of_correct(pred, targets)

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        sdl_kl = compute_kl(net)
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
            best_state_dict = net.state_dict()
            print("best accuracy achieved! :)")

        return acc, test_loss / total, best_acc, sdl_kl, best_state_dict

    def test(best_state_dict):
        net.load_state_dict(best_state_dict)
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = transform(inputs)
                outputs = net(inputs)
                loss = F.nll_loss(outputs.squeeze(), targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                pred = get_likely_index(outputs)
                correct += number_of_correct(pred, targets)

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        return acc, test_loss / total

    val_losss, val_accs, bl_kls = [], [], []

    if not args.benford:

        for epoch in range(0, args.epochs):
            train_loss, train_acc = train(epoch)
            val_acc, val_loss, best_acc, bl_kl, best_state_dict = eval(best_acc, best_state_dict)
            scheduler.step()
            val_losss.append(val_loss)
            val_accs.append(val_acc)
            bl_kls.append(bl_kl)
            if epoch % 10 == 0:
                np.save(f"{save_dir}test_loss_{args.seed}.npy", np.asarray(val_losss))
                np.save(f"{save_dir}test_accs_{args.seed}.npy", np.asarray(val_accs))
                np.save(f"{save_dir}benford_kl_{args.seed}_scale{args.scale}_{args.benford}.npy", np.asarray(bl_kls))

        test_acc, test_loss = test(best_state_dict=best_state_dict)
        print(f"test acc {test_acc}, test_loss {test_loss}")
        np.save(f"{save_dir}test_loss_acc_{args.seed}.npy", np.asarray([test_acc, test_loss]))
        np.save(f"{save_dir}test_loss_{args.seed}.npy", np.asarray(val_losss))
        np.save(f"{save_dir}test_accs_{args.seed}.npy", np.asarray(val_accs))
        np.save(f"{save_dir}benford_kl_{args.seed}_scale{args.scale}_{args.benford}.npy", np.asarray(bl_kls))

    else:

        val_losssb, val_accsb, bl_kls = [], [], []
        benford_epochs = []
        train_benford = False
        if args.resume:
            train_benford = True

        for epoch in range(0, args.epochs):
            if train_benford:
                bl_kl = train_bl(epoch, n_quantiles=100000, scale=args.scale)
                benford_epochs.append(epoch)
            else:
                train_loss, train_acc = train(epoch)
            val_acc, val_loss, best_acc, bl_kl, best_state_dict = eval(best_acc, best_state_dict)
            scheduler.step()
            train_benford = early_stopper.early_stop(val_loss)
            val_losss.append(val_loss)
            val_accs.append(val_acc)
            bl_kls.append(bl_kl)
            if epoch % 10 == 0:
                np.save(f"{save_dir}test_loss_benford_{args.seed}_scale{args.scale}.npy", np.asarray(val_losssb))
                np.save(f"{save_dir}test_accs_benford_{args.seed}_scale{args.scale}.npy", np.asarray(val_accsb))
                np.save(f"{save_dir}benford_kl_{args.seed}_scale{args.scale}.npy", np.asarray(bl_kls))
                np.save(f"{save_dir}benford_epochs_{args.seed}_scale{args.scale}.npy", np.asarray(benford_epochs))
        test_acc, test_loss = test(best_state_dict=best_state_dict)
        print(f"test acc {test_acc}, test_loss {test_loss}")
        np.save(f"{save_dir}test_loss_benford_{args.seed}_scale{args.scale}.npy", np.asarray(val_losssb))
        np.save(f"{save_dir}test_accs_benford_{args.seed}_scale{args.scale}.npy", np.asarray(val_accsb))
        np.save(f"{save_dir}benford_kl_{args.seed}_scale{args.scale}.npy", np.asarray(bl_kls))
        np.save(f"{save_dir}benford_epochs_{args.seed}_scale{args.scale}.npy", np.asarray(benford_epochs))


if __name__ == "__main__":
    seeds = np.random.randint(0, int(1e6), size=(25,))
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--seed', nargs='*', type=int, choices=np.arange(0,int(1e6)).tolist())
    parser.add_argument('--early_stop_patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--benford', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--scale', default=1, type=float, help='scaling factor for the benford optimization')
    parser.add_argument('--benford_iter', default=10, type=check_positive, help='number of benford iterations')

    args = parser.parse_args()

    if args.seed is not None:
        seeds = args.seed

    for seed in seeds:
        args.seed = seed
        main(args)
