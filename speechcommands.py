# from https://github.com/pytorch/audio/blob/main/torchaudio/datasets/speechcommands.py


import os
from pathlib import Path
from typing import Optional, Tuple, Union

from torch import Tensor
import torch
import os  
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchaudio

def _load_waveform(
    root: str,
    filename: str,
    exp_sample_rate: int,
):
    path = os.path.join(root, filename)
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return waveform

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SAMPLE_RATE = 16000
_CHECKSUMS = {
    "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def _get_speechcommands_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, SAMPLE_RATE, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    """*Speech Commands* :cite:`speechcommandsv2` dataset.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
    ) -> None:

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "http://download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path or set `download=True` to download it"
            )

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple of the following items;
            str:
                Path to the audio
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple of the following items;
            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)
    

class SubsetSC(SPEECHCOMMANDS):
    
    def __init__(self, data_path,subset: str = None,url="speech_commands_v0.02"):
        super().__init__(root=data_path,url=url,download=True)
        self.data_path = data_path
        self.folder_in_archive = "SpeechCommands"
        self.HASH_DIVIDER = "_nohash_"
        self.EXCEPT_FOLDER = "_background_noise_"

        folder_in_archive = os.path.join(self.folder_in_archive, url)
        self.path = os.path.join(data_path,folder_in_archive)
        print('self.path',self.path)
        def _load_list(root, *filenames):
            output = []
            for filename in filenames:
                filepath = os.path.join(root, filename)
                with open(filepath) as fileobj:
                    output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
            return output

        if subset == "validation":
            self._walker = _load_list(self.path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self.path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self.path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self.path).glob('*/*.wav'))

            self._walker = [
                w for w in walker
                if self.HASH_DIVIDER in w
                and self.EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
            ]

def path_index(dataset):
    path_to_index = {}
    index_to_path = {}
    index = 0
    for waveform, sample_rate, label, speaker_id, utterance_number in dataset:
        path = os.path.join(label,speaker_id +'_nohash_' + str(utterance_number) +'.wav' )
        path_to_index[path] = index
        index_to_path[index] = path
        index+=1
    return path_to_index,index_to_path

# from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
def label_to_index(word,labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

# from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
def index_to_label(index,labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

# From Nvidia Nemo
def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2

def data_processing(data,data_type,labels,stft_transform,args,path_to_index=None):
    spectrograms = []
    targets = []
    indexes = []

    for (waveform, sample_rate, target_text, speaker_id, utterance_number) in data: # waveform, sample_rate, label, speaker_id, utterance_number
        if data_type == 'train':
            spec = stft_transform(waveform).squeeze(0).transpose(0, 1).contiguous()
        else:
            spec = stft_transform(waveform).squeeze(0).transpose(0, 1).contiguous()
            path = os.path.join(target_text,speaker_id +'_nohash_' + str(utterance_number) +'.wav')
            indexes += [torch.tensor(path_to_index[path])]
        spectrograms.append(spec)
        targets += [label_to_index(target_text,labels)]

    targets = torch.stack(targets)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3).contiguous()
    if data_type == 'train':
        return torch.abs(spectrograms), targets
    else:
        indexes = torch.stack(indexes)
        return torch.abs(spectrograms), targets,indexes

def make_relativepath_index(file_paths):
    path_to_index = {}
    index_to_path = {}
    index = 0
    with open(file_paths) as f:
        lines = f.readlines()
        for path in lines:
            path = path.strip()
            path_to_index[path] = index
            index_to_path[index] = path
            index+=1
    return path_to_index,index_to_path

def musan_loader(file_paths,musan_path,speech_command_path,batch_size,path_to_index,audio_transforms):
    spectrograms = []
    targets = []  
    indexes = []
    with open(file_paths) as f:
        lines = f.readlines()
        for mixture_path in lines:
            mixture,sample_rate = torchaudio.load(os.path.join(musan_path,mixture_path).strip())
            target = mixture_path.split('/')[0]
            targets += [label_to_index(target,labels)]
            spec = audio_transforms(mixture).squeeze(0).transpose(0, 1).contiguous()
            spectrograms.append(spec)
            indexes += [torch.tensor(path_to_index[mixture_path])]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3).contiguous()
    targets = torch.stack(targets)
    indexes = torch.stack(indexes)
    musan_dataset = TensorDataset(spectrograms,targets,indexes) # create your datset
    musan_dataloader = DataLoader(musan_dataset, batch_size=batch_size) 
    return musan_dataloader

def load_config(config_path,config_name):
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(config_path)
    config = config._sections[config_name]
    return config

class step_counter(object):
    def __init__(self):
        self.count = 0

    def increase_one(self):
        self.count += 1

    def get(self):
        return self.count 
    
if __name__ == "__main__":
    data_path = "./data"
    train_set = SubsetSC(data_path,"training")
    dev_set = SubsetSC(data_path,"validation")
    test_set = SubsetSC(data_path,"testing")
    print(len(train_set), len(dev_set), len(test_set))