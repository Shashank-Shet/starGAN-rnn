import torchaudio
import torch
import torch.nn.functional as nn
import numpy as np
from torch.utils.data import IterableDataset

DATASET_PATH = "./dataset-preprocessed/"

INSTRUMENTS = [
    "Bansuri",
    "Shehnai",
    "Santoor",
    "Sarod",
    "Sitar"
]

INSTRUMENT_LABELS = {
    "Bansuri" : 0,
    "Shehnai" : 1,
    "Santoor" : 2,
    "Sarod"   : 3,
    "Sitar"   : 4
}

NUM_FILES_PER_INSTRUMENT = [
    728,
    302,
    479,
    456,
    1291
]


class DataSource:

    def __init__(self, stop_iter=302):
        self.stop_iter = stop_iter
        self.counter = 0

    def __next__(self):
        if self.counter >= self.stop_iter:
            raise StopIteration()
        file_indices = [ np.random.randint(1, NUM_FILES_PER_INSTRUMENT[i]) for i in range(len(INSTRUMENTS)) ]
        x_tensor_list = []
        y_tensor_list = []
        for instrument, index in zip(INSTRUMENTS, file_indices):
            temp = torch.tensor(INSTRUMENT_LABELS[instrument])
            data = torch.load(f"./dataset-preprocessed/{instrument}/{instrument}_{index}.pt")
            labels = temp.repeat(data.shape[0])
            x_tensor_list.append(data)
            y_tensor_list.append(labels)
        self.counter += 1
        X = torch.vstack(x_tensor_list)
        y = torch.cat(y_tensor_list)
        return X, y


class MyIterableDataset(IterableDataset):

    def __init__(self, train=True):
        if train is True:
            self.source = DataSource()
        else:
            self.source = DataSource(100)

    def __iter__(self):
        return self.source
