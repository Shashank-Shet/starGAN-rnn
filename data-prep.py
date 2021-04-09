import torchaudio
import torch
import torch.nn.functional as nn
import numpy as np

DATASET_PATH = "./dataset/"

INSTRUMENTS = [
    "Bansuri",
    "Shehnai",
    "Santoor",
    "Sarod",
    "Sitar"
]

