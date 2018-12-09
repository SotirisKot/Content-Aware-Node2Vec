import numpy as np
import torch
from torch.utils.data import Dataset
from node2vec_utils import Utils
import pandas as pd

class Node2VecDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, filepath):
        print('Loading data')
        self.data = pd.read_csv('dataset.txt', sep=" ", header=None)
        print('Done loading data')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        example = self.data.iloc[index]
        sample = {'center': example[0], 'context': example[1]}
        return sample
