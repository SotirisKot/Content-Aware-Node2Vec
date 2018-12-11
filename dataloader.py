import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from baseline0.baseline_node2vec_utils import Utils
import pandas as pd


class Node2VecDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, filepath, utils, batch_size, neg_samples):
        self.utils = utils
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        print('Loading data')
        self.data = pd.read_csv(filepath, sep=" ", header=None)
        print('Done loading data')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        example = self.data.iloc[index]
        neg_v = np.random.choice(self.utils.sample_table, size=self.neg_samples)
        sample = {'center': example[0], 'context': example[1], 'neg': Variable(torch.LongTensor(neg_v))}
        print(sample)
        return sample

