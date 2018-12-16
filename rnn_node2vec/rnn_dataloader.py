import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import pandas as pd


class Node2VecDataset(Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, utils, neg_samples):
        self.utils = utils
        # self.neg_samples = neg_samples
        # self.span = 2 * self.utils.window_size
        # self.data_gen = self._data_generator()
        print('Loading data')
        # self.data = self.utils.walks
        self.data = pd.read_csv('dataset.txt', sep=' ', header=None)
        print('Done loading data')

    def __len__(self):
        """Denotes the total number of samples"""
        # length = len(self.data) * ((self.utils.walk_length - self.span) * self.span)
        return len(self.data)

    def __getitem__(self, index):
        example = self.data.iloc[index]
        # phr, context = next(self.data_gen)
        # sample = {'center': phr, 'context': context}
        sample = {'center': example[0], 'context': example[1]}
        return sample

    def _data_generator(self):
        for walk in self.data:
            for idx, phr in enumerate(walk):
                # for each window position
                for w in range(-self.utils.window_size, self.utils.window_size + 1):
                    context_word_pos = idx + w
                    # make sure not jump out sentence
                    if context_word_pos < 0:
                        break
                    elif idx + self.utils.window_size >= len(walk):
                        break
                    elif idx == context_word_pos:
                        continue
                    context_word_idx = walk[context_word_pos]
                    yield phr, context_word_idx
