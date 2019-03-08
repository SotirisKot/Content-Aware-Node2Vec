import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import config


class Node2VecDataset(Dataset):
    def __init__(self, utils, neg_samples):
        self.utils = utils
        self.neg_samples = neg_samples
        self.span = 2 * self.utils.window_size
        self.data_gen = self._data_generator()
        print('Loading data')
        self.data = self.utils.walks
        print('Done loading data')

    def __len__(self):
        """Denotes the total number of samples"""
        length = len(self.data) * ((self.utils.walk_length - self.span) * self.span)
        return length

    def __getitem__(self, index):
        phr, context = next(self.data_gen)
        # TODO - MAYBE return them as integers so the sample will be a dict of tensors..it might help with average
        sample = {'center': phr, 'context': context}
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
                    elif phr == walk[context_word_pos]:
                        continue
                    context_word_idx = walk[context_word_pos]
                    if config.model == 'baseline':
                        yield int(phr), int(context_word_idx)
                    else:
                        yield phr, context_word_idx

    def reset_generator(self):
        del self.data_gen
        self.data_gen = self._data_generator()
