import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import config
import os
import pickle


class Node2VecDataset(Dataset):
    def __init__(self, utils, neg_samples):
        self.utils = utils
        self.neg_samples = neg_samples
        self.span = 2 * self.utils.window_size
        self.data_gen = self._data_generator()

        print('Loading data')
        self.data = self.utils.walks
        print('Done loading data')

        if config.write_data:
            print('Writing data in disk if we need to resume training...')
            with open("{}".format(os.path.join(config.checkpoint_dir, '{}_walks.p'.format(config.dataset_name))), 'wb') as dump_file:
                pickle.dump(self.data, dump_file)
            print('Done writing data...')

    def __len__(self):
        """Denotes the total number of samples"""
        length = len(self.data) * ((self.utils.walk_length - self.span) * self.span)
        return length

    def __getitem__(self, index):
        phr, context = next(self.data_gen)
        sample = {'center': phr, 'context': context}
        return sample

    def _data_generator(self):
        for walk in self.data:
            for idx, phr in enumerate(walk):
                # for each window position
                for w in range(-self.utils.window_size, self.utils.window_size + 1):
                    context_word_pos = idx + w
                    if context_word_pos < 0:
                        break
                    elif idx + self.utils.window_size >= len(walk):
                        break
                    elif idx == context_word_pos:
                        continue
                    elif phr == walk[context_word_pos]:
                        continue
                    context_word_idx = walk[context_word_pos]
                    yield phr, context_word_idx

    def reset_generator(self):
        del self.data_gen
        self.data_gen = self._data_generator()
