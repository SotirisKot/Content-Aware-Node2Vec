import collections
import numpy as np
from collections import deque
import math
import os
import random
from tqdm import tqdm

np.random.seed(12345)
data_index = 0


class Utils(object):
    def __init__(self, walks, window_size):
        self.stop = True
        self.window_size = window_size
        self.walks = walks
        data, self.frequencies, self.vocab_words = self.build_dataset(self.walks)
        self.train_data = data
        # the sample_table it is used for negative sampling as they do in the original word2vec
        self.sample_table = self.create_sample_table()

    def build_word_vocab(self, walks):
        vocabulary = []  # in node2vec the words are nodeids and each walk represents a sentence
        for walk in tqdm(walks):
            for token in walk:
                vocabulary.append(token)
        vocab_size = len(vocabulary)
        return vocab_size, vocabulary

    def build_dataset(self, walks):
        print('Building dataset..')
        vocab_size, vocabulary = self.build_word_vocab(walks)
        count = []
        count.extend(collections.Counter(vocabulary).most_common(vocab_size - 1))
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = []
        for word in vocabulary:
            index = dictionary[word]
            data.append(index)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dictionary

    def create_sample_table(self):
        print('Creating sample table..')
        count = [element[1] for element in self.frequencies]
        pow_frequency = np.array(count) ** 0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def get_neg_sample_batch(self, pos_pairs, num_neg_samples):
        neg_v = np.random.choice(self.sample_table, size=(len(pos_pairs), num_neg_samples)).tolist()
        return neg_v

    def get_num_batches(self, batch_size):
        num_batches = len(self.word_pairs) / batch_size
        print(num_batches)
        num_batches = int(math.ceil(num_batches))
        return num_batches

    def generate_batch(self, window_size, batch_size, neg_samples):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        if data_index + span > len(data):
            data_index = 0
            self.stop = False
        buffer = data[data_index:data_index + span]
        pos_u = []
        pos_v = []
        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size] + buffer[window_size + 1:]
            labels[i] = buffer[window_size]
            if data_index + span > len(data):
                data_index = 0
                self.stop = False
            else:
                buffer = data[data_index:data_index + span]
            pos_u.append(labels[i])
            for j in range(span - 1):
                pos_v.append(context[i, j])
        neg_v = np.random.choice(self.sample_table, size=(batch_size*neg_samples))
        return np.array(pos_u), np.array(pos_v), neg_v

