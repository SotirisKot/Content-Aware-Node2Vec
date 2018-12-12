import collections
from pprint import pprint

import numpy as np
from collections import deque
import math
import os
import random
from torch.utils.data import Dataset
from tqdm import tqdm

np.random.seed(1997)
data_index = 0
walk_index = 0


class Utils(object):
    def __init__(self, walks, window_size, walk_length):
        self.stop = True
        self.walk_length = walk_length
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
        count.extend(collections.Counter(vocabulary).most_common(vocab_size))
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = []
        for walk in tqdm(self.walks):
            for idx, nodeid in enumerate(walk):
                index = dictionary[nodeid]
                walk[idx] = index
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

    def get_walk(self):
        global walk_index
        try:
            walk = self.walks[walk_index]
            walk_index += 1
            return walk
        except:
            print('No more walks..')
            self.stop = False

    def generate_batch(self, window_size, batch_size, neg_samples):
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        if data_index + span > len(self.current_walk):
            data_index = 0
        buffer = self.current_walk[data_index:data_index + span]
        pos_u = []
        pos_v = []
        batch_len = 0
        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size] + buffer[window_size + 1:]
            labels[i] = buffer[window_size]
            if data_index + span > len(self.current_walk):
                data_index = 0
                self.current_walk = self.get_walk()
                if self.stop:
                    buffer[:] = self.current_walk[:span]
            else:
                buffer = self.current_walk[data_index:data_index + span]
            if self.stop:
                batch_len += 1
                # pos_u.append(labels[i])
                for j in range(span - 1):
                    pos_u.append(labels[i])
                    pos_v.append(context[i, j])
            else:
                batch_len += 1
                # pos_u.append(labels[i])
                for j in range(span - 1):
                    pos_u.append(labels[i])
                    pos_v.append(context[i, j])
                break
        neg_v = np.random.choice(self.sample_table, size=(batch_len * 2 * window_size, neg_samples))
        return np.array(pos_u), np.array(pos_v), neg_v, batch_len

    def node2vec_yielder(self, window_size, neg_samples):
        for walk in tqdm(self.walks):
            for idx, phr in enumerate(walk):
                # for each window position
                pos_context = []
                for w in range(-window_size, window_size + 1):
                    context_word_pos = idx + w
                    # make sure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(walk) or idx == context_word_pos:
                        continue
                    context_word_idx = walk[context_word_pos]
                    pos_context.append(context_word_idx)
                neg_v = np.random.choice(self.sample_table, size=(neg_samples)).tolist()
                yield phr, pos_context, neg_v

    def node2vec_batch_yielder(self, window_size, batch_size):
        batch_pairs = []
        for walk in tqdm(self.walks):
            for idx, phr in enumerate(walk):
                # for each window position
                for w in range(-window_size, window_size + 1):
                    context_word_pos = idx + w
                    # make sure not jump out sentence
                    if context_word_pos < 0:
                        break
                    elif idx + window_size >= len(walk):
                        break
                    elif idx == context_word_pos:
                        continue
                    batch_pairs.append((phr, walk[context_word_pos]))
                    if len(batch_pairs) == batch_size:
                        yielded_batch = batch_pairs
                        batch_pairs = []
                        yield yielded_batch

    def create_save_dataset(self, window_size):
        with open('dataset.txt', 'w') as dataset:
            for walk in tqdm(self.walks):
                for idx, phr in enumerate(walk):
                    # for each window position
                    pos_context = []
                    for w in range(-window_size, window_size + 1):
                        context_word_pos = idx + w
                        # make sure not jump out sentence
                        if context_word_pos < 0:
                            break
                        elif idx + window_size >= len(walk):
                            break
                        elif idx == context_word_pos:
                            continue
                        context_word_idx = walk[context_word_pos]
                        pos_context.append(context_word_idx)
                    if len(pos_context) != 0:
                        for pos in pos_context:
                            dataset.write(str(phr) + ' ' + str(pos) + '\n')

    def get_num_batches(self, batch_size):
        num_batches = len(self.walks) * 80 / batch_size
        num_batches = int(math.ceil(num_batches))
        return num_batches


if __name__ == "__main__":
    # walks = [['1', '23345', '3356', '4446', '5354', '6123', '74657', '8445', '97890', '1022', '1133'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1'],
    #          ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1','9999']]
    walks = [['1', '23345', '3356', '4446', '5354', '6123', '74657', '8445', '97890', '1022', '1133'],
             ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1'],
             ['6914', '1022', '97890', '8445', '74657', '6123', '5354', '4446', '3356', '23345', '1']]
    utils = Utils(walks, 2)
    for batch_pairs in utils.node2vec_yielder(window_size=5, batch_size=64):
        print(batch_pairs)

    # print(len(pos_u))
    # print(batch_size)
    # pprint(pos_v)
    # print(neg_v)
    print(utils.vocab_words)
    # for pos_u, pos_v in utils.node2vec_yielder(window_size=2):
    #     print(pos_u)
    #     print(pos_v)
    # while utils.stop:
    #     pos_u, pos_v, neg_v, batch_size = utils.generate_batch(window_size=2, batch_size=6, neg_samples=5)
    #     print(pos_u)
    #     print(pos_v)
    # print(neg_v)
    # neg_v = Variable(torch.LongTensor(neg_v))
    # print(neg_v)
    # pos_u = [phr2idx(utils.phrase_dic[item], utils.word2idx) for item in pos_u]
    # print(pos_u)
    # pos_v = [phr2idx(utils.phrase_dic[item], utils.word2idx) for item_list in pos_v for item in item_list]
    # print(pos_v)
    # neg_v = [phr2idx(utils.phrase_dic[item], utils.word2idx) for item in neg_v]
    # print(neg_v)
    # print(pos_u)
    # print(pos_v)
    # print(neg_v)
    # exit()
    # # #print(neg_v)
    # pos = [Variable(torch.LongTensor(pos_ind), requires_grad=False) for pos_ind in pos_u]
    # pos_v = [Variable(torch.LongTensor(pos_ind), requires_grad=False) for pos_ind in pos_v]
    # neg_v = [Variable(torch.LongTensor(item_list), requires_grad=False) for item_list in neg_v]
    # model = SkipGram(utils.vocabulary_size, 128, neg_sample_num=2)
    # # # print(pos)
    # # # print(pos_v)
    # # # print(neg_v)
    # loss = model(pos, pos_v, neg_v, 4)
    # neg_v = [phr2idx(utils.phrase_dic[item], utils.word2idx) for item in neg_v]
    # print(neg_v)
    # print(neg_v.shape)

    # words_neg_sample = []
    # for item_list in neg_v:
    #     for item in item_list:
    #         words_neg_sample.append(phr2idx(utils.phrase_dic[item], utils.word2idx))
    # print('hiii: ',words_neg_sample)
    # for idx,item_list in enumerate(neg_v):
    #     print(idx,item_list)
    #     for idx1, i in enumerate(item_list):
    #         neg_v[idx][idx1] = phr2idx(utils.phrase_dic[i], utils.word2idx)
    # print(neg_v)
    # print(neg_v.shape)
    # neg_v = np.asarray(neg_v)

    # print(neg_v)
    # print(neg_v.shape)
    # print(pos_u)
    # print(pos_v)
    # print('neg_v:' ,neg_v[0][0])
    # print(neg_v.shape)
    # print(utils.word2idx)
    # exit()
    # loss = model(pos_u, pos_v, neg_v, 4)