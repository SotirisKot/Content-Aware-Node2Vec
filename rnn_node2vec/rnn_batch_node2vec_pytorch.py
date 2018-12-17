from pprint import pprint

import torch
import re
from torch.autograd import Variable
import torch.optim as optim
import time
from rnn_node2vec_utils import Utils
from rnn_batch_skipgram import node2vec_rnn
from torch.utils.data import DataLoader
from rnn_dataloader import Node2VecDataset
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()).strip()


def tokenize(x):
    return bioclean(x)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_index(w, vocab):
    try:
        return vocab[w]
    except KeyError:
        return vocab['UNKN']


def phr2idx(phr, word_vocab):
    p = [get_index(t, word_vocab) for t in phr.split()]
    return p


def print_params(model):
    print(40 * '=')
    print(model)
    print(40 * '=')
    total_params = 0
    for parameter in model.parameters():
        print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')


class Node2Vec:
    def __init__(self, walks, output_file, walk_length, embedding_dim=128, rnn_size=50, epochs=10, batch_size=32, window_size=10,
                 neg_sample_num=5):
        self.utils = Utils(walks, window_size, walk_length)
        self.vocabulary_size = self.utils.vocabulary_size
        self.node2phr = self.utils.phrase_dic
        self.word2idx = self.utils.word2idx
        self.embedding_dim = embedding_dim
        self.rnn_size = rnn_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        # self.odir_checkpoint = 'drive/My Drive/pytorch-node2vec-umls-relations/checkpoints/'
        # self.odir_embeddings = 'drive/My Drive/pytorch-node2vec-umls-relations/embeddings/'
        self.odir_checkpoint = '/home/paperspace/sotiris/'
        self.odir_embeddings = '/home/paperspace/sotiris/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = node2vec_rnn(self.vocabulary_size, self.embedding_dim, self.rnn_size, self.neg_sample_num,
                             self.batch_size,
                             self.window_size)
        print_params(model)
        params = model.parameters()
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()

        optimizer = optim.Adam(params, lr=0.001)
        dataset = Node2VecDataset(self.utils, self.neg_sample_num)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=True)

        for epoch in range(self.epochs):
            batch_num = 0
            batch_costs = []
            for sample in tqdm(dataloader):
                max_phr_len = 0
                max_pos_len = 0
                max_neg_len = 0
                center = sample['center']
                context = sample['context']
                neg_v = np.random.choice(self.utils.sample_table, size=(len(center) * self.neg_sample_num)).tolist()

                phr = [phr2idx(self.utils.phrase_dic[int(phr)], self.word2idx) for phr in center]
                max_phr_len = max([max_phr_len] + [len(pos_u) for pos_u in phr])

                pos_context = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in context]
                max_pos_len = max([max_pos_len] + [len(pos_ind) for pos_ind in pos_context])

                neg_v = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in neg_v]
                max_neg_len = max([max_neg_len] + [len(neg_ind) for neg_ind in neg_v])

                batch_phr_inds = np.stack(pad_sequences(sequences=[b for b in phr], maxlen=max_phr_len))
                batch_pos_inds = np.stack(pad_sequences(sequences=[b for b in pos_context], maxlen=max_pos_len))
                batch_neg_inds = np.stack(pad_sequences(sequences=[b for b in neg_v], maxlen=max_neg_len))

                optimizer.zero_grad()
                loss = model(batch_phr_inds, batch_pos_inds, batch_neg_inds)
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())

                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {} '.format(
                        sum(batch_costs) / float(len(batch_costs)),
                        batch_num))
                    batch_costs = []
                batch_num += 1
            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + 'part_of_rnn_checkpoint_epoch_{}.pth.tar'.format(
                                epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file,
                                        idx2word=self.utils.idx2word,
                                        use_cuda=True)
