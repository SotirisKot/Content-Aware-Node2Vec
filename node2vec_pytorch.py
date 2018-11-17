import numpy as np
import math
import random
import torch
import re
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import time
from node2vec_utils import Utils
from skipgram_pytorch import SkipGram

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


class Node2Vec:
    def __init__(self, walks, output_file, embedding_dim=128, epochs=10, batch_size=32, window_size=10, neg_sample_num=5):
        self.utils = Utils(walks, window_size)
        self.vocabulary_size = self.utils.vocabulary_size
        self.node2phr = self.utils.phrase_dic
        self.word2idx = self.utils.word2idx
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        self.odir_checkpoint = '/home/paperspace/sotiris/thesis/'
        self.odir_embeddings = '/home/paperspace/sotiris/thesis/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = SkipGram(self.vocabulary_size, self.embedding_dim, self.neg_sample_num, self.batch_size, self.window_size)
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.025)
        total_batches = self.utils.get_num_batches(self.batch_size)  # not very accurate but just for an insight
        for epoch in range(self.epochs):
            batch_num = 0
            while self.utils.stop:
                start = time.time()
                pos_u, pos_v, neg_v = self.utils.generate_batch(self.window_size, self.batch_size, self.neg_sample_num)
                pos_u = [Variable(torch.LongTensor(phr2idx(self.node2phr[item], self.utils.word2idx)), requires_grad=False).cuda() for item in pos_u]
                pos_v = [Variable(torch.LongTensor(phr2idx(self.node2phr[item], self.utils.word2idx)), requires_grad=False).cuda() for item in pos_v]
                neg_v = [Variable(torch.LongTensor(phr2idx(self.node2phr[item], self.utils.word2idx)), requires_grad=False).cuda() for item in neg_v]

                # if torch.cuda.is_available():
                #     pos_u = [pos.cuda() for pos in pos_u]
                #     pos_v = [pos.cuda() for pos in pos_v]
                #     neg_v = [neg.cuda() for neg in neg_v]
                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                if batch_num % 10 == 0:
                    print('Epoch: {}, Batch Loss: {}, num_batch: {}/{} '.format(epoch,loss.item(), batch_num, total_batches))
                    print('It took', time.time() - start, 'seconds, for 10 batches.')
                batch_num += 1
            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state, filename=self.odir_checkpoint + 'isa_average_words_checkpoint_epoch_{}.pth.tar'.format(epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file, idx2word=self.utils.idx2word, use_cuda=True)