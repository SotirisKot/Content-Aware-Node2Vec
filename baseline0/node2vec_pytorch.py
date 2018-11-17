import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
from node2vec_utils import Utils
from skipgram_pytorch import skipgram


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Node2Vec:
    def __init__(self, walks, output_file,embedding_dim=128, epochs=10, batch_size=16, window_size=10, neg_sample_num=5):
        self.utils = Utils(walks, window_size)
        self.vocabulary_size = len(self.utils.vocab_words)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        self.odir = 'drive/My Drive/pytorch-node2vec-umls-relations/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = skipgram(self.vocabulary_size, self.embedding_dim)
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.025)
        for epoch in range(self.epochs):
            #num_batches = self.utils.get_num_batches(batch_size=self.batch_size)
            batch_num = 0
            while(self.utils.stop):#for batch_num in range(num_batches):
                # if (not self.utils.data_ended):
                #     pos_pairs = self.utils.get_batch(self.batch_size)
                # else:
                #     pos_pairs = self.utils.get_batch_from_deque(self.batch_size)

                # neg_v = self.utils.get_neg_sample_batch(pos_pairs, self.neg_sample_num)
                # pos_u = [pair[0] for pair in pos_pairs]
                # pos_v = [pair[1] for pair in pos_pairs]
                pos_u, pos_v, neg_v = self.utils.generate_batch(self.window_size,self.batch_size,self.neg_sample_num)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, self.batch_size)
                loss.backward()
                optimizer.step()
                if batch_num % 5000 == 0:
                    print('Batch Loss: {}, num_batch: {} '.format(loss.item(), batch_num))
                batch_num += 1
            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state, filename=self.odir + 'isa_undirected_checkpoint_epoch_{}.pth.tar'.format(epoch + 1))
            self.utils.stop = True
            #self.utils.data_ended = False
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.output_file, idx2word=self.utils.vocab_words, use_cuda=True)
