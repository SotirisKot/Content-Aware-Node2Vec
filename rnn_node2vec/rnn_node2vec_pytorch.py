import torch
import re
from torch.autograd import Variable
import torch.optim as optim
import time
from rnn_node2vec_utils import Utils
from rnn_skipgram import node2vec_rnn
from tqdm import tqdm


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
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')


def back_prop(batch_costs, optimizer):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    return batch_aver_cost


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
        for epoch in range(self.epochs):
            batch_num = 0
            batch_costs = []
            batch_aver_costs = []
            for phr, pos, neg in self.utils.node2vec_yielder(self.window_size, self.neg_sample_num):

                phr = phr2idx(self.utils.phrase_dic[int(phr)], self.word2idx)
                pos_context = phr2idx(self.utils.phrase_dic[int(pos)], self.word2idx)
                neg_v = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in neg]

                optimizer.zero_grad()
                instance_cost = model(phr, pos_context, neg_v)
                batch_costs.append(instance_cost)

                if len(batch_costs) == self.batch_size:
                    batch_aver_cost = back_prop(batch_costs, optimizer)
                    batch_costs = []
                    batch_num += 1
                    batch_aver_costs.append(batch_aver_cost)

                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {} '.format(
                        sum(batch_aver_costs) / float(len(batch_aver_costs)),
                        batch_num))

            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + 'part_of_rnn_instances_checkpoint_epoch_{}.pth.tar'.format(
                                epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file,
                                        idx2word=self.utils.idx2word,
                                        use_cuda=True)
