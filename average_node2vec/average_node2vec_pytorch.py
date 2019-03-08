import torch
import re
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

from average_node2vec_utils import Utils
from average_skipgram_pytorch import SkipGram
from average_dataloader import Node2VecDataset
from torch.utils.data import DataLoader

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


class Node2Vec:
    def __init__(self, walks, output_file, walk_length, embedding_dim=128, epochs=10, batch_size=32, window_size=10,
                 neg_sample_num=5):
        self.utils = Utils(walks, window_size, walk_length)
        self.vocabulary_size = self.utils.vocabulary_size
        self.node2phr = self.utils.phrase_dic
        self.word2idx = self.utils.word2idx
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        self.odir_checkpoint = '/home/sotkot/checkpoints/'
        self.odir_embeddings = '/home/sotkot/embeddings/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = SkipGram(self.vocabulary_size, self.embedding_dim, self.neg_sample_num, self.batch_size,
                         self.window_size)
        print_params(model)
        params = model.parameters()
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()

        optimizer = optim.SparseAdam(params, lr=0.001)
        dataset = Node2VecDataset(self.utils, self.neg_sample_num)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            batch_num = 0
            batch_costs = []

            for sample in tqdm(dataloader):
                pos_u = sample['center']
                pos_v = sample['context']
                size = len(pos_u)
                neg_v = np.random.choice(self.utils.sample_table, size=(size * self.neg_sample_num)).tolist()

                pos_u = [torch.LongTensor(phr2idx(self.utils.phrase_dic[int(item)], self.utils.word2idx)) for item in pos_u]
                pos_v = [torch.LongTensor(phr2idx(self.utils.phrase_dic[int(item)], self.utils.word2idx)) for item in pos_v]
                neg_v = [torch.LongTensor(phr2idx(self.utils.phrase_dic[int(item)], self.utils.word2idx)) for item in neg_v]

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())

                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {} '.format(
                        sum(batch_costs) / float(len(batch_costs)),
                        batch_num))
                    batch_costs = []
                batch_num += 1

                if batch_num % 1000000 == 0:
                    state = {'batches': batch_num, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    save_checkpoint(state,
                                    filename=self.odir_checkpoint + 'isa_checkpoint_batch_{}.pth.tar'.format(
                                        batch_num))

            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + 'isa_checkpoint_epoch_{}.pth.tar'.format(
                                epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file, idx2word=self.utils.idx2word,
                                        use_cuda=True)
