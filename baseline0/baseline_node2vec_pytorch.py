import torch
from torch.autograd import Variable
import torch.optim as optim
import time
from baseline_node2vec_utils import Utils
from baseline_skipgram_pytorch import SkipGram
from torch.utils.data import DataLoader
from baseline_dataloader import Node2VecDataset
from tqdm import tqdm
import numpy as np
np.random.seed(1997)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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
    def __init__(self, walks, output_file, walk_length, embedding_dim=128, epochs=10, batch_size=16, window_size=10, neg_sample_num=5):
        self.utils = Utils(walks, window_size, walk_length)
        self.vocabulary_size = len(self.utils.vocab_words)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        # self.odir_checkpoint = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'
        # self.odir_embeddings = 'drive/My Drive/node2vec_average_embeddings/embeddings/'
        self.odir_checkpoint = '/home/paperspace/sotiris/thesis/baseline0/'
        self.odir_embeddings = '/home/paperspace/sotiris/thesis/baseline0/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = SkipGram(self.vocabulary_size, self.embedding_dim)
        print_params(model)
        params = model.parameters()
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()

        optimizer = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        dataset = Node2VecDataset(self.utils, self.batch_size, self.neg_sample_num)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        for epoch in range(self.epochs):
            batch_num = 0
            batch_costs = []
            for index, sample in enumerate(tqdm(dataloader)):
                pos_u = sample['center']
                pos_v = sample['context']
                neg_v = np.random.choice(self.utils.sample_table, size=(self.batch_size, self.neg_sample_num))

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())
                del pos_u, pos_v, neg_v

                if index % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {}'.format(
                            sum(batch_costs) / float(len(batch_costs)),
                            batch_num))
                    batch_costs = []
                batch_num += 1
            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + 'part_of_baseline_3_checkpoint_epoch_{}.pth.tar'.format(
                                epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file, idx2word=self.utils.vocab_words, use_cuda=True)
