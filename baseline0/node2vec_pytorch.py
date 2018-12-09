import torch
from torch.autograd import Variable
import torch.optim as optim
import time
from node2vec_utils import Utils
from skipgram_pytorch import SkipGram
from tqdm import tqdm
import numpy as np

from dataloader import Node2VecDataset
from torch.utils.data import Dataset, DataLoader


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
        # self.odir_checkpoint = 'drive/My Drive/node2vec_average_embeddings/checkpoints/'
        # self.odir_embeddings = 'drive/My Drive/node2vec_average_embeddings/embeddings/'
        self.odir_checkpoint = '/home/paperspace/sotiris/'
        self.odir_embeddings = '/home/paperspace/sotiris/'
        self.output_file = output_file
        self.wv = {}

    def train(self):
        model = SkipGram(self.vocabulary_size, self.embedding_dim)
        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.025)
        total_batches = self.utils.get_num_batches(self.batch_size)
        dataset = Node2VecDataset('dataset.txt')
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        for epoch in range(self.epochs):
            batch_costs = []
            start = time.time()
            batch_num = 0
            # while self.utils.stop:
            #     pos_u, pos_v, neg_v, batch_len = self.utils.generate_batch(self.window_size, self.batch_size, self.neg_sample_num)
            # for pos_u, pos_v, neg_v in self.utils.node2vec_yielder(self.window_size, self.neg_sample_num):
            for sample in tqdm(dataloader):
                pos_u = sample['center']
                pos_v = sample['context']
                neg_v = np.random.choice(self.utils.sample_table, size=(pos_u.shape[0], self.neg_sample_num))

                pos_u = Variable(torch.LongTensor(pos_u), requires_grad=False)
                pos_v = Variable(torch.LongTensor(pos_v), requires_grad=False)
                neg_v = Variable(torch.LongTensor(neg_v), requires_grad=False)

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, pos_u.shape[0])
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())
                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, batches: {}/{} '.format(
                        sum(batch_costs) / float(len(batch_costs)),
                        batch_num, total_batches))
                    print('It took', time.time() - start, 'seconds.')
                    start = time.time()
                    batch_costs = []
                batch_num += 1
            print()
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + 'part_of_baseline_checkpoint_epoch_{}.pth.tar'.format(
                                epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file, idx2word=self.utils.vocab_words, use_cuda=True)
