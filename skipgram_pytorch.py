import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from tqdm import tqdm


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, neg_sample_num, batch_size, window_size):
        super(SkipGram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.neg_sample_num = neg_sample_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def get_average_embedings(self, pos_u, pos_v, neg_v):
        pos_u_average = torch.Tensor(self.batch_size * 2 * self.window_size, self.embedding_dim)
        for idx, phrase_idxs in enumerate(pos_u):
            embed_u = self.u_embeddings(phrase_idxs)
            if len(phrase_idxs) > 1:
                embed = torch.sum(embed_u, dim=0)
                pos_u_average[idx] = embed.div(len(phrase_idxs))
            else:
                pos_u_average[idx] = embed_u

        pos_v_average = torch.Tensor(self.batch_size * 2 * self.window_size, self.embedding_dim)
        for idx, phrase_idxs in enumerate(pos_v):
            embed_v = self.v_embeddings(phrase_idxs)
            if len(phrase_idxs) > 1:
                embed = torch.sum(embed_v, dim=0)
                pos_v_average[idx] = embed.div(len(phrase_idxs))
            else:
                pos_v_average[idx] = embed_v

        neg_v_average = torch.Tensor(pos_u_average.shape[0] * self.neg_sample_num, self.embedding_dim)
        for idx, phrase_idxs in enumerate(neg_v):
            neg_embed_v = self.v_embeddings(phrase_idxs)
            if len(phrase_idxs) > 1:
                embed = torch.sum(neg_embed_v, dim=0)
                neg_v_average[idx] = embed.div(len(phrase_idxs))
            else:
                neg_v_average[idx] = neg_embed_v

        neg_v_average = neg_v_average.view(pos_u_average.shape[0], self.neg_sample_num, self.embedding_dim)
        return pos_u_average, pos_v_average, neg_v_average

    def forward(self, pos_u, pos_v, neg_v):
        embed_u, embed_v, neg_embed_v = self.get_average_embedings(pos_u, pos_v, neg_v)
        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.exp(neg_score)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = - torch.log(1 + neg_score)
        loss = - (score + neg_score)
        return loss.sum() / self.batch_size

    def save_embeddings(self, file_name, idx2word, use_cuda=False):
        wv = {}
        if use_cuda:
            embedding_u = self.u_embeddings.weight.cpu().data.numpy()
            embedding_v = self.v_embeddings.weight.cpu().data.numpy()
        else:
            embedding_u = self.u_embeddings.weight.data.numpy()
            embedding_v = self.v_embeddings.weight.data.numpy()

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(idx2word), self.embedding_dim))
        for wid, w in idx2word.items():
            e_u = embedding_u[wid]
            e_v = embedding_v[wid]
            e = (e_u + e_v) / 2
            wv[w] = e
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
        return wv
