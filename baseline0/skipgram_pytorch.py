import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
my_seed = 1997
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v, batch_size):
        print(pos_v)
        embed_u = self.u_embeddings(pos_u)
        embed_v = self.v_embeddings(pos_v)
        neg_embed_v = self.v_embeddings(neg_v)
        print(embed_u.size())
        print(embed_v.size())
        embed_score = embed_u.expand_as(embed_v)
        print(embed_score.size())
        exit()
        embed_neg = embed_u.expand_as(neg_embed_v)
        score = torch.mul(embed_score, embed_v)
        score = torch.sum(score, dim=1)
        neg_score = torch.mul(embed_neg, neg_embed_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = torch.exp(neg_score)
        neg_score = torch.sum(neg_score)
        neg_score = - torch.log(1 + neg_score)
        loss = score.sum() + neg_score
        return -1 * loss
        # log_target = F.logsigmoid(score)
        # neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        # sum_log_sampled = F.logsigmoid(-1 * neg_score)
        # sum_log_sampled = torch.sum(sum_log_sampled, dim=1)
        # loss = log_target.sum() + sum_log_sampled
        # return -1 * loss.sum() / batch_size

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
