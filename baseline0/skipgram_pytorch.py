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

    def forward(self, pos_u, pos_v, neg_v,neg_samples):
        embed_u = self.u_embeddings(pos_u)
        embed_v = self.v_embeddings(pos_v)
        embed_v = embed_v.view(len(pos_u), -1, self.embedding_dim)
        neg_embed_v = self.v_embeddings(neg_v)
        neg_embed_v = neg_embed_v.view(len(pos_u), neg_samples, self.embedding_dim)
        score = torch.bmm(embed_v, embed_u.unsqueeze(2)).squeeze()
        log_target = F.logsigmoid(score)
        log_target = torch.sum(log_target, dim=1)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        sum_log_sampled = F.logsigmoid(-1 * neg_score)
        sum_log_sampled = torch.sum(sum_log_sampled, dim=1)
        loss = log_target + sum_log_sampled
        return -1 * loss.sum() / float(len(pos_u))

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
