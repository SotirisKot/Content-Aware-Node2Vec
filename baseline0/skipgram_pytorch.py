import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
        self.embedding_dim = embedding_dim
        self.init_emb()
  def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
  def forward(self, pos_u, pos_v, neg_v, batch_size):
        embed_u = self.u_embeddings(pos_u)
        embed_v = self.v_embeddings(pos_v)
        score  = torch.mul(embed_u, embed_v)
        print('score:', score.size())
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score)  
        print('log_target: ', log_target.size())    
        neg_embed_v = self.v_embeddings(neg_v)
        print(neg_embed_v.size())
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        print('neg_score size: ',neg_score.size())
        sum_log_sampled = F.logsigmoid(-1*neg_score)
        sum_log_sampled = torch.sum(sum_log_sampled, dim=1)
        print('sum_log_sampled: ', sum_log_sampled.size())
        loss = log_target + sum_log_sampled
        return -1*loss.sum()/batch_size
  def save_embeddings(self, file_name, idx2word, use_cuda=False):
        wv = {}
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(idx2word), self.embedding_dim))
        for wid, w in idx2word.items():
            e = embedding[wid]
            wv[w] = e
            e = ' '.join(map(lambda x: str(x), e))                
            fout.write('%s %s\n' % (w, e))
        return wv

