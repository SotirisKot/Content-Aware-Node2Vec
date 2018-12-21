from pprint import pprint
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pdb
from tqdm import tqdm
my_seed = 1997
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)


class node2vec_rnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_size, neg_sample_num, batch_size, window_size, scale=1e-4, max_norm=1):
        super(node2vec_rnn, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.neg_sample_num = neg_sample_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.rnn_size = rnn_size
        print(self.rnn_size)
        self.scale = scale
        self.max_norm = max_norm
        # self.h0 = nn.Parameter(torch.randn(1, 1, self.rnn_size).uniform_(-self.scale, self.scale))
        self.the_rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.rnn_size, num_layers=1, bidirectional=False,
                              bias=True, dropout=0, batch_first=True)
        self.init_weights(self.scale)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data[0] = 0
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def init_weights(self, scale=1e-4):
        for param in self.the_rnn.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

    def fix_input(self, phr_inds=None, pos_inds=None, neg_inds=None):
        if self.training:
            seq_lengths_phr = torch.LongTensor([len(seq) for seq in phr_inds]).cuda()
            seq_phr = self.pad_sequences(phr_inds, seq_lengths_phr)

            seq_lengths_pos = torch.LongTensor([len(seq) for seq in pos_inds]).cuda()
            seq_pos = self.pad_sequences(pos_inds, seq_lengths_pos)

            seq_lengths_neg = torch.LongTensor([len(seq) for seq in neg_inds]).cuda()
            seq_neg = self.pad_sequences(neg_inds, seq_lengths_neg)

            return seq_phr, seq_pos, seq_neg
        else:
            phr = Variable(torch.LongTensor(phr_inds), requires_grad=False)
            return phr

    def pad_sequences(self, vectorized_seqs, seq_lengths):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, -seqlen:] = torch.LongTensor(seq)
        return seq_tensor

    def get_words_embeds(self, phr, pos, neg):
        phr = self.u_embeddings(phr)
        pos = self.v_embeddings(pos)
        neg = self.v_embeddings(neg)
        return phr, pos, neg

    def rnn_representation_one(self, inp):
        if self.training:
            batch_size = inp.shape[0]
            # inp, hn = self.the_rnn(inp, self.h0.repeat(1, batch_size, 1))
            inp, hn = self.the_rnn(inp)
            last_timestep = hn[-1]
        else:
            # inp, hn = self.the_rnn(inp, self.h0)
            inp, hn = self.the_rnn(inp)
            last_timestep = hn[-1]
        #
        return last_timestep

    def get_rnn_representation(self, phr, pos, neg):
        phr = self.rnn_representation_one(phr)
        pos = self.rnn_representation_one(pos)

        # because negative sampling is 1..thats why i can pass it like that.
        # otherwise you have to use a for loop or something.
        neg = self.rnn_representation_one(neg)

        return phr, pos, neg

    def get_loss(self, phr_emb, context_emb, neg_emb):
        score = torch.mul(phr_emb, context_emb)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score)
        # neg_score = torch.bmm(neg_emb, phr_emb.unsqueeze(2)).squeeze()
        neg_score = torch.mul(phr_emb, neg_emb)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score)
        # sum_log_sampled = torch.sum(sum_log_sampled, dim=1)
        loss = log_target + sum_log_sampled
        return -1 * torch.mean(loss)

    def forward(self, phr_inds=None, pos_inds=None, neg_inds=None, average=False, concat=False):
        if self.training:
            phr, pos, neg = self.fix_input(phr_inds, pos_inds, neg_inds)
            phr, pos, neg = self.get_words_embeds(phr, pos, neg)
            phr, pos, neg = self.get_rnn_representation(phr, pos, neg)
            loss = self.get_loss(phr, pos, neg)
            return loss
        else:  # here it is used for inference---> you can encode one sentence
            if average:
                phr = self.fix_input(phr_inds=phr_inds)
                print(phr)
                phr_emb = []
                for word in phr:
                    word_emb_u = self.u_embeddings(word)
                    word_emb_v = self.v_embeddings(word)
                    word_emb = (word_emb_u + word_emb_v) / 2
                    print(word_emb)
                    phr_emb.append(word_emb)
                phr_emb = torch.stack(phr_emb)
                phr_emb = self.rnn_representation_one(phr_emb)
                return phr_emb
            else:
                phr = self.fix_input(phr_inds=phr_inds)
                #
                phr_emb_u = self.u_embeddings(phr)
                phr_emb_u = self.rnn_representation_one(phr_emb_u)
                #
                phr_emb_v = self.v_embeddings(phr)
                phr_emb_v = self.rnn_representation_one(phr_emb_v)
                #
                if concat:
                    phr_emb = torch.cat(phr_emb_u, phr_emb_v)
                else:
                    phr_emb = (phr_emb_u + phr_emb_v) / 2
                #
                return phr_emb

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
