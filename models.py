from pprint import pprint
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
import os
import pickle
import logging
import time


''' SEEDS '''
my_seed = 1997
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)

'''OUTPUT DIR'''
output_dir = config.checkpoint_dir

'''
Initialization of the logger
Uncomment and use the logger
'''

# handler = None
#
#
# def init_logger(handler):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     od = output_dir.split('/')[-1]
#     logger = logging.getLogger(od)
#     if handler is not None:
#         logger.removeHandler(handler)
#     handler = logging.FileHandler(os.path.join(output_dir, 'model.log'))
#     formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
#     return logger, handler


'''
A function to unsort an output
'''


def unsort(output, perm_idx, dim=0):
    _, unperm_idx = perm_idx.sort(0)
    output = output.index_select(dim, unperm_idx)
    return output


'''
Average the words in a phrase and use it as an encoding
'''


class AverageNode2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, neg_sample_num, batch_size, window_size):
        super(AverageNode2Vec, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True, padding_idx=0)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.neg_sample_num = neg_sample_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data[0] = 0
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def fix_input(self, phr_inds=None, pos_inds=None, neg_inds=None):

        seq_lengths_phr = torch.LongTensor([len(seq) for seq in phr_inds])
        seq_lengths_pos = torch.LongTensor([len(seq) for seq in pos_inds])
        seq_lengths_neg = torch.LongTensor([len(seq) for seq in neg_inds])

        seq_phr, phr_lengths = self.pad_sequences(phr_inds, seq_lengths_phr)
        seq_pos, pos_lengths = self.pad_sequences(pos_inds, seq_lengths_pos)
        seq_neg, neg_lengths = self.pad_sequences(neg_inds, seq_lengths_neg)

        if torch.cuda.is_available():
            return seq_phr.cuda(), \
                   phr_lengths.cuda(), \
                   seq_pos.cuda(), \
                   pos_lengths.cuda(), \
                   seq_neg.cuda(), \
                   neg_lengths.cuda()
        else:
            return seq_phr, \
                   phr_lengths, \
                   seq_pos, \
                   pos_lengths, \
                   seq_neg, \
                   neg_lengths

    def pad_sequences(self, vectorized_seqs, seq_lengths):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

        return seq_tensor, seq_lengths

    def get_words_embeds(self, phr, pos, neg):
        phr = self.u_embeddings(phr)
        pos = self.v_embeddings(pos)
        neg = self.v_embeddings(neg)
        return phr, pos, neg

    def get_average_embedings(self, pos_u, pos_u_lens, pos_v, pos_v_lens, neg_v, neg_v_lens):

        pos_u_lens = pos_u_lens.float().unsqueeze(1)
        pos_v_lens = pos_v_lens.float().unsqueeze(1)
        neg_v_lens = neg_v_lens.float().unsqueeze(1)

        # for pos_u
        emb_u = torch.sum(pos_u, 1).squeeze(0)
        emb_u = emb_u / pos_u_lens.expand_as(emb_u)

        # for pos_v
        emb_v = torch.sum(pos_v, 1).squeeze(0)
        emb_v = emb_v / pos_v_lens.expand_as(emb_v)

        # for neg_v
        neg_v = torch.sum(neg_v, 1).squeeze(0)
        neg_v = neg_v / neg_v_lens.expand_as(neg_v)
        neg_v = neg_v.view(emb_u.shape[0], -1, self.embedding_dim)

        return emb_u, emb_v, neg_v

    def forward(self, pos_u, pos_v, neg_v):

        pos_u, pos_u_lens, pos_v, pos_v_lens, neg_v, neg_v_lens = self.fix_input(pos_u, pos_v, neg_v)
        pos_u, pos_v, neg_v = self.get_words_embeds(pos_u, pos_v, neg_v)
        embed_u, embed_v, neg_embed_v = self.get_average_embedings(pos_u, pos_u_lens, pos_v, pos_v_lens, neg_v, neg_v_lens)

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        sum_log_sampled = F.logsigmoid(-1 * neg_score)
        sum_log_sampled = torch.sum(sum_log_sampled, dim=1)
        loss = log_target + sum_log_sampled
        return -1 * torch.mean(loss)

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


'''
1st version: GRU encoder--->use the last hidden state as a sentence encoding

2nd version: GRU encoder--->with max pooling and residuals at each timestep,
             2 versions...1-you can leave the padding/2-you can remove the zero padding with max-pooling
             
3nd version: GRU encoder--->with residuals at each timestep and self attention over the output of the bigru
'''


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_size, neg_sample_num, batch_size, window_size, scale=1e-4,
                 max_norm=1):
        super(GRUEncoder, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.neg_sample_num = neg_sample_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.hidden_size = config.hidden_size
        self.nlayers = config.n_layers
        self.bidirectional = config.bidirectional
        self.max_pad = config.max_pad
        self.gru_encoder = config.gru_encoder
        # self.logger, self.handler = init_logger(handler)

        if self.gru_encoder == 3:
            self.attention = SelfAttention(self.hidden_size)

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.dropout = config.dropout
        self.scale = scale
        self.max_norm = max_norm
        self.the_rnn = nn.GRU(input_size=self.embedding_dim,
                              hidden_size=config.hidden_size,
                              num_layers=self.nlayers,
                              bidirectional=self.bidirectional,
                              bias=True,
                              dropout=self.dropout,
                              batch_first=True)

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

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers * self.num_directions, batch_size, self.rnn_size)

    def fix_input(self, phr_inds=None, pos_inds=None, neg_inds=None):
        if self.training:
            seq_lengths_phr = torch.LongTensor([len(seq) for seq in phr_inds])
            seq_lengths_pos = torch.LongTensor([len(seq) for seq in pos_inds])
            seq_lengths_neg = torch.LongTensor([len(seq) for seq in neg_inds])

            seq_phr, phr_lengths, phr_perm = self.pad_sequences(phr_inds, seq_lengths_phr)
            seq_pos, pos_lengths, pos_perm = self.pad_sequences(pos_inds, seq_lengths_pos)
            seq_neg, neg_lengths, neg_perm = self.pad_sequences(neg_inds, seq_lengths_neg)

            if torch.cuda.is_available():
                return seq_phr.cuda(), \
                       phr_lengths, \
                       phr_perm.cuda(), \
                       seq_pos.cuda(), \
                       pos_lengths, \
                       pos_perm.cuda(), \
                       seq_neg.cuda(), \
                       neg_lengths, \
                       neg_perm.cuda()
            else:
                return seq_phr, \
                       phr_lengths, \
                       phr_perm, \
                       seq_pos, \
                       pos_lengths, \
                       pos_perm, \
                       seq_neg, \
                       neg_lengths, \
                       neg_perm
        else:
            # when we evaluate we feed one batch of phrases
            seq_lengths_phr = torch.LongTensor([len(seq) for seq in phr_inds])
            seq_phr, phr_lengths, phr_perm = self.pad_sequences(phr_inds, seq_lengths_phr)
            if torch.cuda.is_available():
                return seq_phr.cuda(), \
                       phr_lengths, \
                       phr_perm.cuda()
            else:
                return seq_phr, \
                       phr_lengths, \
                       phr_perm

    def pad_sequences(self, vectorized_seqs, seq_lengths):

        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

        # Sort tensors by their length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, seq_lengths, perm_idx

    def get_words_embeds(self, phr, pos, neg):
        phr = self.u_embeddings(phr)
        pos = self.v_embeddings(pos)
        neg = self.v_embeddings(neg)
        return phr, pos, neg

    def encode(self, inp, seq_lens=None, perm_idx=None):
        if self.gru_encoder == 1:
            # it is a uni-directional gru
            # Handling padding in Recurrent Networks
            gru_input = pack_padded_sequence(inp, seq_lens.data.cpu().numpy(), batch_first=True)

            hn = self.the_rnn(gru_input)[1].squeeze(0)  # get the last hidden states for the batch..we must unsort it

            # unsort hidden and return last timesteps
            hn = unsort(hn, perm_idx, 0)

            return hn, None
        elif self.gru_encoder == 2:
            # Handling padding in Recurrent Networks
            gru_input = pack_padded_sequence(inp, seq_lens.data.cpu().numpy(), batch_first=True)
            output = self.the_rnn(gru_input)[0]

            # Unpack and pad
            output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

            # # residual-- it will help to learn better word embeddings
            out_forward = output[:, :, :self.hidden_size]
            out_backward = output[:, :, self.hidden_size:]
            output = out_forward + out_backward
            # residual
            output = output + inp

            # Un-sort by length to get the original ordering
            output = unsort(output, perm_idx, 0)

            # Pooling
            # apply max-pooling
            # 2 versions...1-you can leave the padding/2-you can ignore the zero padding while max-pooling(works better)
            if not self.max_pad:
                output[output == 0] = -1e9

            emb, indxs = torch.max(output, 1)

            return emb, indxs
        else:
            # Handling padding in Recurrent Networks
            gru_input = pack_padded_sequence(inp, seq_lens.data.cpu().numpy(), batch_first=True)
            output = self.the_rnn(gru_input)[0]

            # Unpack and pad
            output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

            # # residual-- it will help to learn better word embeddings
            out_forward = output[:, :, :self.hidden_size]
            out_backward = output[:, :, self.hidden_size:]
            output = out_forward + out_backward
            # residual
            output = output + inp

            # attention
            output, attention = self.attention(output, seq_lens)

            # unsort both output and attention
            output = unsort(output, perm_idx, 0)
            attention = unsort(attention, perm_idx, 0)

            return output, attention

    def get_rnn_representation(self, phr, phr_lens, phr_perm, pos, pos_lens, pos_perm, neg, neg_lens, neg_perm):
        phr, _ = self.encode(phr, phr_lens, phr_perm)
        pos, _ = self.encode(pos, pos_lens, pos_perm)
        neg, _ = self.encode(neg, neg_lens, neg_perm)
        neg = neg.view(phr.shape[0], self.neg_sample_num, -1)

        return phr, pos, neg

    def get_loss(self, phr_emb, context_emb, neg_emb):
        score = torch.mul(phr_emb, context_emb)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score)

        if self.neg_sample_num == 1:
            neg_score = torch.mul(phr_emb, neg_emb)
            neg_score = torch.sum(neg_score, dim=1)
            sum_log_sampled = F.logsigmoid(-1 * neg_score)
        else:
            neg_score = torch.bmm(neg_emb, phr_emb.unsqueeze(2)).squeeze()
            sum_log_sampled = F.logsigmoid(-1 * neg_score)
            sum_log_sampled = torch.sum(sum_log_sampled, dim=1)

        loss = log_target + sum_log_sampled
        return -1 * torch.mean(loss)
        
    def forward(self, phr_inds=None, pos_inds=None, neg_inds=None):
        if self.training:
            phr, phr_lens, phr_perm, pos, pos_lens, pos_perm, neg, neg_lens, neg_perm = self.fix_input(phr_inds,
                                                                                                       pos_inds,
                                                                                                       neg_inds)
            phr, pos, neg = self.get_words_embeds(phr, pos, neg)
            phr, pos, neg = self.get_rnn_representation(phr,
                                                        phr_lens,
                                                        phr_perm,
                                                        pos,
                                                        pos_lens,
                                                        pos_perm,
                                                        neg,
                                                        neg_lens,
                                                        neg_perm)
            loss = self.get_loss(phr, pos, neg)

            return loss
        else:  # here it is used for inference---> you can encode one sentence or a batch
            emb = self.inference(phr_inds, concat=True)
            return emb

    def inference(self, phrases, concat=False):
        phr, phr_lengths, phr_perm = self.fix_input(phr_inds=phrases)
        ###
        phr_emb_u = self.u_embeddings(phr)
        phr_emb_v = self.v_embeddings(phr)
        ###
        if config.gru_encoder == 1:
            phr_emb_u, _ = self.encode(phr_emb_u, phr_lengths, phr_perm)
            phr_emb_v, _ = self.encode(phr_emb_v, phr_lengths, phr_perm)

            if concat:
                phr_emb = torch.cat((phr_emb_u, phr_emb_v), dim=1)
            else:
                phr_emb = (phr_emb_u + phr_emb_v) / 2
            return phr_emb
        else:
            phr_emb_u, idx_u = self.encode(phr_emb_u, phr_lengths, phr_perm)
            phr_emb_v, idx_v = self.encode(phr_emb_v, phr_lengths, phr_perm)

            if concat:
                phr_emb = torch.cat((phr_emb_u, phr_emb_v), dim=1)
            else:
                phr_emb = (phr_emb_u + phr_emb_v) / 2
            return phr_emb, idx_u, idx_v

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


class SelfAttention(nn.Module):
    ### from https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4

    def __init__(self, attention_size):
        super(SelfAttention, self).__init__()

        self.attention_size = attention_size
        self.attn_weights = nn.Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.Tanh()

        nn.init.uniform_(self.attn_weights.data, -0.005, 0.005)

    def get_mask(self, attns, lens):
        max_len = max(lens.data)
        mask = Variable(torch.ones(attns.size())).detach()

        if attns.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lens.data):  # skip the first sentence because it is the max.
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inp, lengths):
        # inp is the output of the bigru
        # inp ---> B x S x hidden_dim
        scores = self.activation(inp.matmul(self.attn_weights))
        scores = self.softmax(scores)

        # now we have activated the padded elements too..so we must create a mask
        # lengths contain the sequences lengths in the batch
        mask = self.get_mask(scores, lengths)

        # apply the mask
        masked_scores = scores * mask

        # re-normalize the masked scores because we zeroed out some activations(pads)
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        # multiply each hidden state with the attention weights --- dot product
        weighted = torch.mul(inp, scores.unsqueeze(-1).expand_as(inp))
        representations = weighted.sum(1).squeeze()

        return representations, scores
