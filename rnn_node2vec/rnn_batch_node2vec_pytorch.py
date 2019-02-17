from pprint import pprint

import torch
import re
from torch.autograd import Variable
import torch.optim as optim
import time
from rnn_node2vec_utils import Utils
# from rnn_batch_skipgram import node2vec_rnn
from rnn_batch_skipgram_ver2 import node2vec_rnn
from torch.utils.data import DataLoader
from rnn_dataloader import Node2VecDataset
import numpy as np
from tqdm import tqdm
import pickle
import codecs
import os
import config
# from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

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


def clean_dictionary(phrase_dic):
    for nodeid, phrase in phrase_dic.items():
        phrase_dic[nodeid] = tokenize(phrase)
    return phrase_dic


def print_params(model):
    print(40 * '=')
    print(model)
    print(40 * '=')
    total_params = 0
    for parameter in model.parameters():
        print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')


class Node2Vec:
    def __init__(self, walks, output_file, walk_length, embedding_dim=128, rnn_size=50, epochs=10, batch_size=32, window_size=10,
                 neg_sample_num=5):
        self.utils = Utils(walks, window_size, walk_length)
        if walks is not None:
            self.vocabulary_size = self.utils.vocabulary_size
            self.node2phr = self.utils.phrase_dic
            self.word2idx = self.utils.word2idx
        # self.word2idx = self.create_word2idx()
        self.embedding_dim = embedding_dim
        self.rnn_size = rnn_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        self.odir_checkpoint = config.checkpoint_dir
        self.odir_embeddings = config.embeddings_dir
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

        optimizer = optim.Adam(params, lr=config.lr)
        dataset = Node2VecDataset(self.utils, self.neg_sample_num)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False)

        for epoch in range(self.epochs):
            batch_num = 0
            batch_costs = []
            model.train()
            for sample in tqdm(dataloader):
                center = sample['center']
                context = sample['context']
                neg_v = np.random.choice(self.utils.sample_table, size=(len(center) * self.neg_sample_num)).tolist()
                #
                phr = [phr2idx(self.utils.phrase_dic[int(phr)], self.word2idx) for phr in center]
                pos_context = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in context]
                neg_v = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in neg_v]
                #
                optimizer.zero_grad()
                loss = model(phr, pos_context, neg_v)
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())
                #
                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {} '.format(
                        sum(batch_costs) / float(len(batch_costs)),
                        batch_num))
                    batch_costs = []
                batch_num += 1
            print()
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'word2idx': self.word2idx,
                     'idx2word': self.utils.idx2word}
            save_checkpoint(state,
                            filename=self.odir_checkpoint + config.checkpoint_name.format(epoch + 1))
            self.utils.stop = True
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file,
                                        idx2word=self.utils.idx2word,
                                        use_cuda=True)

    def eval(self, train_pos, train_neg, test_pos, test_neg):
        if torch.cuda.is_available():
            print('GPU available!!')
            device = torch.device('gpu')
        else:
            device = torch.device('cpu')

        modelcheckpoint = torch.load('/home/sotiris/Downloads/part_of_rnn_test_new_lr_checkpoint_epoch_1.pth.tar',
                                     map_location=device)
        vocabulary_size = len(modelcheckpoint['word2idx'])

        model = node2vec_rnn(vocabulary_size, self.embedding_dim, self.rnn_size, self.neg_sample_num,
                             self.batch_size,
                             self.window_size)
        print_params(model)
        params = model.parameters()

        if torch.cuda.is_available():
            print('GPU available!!')
            model.cuda()
        #
        model.eval()
        model.load_state_dict(modelcheckpoint['state_dict'])
        #
        print('Number of positive training samples: ', len(train_pos))
        print('Number of negative training samples: ', len(train_neg))
        print('Number of positive testing samples: ', len(test_pos))
        print('Number of negative testing samples: ', len(test_neg))
        phrase_dic = clean_dictionary(pickle.load(
            open(config.phrase_dic, 'rb')))
        word2idx = modelcheckpoint['word2idx']
        node_embeddings = self.create_node_embeddings(model, phrase_dic, word2idx)

        train_pos_edge_embs = self.get_edge_embeddings(train_pos, node_embeddings, phrase_dic)
        train_neg_edge_embs = self.get_edge_embeddings(train_neg, node_embeddings, phrase_dic)
        train_set = np.concatenate([train_pos_edge_embs, train_neg_edge_embs])

        # labels: 1-> link exists, 0-> false edge
        train_labels = np.zeros(len(train_set))
        train_labels[:len(train_pos_edge_embs)] = 1

        # for testing
        test_pos_edge_embs = self.get_edge_embeddings(test_pos, node_embeddings, phrase_dic)
        test_neg_edge_embs = self.get_edge_embeddings(test_neg, node_embeddings, phrase_dic)
        test_set = np.concatenate([test_pos_edge_embs, test_neg_edge_embs])

        # labels: 1-> link exists, 0-> false edge
        test_labels = np.zeros(len(test_set))
        test_labels[:len(test_pos_edge_embs)] = 1

        # train the classifier and evaluate in the test set
        classifier = LogisticRegression(random_state=0)
        classifier.fit(train_set, train_labels)

        # evaluate
        test_preds = classifier.predict_proba(test_set)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, test_preds)
        test_auc = auc(false_positive_rate, true_positive_rate)
        test_roc = roc_auc_score(test_labels, test_preds)
        print('node2vec Test ROC score: ', str(test_roc))
        print('node2vec Test AUC score: ', str(test_auc))

    def create_node_embeddings(self, model, phrase_dic, word2idx):
        with torch.no_grad():
            file_name = 'rnn_inference_phrases_with_names.emb'
            node_embeddings = {}
            node_embeddings_phrases = {}
            fout = open(file_name, 'w')
            fout.write('%d %d\n' % (len(word2idx), self.embedding_dim))
            for phr_id in tqdm(phrase_dic.keys()):
                phrase = phrase_dic[phr_id]
                phr = [phr2idx(phrase, word2idx)]
                phrase_emb = model(phr).squeeze(0)
                node_embeddings[phr_id] = phrase_emb.numpy()
                node_embeddings_phrases[phrase] = phrase_emb.numpy()
                e = ' '.join(map(lambda x: str(x), phrase_emb.numpy()))
                fout.write('%s %s\n' % (phrase, e))
            print(len(node_embeddings))
            with open("{}.p".format('node_embeddings_phrases'), 'wb') as dump_file:
                pickle.dump(node_embeddings_phrases, dump_file)
            return node_embeddings

    def get_edge_embeddings(self, edge_list, node_embeddings, phrase_dic):
        # create a list containing edge embeddings
        edge_embeddings = []
        for edge in edge_list:
            emb_node1 = node_embeddings[edge[0]]
            emb_node2 = node_embeddings[edge[1]]
            hadamard = np.multiply(emb_node1, emb_node2)
            edge_embeddings.append(hadamard)
        edge_embeddings = np.array(edge_embeddings)
        return edge_embeddings
