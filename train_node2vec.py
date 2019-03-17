from pprint import pprint
import torch
import re
from torch.autograd import Variable
import torch.optim as optim
import time
from utils import Utils
import models
from torch.utils.data import DataLoader
from dataloader import Node2VecDataset
import numpy as np
from tqdm import tqdm
import pickle
import codecs
import os
from random import shuffle
import itertools
import json
import seaborn as sns
from collections import Counter
from collections import OrderedDict
import logging
import webbrowser
import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

use_cuda = torch.cuda.is_available()
sns.set()
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                            t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                          '').strip().lower()).split()


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
    p = [get_index(t, word_vocab) for t in phr]
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


output_dir = '/home/sotiris/Documents/logger/'


def init_logger(handler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    od = output_dir.split('/')[-1]
    logger = logging.getLogger(od)
    if handler is not None:
        logger.removeHandler(handler)
    handler = logging.FileHandler(os.path.join(output_dir, 'model_test.log'))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler


# in the paper they get edge embeddings..they use a lot of methods but for link prediction they state that
# the Hadamard product is highly stable and gives the best performance.
def get_edge_embeddings(edge_list, node_embeddings, model_type, phrase_dic):
    # create a list containing edge embeddings
    if model_type == 'average':
        edge_embeddings = []
        for idx, edge in enumerate(edge_list):
            phrase_node1 = get_average_embedding(phrase_dic[edge[0]], node_embeddings)
            phrase_node2 = get_average_embedding(phrase_dic[edge[1]], node_embeddings)
            hadamard = np.multiply(phrase_node1, phrase_node2)
            edge_embeddings.append(hadamard)
        edge_embeddings = np.array(edge_embeddings)
        return edge_embeddings
    else:
        edge_embeddings = []
        for edge in edge_list:
            emb_node1 = node_embeddings[edge[0]]
            emb_node2 = node_embeddings[edge[1]]
            hadamard = np.multiply(emb_node1, emb_node2)
            edge_embeddings.append(hadamard)
        edge_embeddings = np.array(edge_embeddings)
        return edge_embeddings


def get_average_embedding(phrase, node_embeddings):
    length = len(phrase)
    # TODO check if i have to remove split
    # phrase = phrase.split()
    for idx, word in enumerate(phrase):
        if idx == 0:
            sum = node_embeddings[word]
        else:
            sum = np.add(sum, node_embeddings[word])

    average_embedding = np.divide(sum, float(length))
    return average_embedding


def load_embeddings(file):
    node_embeddings = {}
    with codecs.open("{}".format(file), 'r', 'utf-8') as embeddings:
        embeddings.readline()
        for i, line in enumerate(embeddings):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            assert len(embedding) == 50
            node_embeddings[word] = embedding
    return node_embeddings


class Node2Vec:
    def __init__(self, walks, output_file, walk_length, embedding_dim=128, rnn_size=50, epochs=10, batch_size=32,
                 window_size=10,
                 neg_sample_num=5):
        self.utils = Utils(walks, window_size, walk_length)
        if walks is not None:
            self.vocabulary_size = self.utils.vocabulary_size
            self.node2phr = self.utils.phrase_dic
            self.word2idx = self.utils.word2idx
        self.embedding_dim = embedding_dim
        self.rnn_size = rnn_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_sample_num = neg_sample_num
        self.odir_checkpoint = config.checkpoint_dir
        self.odir_embeddings = config.embeddings_dir
        self.output_file = output_file
        self.model_type = config.model
        self.wv = {}

    def train(self):
        # initialize the model
        if self.model_type == 'rnn':
            model = models.GRUEncoder(self.vocabulary_size,
                                      self.embedding_dim,
                                      self.rnn_size,
                                      self.neg_sample_num,
                                      self.batch_size,
                                      self.window_size)
        elif self.model_type == 'average':
            model = models.AverageNode2Vec(self.vocabulary_size,
                                           self.embedding_dim,
                                           self.neg_sample_num,
                                           self.batch_size,
                                           self.window_size)
        else:
            model = models.BaselineNode2Vec(self.vocabulary_size,
                                            self.embedding_dim)

        print_params(model)
        params = model.parameters()
        if use_cuda:
            print('GPU available!!')
            model.cuda()

        if self.model_type == 'rnn':
            optimizer = optim.Adam(params, lr=config.lr)
        else:
            optimizer = optim.SparseAdam(params, lr=config.lr)

        dataset = Node2VecDataset(self.utils, self.neg_sample_num)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False)

        for epoch in range(self.epochs):
            batch_num = 0
            last_batch_num = -1
            batch_costs = []
            # if we resume training load the last checkpoint
            if config.resume_training:
                if use_cuda:
                    print('GPU available..will resume training!!')
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')

                modelcheckpoint = torch.load(config.checkpoint_to_load, map_location=device)
                model.load_state_dict(modelcheckpoint['state_dict'])
                optimizer.load_state_dict(modelcheckpoint['optimizer'])
                last_batch_num = modelcheckpoint['batch_num']
            #
            model.train()
            for sample in tqdm(dataloader):
                # if we resume training--continue from the last batch we stopped
                if batch_num <= last_batch_num:
                    batch_num += 1
                    continue
                ###
                phr = sample['center']
                pos_context = sample['context']
                neg_v = np.random.choice(self.utils.sample_table, size=(len(phr) * self.neg_sample_num)).tolist()
                ###
                if self.model_type is 'rnn' or self.model_type is 'average':
                    ###
                    phr = [phr2idx(self.utils.phrase_dic[int(phr_id)], self.word2idx) for phr_id in phr]
                    pos_context = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in pos_context]
                    neg_v = [phr2idx(self.utils.phrase_dic[int(item)], self.word2idx) for item in neg_v]
                    ###
                # --------------
                optimizer.zero_grad()
                loss = model(phr, pos_context, neg_v)
                loss.backward()
                optimizer.step()
                batch_costs.append(loss.cpu().item())
                # --------------

                # print the average cost every 5000 batches
                if batch_num % 5000 == 0:
                    print('Batches Average Loss: {}, Batches: {} '.format(
                        sum(batch_costs) / float(len(batch_costs)),
                        batch_num))
                    batch_costs = []
                ###

                # save the model every 500000 batches
                if batch_num % 500000 == 0:
                    print("Saving at {} batches".format(batch_num))
                    state = {'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'word2idx': self.word2idx,
                             'idx2word': self.utils.idx2word,
                             'batch_num': batch_num}
                    save_checkpoint(state,
                                    filename=self.odir_checkpoint + 'isa_gru_checkpoint_batch_{}'.format(batch_num))
                ###
                batch_num += 1

            # reset the yielder on the dataset class
            if epoch + 1 != self.epochs:
                dataset.reset_generator()

            # save the model on each epoch
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'word2idx': self.word2idx,
                     'idx2word': self.utils.idx2word}

            save_checkpoint(state, filename=self.odir_checkpoint + config.checkpoint_name.format(epoch + 1))
            # TODO do something better here
            config.checkpoint_name = config.checkpoint_name.format(epoch + 1)

        # training has finished..save the word embeddings
        print("Optimization Finished!")
        self.wv = model.save_embeddings(file_name=self.odir_embeddings + self.output_file,
                                        idx2word=self.utils.idx2word,
                                        use_cuda=True)

    def eval(self, train_pos, train_neg, test_pos, test_neg, embeddings_file=None, checkpoint_file=None):
        phrase_dic = clean_dictionary(pickle.load(open(config.phrase_dic, 'rb')))
        if self.model_type == 'rnn':
            if use_cuda:
                print('GPU available!!')
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            modelcheckpoint = torch.load(checkpoint_file, map_location=device)
            vocabulary_size = len(modelcheckpoint['word2idx'])
            model = models.GRUEncoder(vocabulary_size,
                                      self.embedding_dim,
                                      self.rnn_size,
                                      self.neg_sample_num,
                                      self.batch_size,
                                      self.window_size)
            print_params(model)
            #
            if use_cuda:
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
            word2idx = modelcheckpoint['word2idx']
            node_embeddings = self.create_node_embeddings(model, phrase_dic, word2idx)

        else:
            node_embeddings = load_embeddings(embeddings_file)

        train_pos_edge_embs = get_edge_embeddings(train_pos, node_embeddings, self.model_type, phrase_dic)
        train_neg_edge_embs = get_edge_embeddings(train_neg, node_embeddings, self.model_type, phrase_dic)
        train_set = np.concatenate([train_pos_edge_embs, train_neg_edge_embs])

        # labels: 1-> link exists, 0-> false edge
        train_labels = np.zeros(len(train_set))
        train_labels[:len(train_pos_edge_embs)] = 1

        # for testing
        test_pos_edge_embs = get_edge_embeddings(test_pos, node_embeddings, self.model_type, phrase_dic)
        test_neg_edge_embs = get_edge_embeddings(test_neg, node_embeddings, self.model_type, phrase_dic)
        test_set = np.concatenate([test_pos_edge_embs, test_neg_edge_embs])
        # test_set_phrases = test_pos + test_neg

        # labels: 1-> link exists, 0-> false edge
        test_labels = np.zeros(len(test_set))
        test_labels[:len(test_pos_edge_embs)] = 1
        # test_labels_phrases = np.zeros(len(test_set_phrases))
        # test_labels_phrases[:len(test_pos)] = 1

        # train the classifier and evaluate in the test set

        # shuffle train set
        idx_list = [i for i in range(len(train_labels))]
        shuffle(idx_list)
        train_set = train_set[idx_list]
        train_labels = train_labels[idx_list]

        # shuffle test set
        idx_list = [i for i in range(len(test_labels))]
        shuffle(idx_list)
        test_set = test_set[idx_list]
        # test_set_phrases = test_set_phrases[idx_list]
        test_labels = test_labels[idx_list]
        # test_labels_phrases = test_labels_phrases[idx_list]

        classifier = LogisticRegression()
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
            json_list_ultra = []
            keys = list(phrase_dic.keys())
            for phridx in tqdm(range(0, len(keys), self.batch_size)):
                batch = keys[phridx:phridx + self.batch_size]
                phrases = [phrase_dic[key] for key in batch]
                phr = [phr2idx(phrase, word2idx) for phrase in phrases]
                phrase_emb, idx_u, idx_v = model(phr)

                ###  create weights for visualizing the words that are getting pooled more often
                # json_list = create_pooling_weights_for_batch(idx_u, phrase_dic, batch)
                json_list = create_attention_weights_for_batch(idx_u, phrase_dic, batch)
                for triplet in json_list:
                    json_list_ultra.append(triplet)
                ###

                for idx, phr_id in enumerate(batch):
                    phrase = phrase_dic[phr_id]
                    phrase = ' '.join(phrase)
                    ###
                    if use_cuda:
                        node_embeddings[phr_id] = phrase_emb[idx].cpu().numpy()
                        node_embeddings_phrases[phrase] = phrase_emb[idx].cpu().numpy()
                        e = ' '.join(map(lambda x: str(x), phrase_emb[idx].cpu().numpy()))
                    else:
                        node_embeddings[phr_id] = phrase_emb[idx].numpy()
                        node_embeddings_phrases[phrase] = phrase_emb[idx].numpy()
                        e = ' '.join(map(lambda x: str(x), phrase_emb[idx].numpy()))
                    ###
                    fout.write('%s %s\n' % (phrase, e))

            with open("{}.p".format('node_embeddings_phrases'), 'wb') as dump_file:
                pickle.dump(node_embeddings_phrases, dump_file)

            # with open("json_pooling_weights.json", 'w') as fp:
            #     json.dump(json_list_ultra, fp)

            #### create html file with the heatmap of each phrase
            plot_attention(json_list_ultra, 'attention_part_of.html')
            ####
            return node_embeddings


def create_pooling_weights_for_batch(idx_u, phrase_dic, batch):
    counts = []
    for idx in idx_u:
        counts.append(Counter(idx.numpy().tolist()))
    json_list = []
    for idx, phr_id in enumerate(batch):
        freqs = counts[idx]
        sum_val = sum(list(freqs.values()))
        keys = list(freqs.keys())
        phrase = phrase_dic[phr_id]
        phrase_str = ' '.join(phrase)
        scores = {}
        keys = sorted(keys)
        i = 0
        for key in keys:
            if phrase[key] in scores.keys():
                key_phr = str(i) + "-" + phrase[key]
                scores[key_phr] = float(freqs[key]) / sum_val
                i += 1
            else:
                scores[phrase[key]] = float('{0:.4f}'.format(float(freqs[key]) / sum_val))
        json_list.append((phrase_str, scores))
    return json_list


def create_attention_weights_for_batch(idx_u, phrase_dic, batch):
    json_list = []
    scores = {}
    for idx, phr_id in enumerate(batch):
        phrase = phrase_dic[phr_id]
        phrase_str = ' '.join(phrase)
        attn_phrase = idx_u[idx]
        i = 0
        for idx_word, word in enumerate(phrase):
            if word in scores.keys():
                double_phr = str(i) + "-" + word
                scores[double_phr] = attn_phrase[idx_word]
            else:
                scores[word] = attn_phrase[idx_word]

        json_list.append((phrase_str, scores))

    return json_list


def create_confusion_matrix(preds, phrase_dic, test_labels_phrases, test_set_phrases):
    json_list_false_negative = []
    json_list_false_positive = []
    json_list_true_negative = []
    json_list_true_positive = []
    for idx, pred in enumerate(preds):
        if pred[0] > pred[1] and test_labels_phrases[idx] == 1:
            edge = test_set_phrases[idx]
            phrase1 = " ".join(phrase_dic[edge[0]])
            phrase2 = " ".join(phrase_dic[edge[1]])
            json_list_false_negative.append(OrderedDict([("phrase1: ", str(phrase1)), ("phrase2: ", str(phrase2))]))
        elif pred[0] < pred[1] and test_labels_phrases[idx] == 0:
            edge = test_set_phrases[idx]
            phrase1 = " ".join(phrase_dic[edge[0]])
            phrase2 = " ".join(phrase_dic[edge[1]])
            json_list_false_positive.append(OrderedDict([("phrase1: ", str(phrase1)), ("phrase2: ", str(phrase2))]))
        elif pred[0] < pred[1] and test_labels_phrases[idx] == 1:
            edge = test_set_phrases[idx]
            phrase1 = " ".join(phrase_dic[edge[0]])
            phrase2 = " ".join(phrase_dic[edge[1]])
            json_list_true_positive.append(OrderedDict([("phrase1: ", str(phrase1)), ("phrase2: ", str(phrase2))]))
        elif pred[0] > pred[1] and test_labels_phrases[idx] == 0:
            edge = test_set_phrases[idx]
            phrase1 = " ".join(phrase_dic[edge[0]])
            phrase2 = " ".join(phrase_dic[edge[1]])
            json_list_true_negative.append(OrderedDict([("phrase1: ", str(phrase1)), ("phrase2: ", str(phrase2))]))

    with open("json_false_positive.json", 'w') as fp:
        json.dump(json_list_false_positive, fp)

    with open("json_false_negative.json", 'w') as fp:
        json.dump(json_list_false_negative, fp)

    with open("json_true_positive.json", 'w') as fp:
        json.dump(json_list_true_positive, fp)

    with open("json_true_negative.json", 'w') as fp:
        json.dump(json_list_true_negative, fp)


def plot_attention(json_list, filename):
    html_content = '<!DOCTYPE html><html><head> ' \
                   '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">' \
                   '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>' \
                   '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>' \
                   '<style type="text/css">body { padding: 10px}span { border: 0px solid;}</style> </head>' \
                   '<body>'

    html_content += '<ul>'
    for words, attention in json_list:
        html_content += '<li>'
        for word in words.split():
            try:
                html_content += '<span style= "background-color:rgba(255, 0, 0, {0:.1f});">{1} </span>'.format(
                    attention[word], word)
            except KeyError:
                html_content += '<span>{} </span>'.format(word)
        html_content += '<br/>'
        html_content += '</li>'
    html_content += '</ul>'
    html_content += '</body>'
    path = os.path.abspath(filename)
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(html_content)
    webbrowser.open(url)
